#!/usr/bin/env python3
"""
20-year MODIS cloud embedding CCA analysis — Nautilus job script.

Reads pre-downloaded embeddings from /workspace/embeddings/ (written by download.py),
matches each tile to ERA5 SST and EIS via ARCO-ERA5 on GCS (public, anonymous),
runs deconfounded PCA -> CCA(1), and saves:
  - /workspace/results/cca_results.txt  : r values for all runs
  - /workspace/results/walk_vae_sst.png : decoded VAE latent walk along SST direction
  - /workspace/results/walk_vae_eis.png : decoded VAE latent walk along EIS direction
                                          (only if EIS data available and r > threshold)

The decoded walk requires the VAE checkpoint at $CHECKPOINT.
If the file is absent the walk step is skipped and only r values are reported.

Environment variables (all have defaults):
  EMBED_DIR     path to downloaded embeddings   default: /workspace/embeddings
  OUT_DIR       where to write results           default: /workspace/results
  CHECKPOINT    VAE .pt checkpoint path          default: /workspace/vae_checkpoint/lightning_model_50_transform.pt
  MANIFEST      manifest.csv path               default: /workspace/repo/manifest.csv
  STREAM_STRIDE sample every N-th day            default: 11  (~520 days across 2002-2022)
  MAX_PER_DAY   max tiles to keep per day        default: 200
"""

import json
import os
import warnings
from pathlib import Path

import gcsfs
import matplotlib
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────
EMBED_DIR     = os.environ.get("EMBED_DIR",     "/workspace/embeddings")
OUT_DIR       = os.environ.get("OUT_DIR",       "/workspace/results")
CHECKPOINT    = os.environ.get("CHECKPOINT",    "/workspace/vae_checkpoint/lightning_model_50_transform.pt")
MANIFEST      = os.environ.get("MANIFEST",      "/workspace/repo/manifest.csv")
STREAM_STRIDE = int(os.environ.get("STREAM_STRIDE", 11))
MAX_PER_DAY   = int(os.environ.get("MAX_PER_DAY",   200))

# ARCO-ERA5 on Google Cloud Storage — public, no auth needed.
# Contains: sea_surface_temperature, temperature (pressure levels),
#           specific_humidity, geopotential, vertical_velocity,
#           total_column_water_vapour, 2m_temperature
ARCO_ERA5 = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1_zarr-v3.zarr"

N_PCA_VAE    = 50
N_PCA_T2V    = 49
N_WALK_STEPS = 9
WALK_SIGMA   = 1.5

os.makedirs(OUT_DIR, exist_ok=True)


# ── ERA5 setup ─────────────────────────────────────────────────────────────
def open_era5():
    print("Opening ARCO-ERA5 (GCS, anonymous)...")
    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(gcsfs.GCSMap(ARCO_ERA5, gcs=fs), consolidated=True)
    lats = ds["latitude"].values
    lons = ds["longitude"].values
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    tree = cKDTree(np.column_stack([lat_grid.ravel(), lon_grid.ravel()]))
    print(f"  ERA5 grid: {len(lons)} lon x {len(lats)} lat")
    print(f"  ERA5 time: {str(ds['time'].values[0])[:10]} -> {str(ds['time'].values[-1])[:10]}")
    return ds, tree


def _match_var(ds, tree, df, varname, level=None):
    """Match a single ERA5 variable to a dataframe of (lat, lon, time) rows."""
    lats     = df["lat"].values
    lons     = df["lon"].values
    times    = pd.to_datetime(df["time"])
    lons_pos = np.where(lons < 0, lons + 360, lons)   # ERA5 uses 0-360
    _, sp_idx = tree.query(np.column_stack([lats, lons_pos]))
    out      = np.full(len(df), np.nan)

    da = ds[varname]
    if level is not None:
        da = da.sel(level=level)

    # Load one unique date at a time to stay memory-efficient
    for date in np.unique(times.dt.date):
        mask = times.dt.date == date
        if not mask.any():
            continue
        try:
            day_da    = da.sel(time=str(date)).load()
            day_times = pd.DatetimeIndex(day_da["time"].values)
            flat      = day_da.values.reshape(len(day_times), -1)
            for i in np.where(mask)[0]:
                ti = day_times.get_indexer([times.iloc[i]], method="nearest")[0]
                if ti >= 0:
                    out[i] = float(flat[ti, sp_idx[i]])
        except Exception as e:
            pass   # missing day in ERA5 — leave NaN
    return out


def match_sst(ds, tree, df):
    raw = _match_var(ds, tree, df, "sea_surface_temperature")
    # ERA5 SST is in Kelvin; convert and mask land (NaN where no SST)
    return np.where(raw > 200, raw - 273.15, np.nan)


def match_eis(ds, tree, df):
    """Estimated Inversion Strength (Wood & Bretherton 2006), in K."""
    Lv = 2.5e6; Rd = 287.0; Rv = 461.0; cp = 1005.0; g = 9.81

    T700   = _match_var(ds, tree, df, "temperature",      level=700)
    T850   = _match_var(ds, tree, df, "temperature",      level=850)
    T1000  = _match_var(ds, tree, df, "temperature",      level=1000)
    q850   = _match_var(ds, tree, df, "specific_humidity", level=850)
    q1000  = _match_var(ds, tree, df, "specific_humidity", level=1000)
    Phi700 = _match_var(ds, tree, df, "geopotential",     level=700)
    T2m    = _match_var(ds, tree, df, "2m_temperature")

    # LTS = theta_700 - theta_1000
    lts = T700 * (1000 / 700) ** 0.286 - T1000

    # Moist adiabatic lapse rate at 850 hPa (K/m)
    e_s  = 6.112e2 * np.exp(17.67 * (T850 - 273.15) / (T850 - 29.65))
    qs   = 0.622 * e_s / (85000 - 0.378 * e_s)
    gm   = (g / cp) * (1 + Lv * qs / (Rd * T850)) / (1 + Lv**2 * qs / (cp * Rv * T850**2))
    gm_km = gm * 1000  # K/km

    z700_km = Phi700 / (g * 1000)

    e_sfc   = np.clip(q1000 * 1013.25e2 / (0.622 + q1000), 10, 5000)
    T_D     = (243.5 * np.log(e_sfc / 611.2) / (17.67 - np.log(e_sfc / 611.2))) + 273.15
    z_lcl   = np.clip(0.125 * (T2m - T_D), 0, 3)  # km

    return lts - gm_km * (z700_km - z_lcl)


# ── Local embedding loader ──────────────────────────────────────────────────
def load_day(year, month, day):
    """
    Load one day's embeddings and metadata from the local PVC.
    Returns a DataFrame with columns: lat, lon, time, vae (2048-D array), t2v (50-D array).
    Returns None if any required file is missing, corrupt, or inconsistent.
    """
    prefix   = f"{year}_{month:02d}_{day:02d}"
    day_dir  = Path(EMBED_DIR) / str(year) / f"{month:02d}" / f"{day:02d}"
    mean_p   = day_dir / f"{prefix}_mean.npy"
    t2v_p    = day_dir / f"{prefix}_tile2vec.npy"
    meta_p   = day_dir / f"{prefix}_centers.json"

    if not (mean_p.exists() and t2v_p.exists() and meta_p.exists()):
        return None

    try:
        vae_arr = np.load(mean_p).squeeze()    # (N, 2048)
        t2v_arr = np.load(t2v_p).squeeze()    # (N, 50)
    except Exception as e:
        print(f"  [skip] {prefix}: corrupt npy ({e})")
        return None

    try:
        with open(meta_p) as f:
            meta = json.load(f)
    except Exception as e:
        print(f"  [skip] {prefix}: corrupt json ({e})")
        return None

    rows, idx = [], 0
    for ts in sorted(meta.keys()):
        for lat, lon in meta[ts]:
            rows.append({"lat": float(lat), "lon": float(lon),
                         "time": pd.Timestamp(ts)})
            idx += 1

    if idx == 0:
        print(f"  [skip] {prefix}: empty centers json")
        return None

    # Truncate to the minimum consistent length across all three files
    n = min(idx, len(vae_arr), len(t2v_arr))
    if n < idx:
        print(f"  [warn] {prefix}: array shorter than centers ({n} vs {idx} tiles), truncating")
    rows = rows[:n]
    vae_arr = vae_arr[:n]
    t2v_arr = t2v_arr[:n]

    df = pd.DataFrame(rows)
    df["vae"] = list(vae_arr)
    df["t2v"] = list(t2v_arr)
    return df


# ── CCA pipeline ───────────────────────────────────────────────────────────
def run_cca(X, target, lat, months, n_pca, tag=""):
    """
    Deconfound (lat, lat², sin/cos month) -> standardise -> PCA -> CCA(1).
    Fit on 80%, evaluate Pearson r on held-out 20%.
    Returns dict with r_pearson, r_spearman, physics_scores, X_deconf.
    """
    valid = np.isfinite(target)
    X, target, lat, months = X[valid], target[valid], lat[valid], months[valid]

    C = np.column_stack([lat, lat**2,
                         np.sin(2 * np.pi * months / 12),
                         np.cos(2 * np.pi * months / 12)])
    reg = LinearRegression(fit_intercept=True).fit(C, X)
    X_deconf = X - reg.predict(C)

    sx = StandardScaler(); sy = StandardScaler()
    X_sc = sx.fit_transform(X_deconf)
    Y_sc = sy.fit_transform(target.reshape(-1, 1))

    n_pca_act = min(n_pca, X.shape[1] - 1, X.shape[0] // 10)
    pca = PCA(n_components=n_pca_act, random_state=42)
    X_pca = pca.fit_transform(X_sc)
    print(f"  [{tag}] PCA({n_pca_act}): {pca.explained_variance_ratio_.sum()*100:.1f}% variance retained")

    X_tr, X_te, Y_tr, Y_te = train_test_split(X_pca, Y_sc, test_size=0.2, random_state=42)
    cca = CCA(n_components=1, max_iter=1000).fit(X_tr, Y_tr)
    Xc, Yc = cca.transform(X_te, Y_te)
    pr, _  = pearsonr(Xc.flatten(), Yc.flatten())
    sr, _  = spearmanr(Xc.flatten(), Yc.flatten())
    print(f"  [{tag}] Pearson r = {pr:.4f}   Spearman rho = {sr:.4f}  (20% test set)")

    phys_dir = pca.components_.T @ cca.x_weights_.flatten()
    phys_dir /= np.linalg.norm(phys_dir)

    Xc_all, _ = cca.transform(X_pca, Y_sc)

    return dict(r_pearson=pr, r_spearman=sr,
                physics_dir=phys_dir, scaler_X=sx,
                X_deconf=X_deconf, X_raw=X,
                physics_scores=Xc_all.flatten())


# ── Decoded walk ────────────────────────────────────────────────────────────
def decoded_walk(pipe, var_vals, tag, phys_label, out_path):
    """Walk the VAE decoder along the CCA direction. Requires VAE checkpoint."""
    import torch
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from vae import VAELightningModule

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = VAELightningModule.load_from_checkpoint(CHECKPOINT)
    model.to(device).eval()
    print(f"  VAE loaded on {device}")

    mean_z   = pipe["X_raw"].mean(axis=0)
    phys_dir = pipe["physics_dir"]
    spread   = np.std(pipe["physics_scores"])

    alphas = np.linspace(-WALK_SIGMA, WALK_SIGMA, N_WALK_STEPS) * spread
    # Bin var_vals by physics score to get expected label at each walk step
    score_pct = np.percentile(pipe["physics_scores"], np.linspace(0, 100, N_WALK_STEPS))
    step_labels = np.interp(score_pct,
                            np.sort(pipe["physics_scores"]),
                            np.sort(var_vals[np.isfinite(var_vals)]))

    imgs = []
    with torch.no_grad():
        for alpha in alphas:
            z   = torch.tensor(mean_z + alpha * phys_dir, dtype=torch.float32).unsqueeze(0).to(device)
            out = model.decoder(z).squeeze(0).cpu().numpy()   # (3, H, W)
            grey = np.mean(out, axis=0)
            lo, hi = np.percentile(grey, [2, 98])
            imgs.append(np.clip((grey - lo) / (hi - lo + 1e-8), 0, 1))

    fig, axes = plt.subplots(1, N_WALK_STEPS,
                             figsize=(N_WALK_STEPS * 2.4, 2.8),
                             facecolor="#1a1a1a")
    fig.suptitle(f"VAE decoded walk — {phys_label}  {tag}",
                 color="white", fontsize=11, fontweight="bold")
    for col, (img, lbl) in enumerate(zip(imgs, step_labels)):
        ax = axes[col]
        ax.imshow(img, cmap="bone", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")
        ax.set_title(f"{lbl:.1f}", color="white", fontsize=9, fontweight="bold", pad=3)

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
    print(f"  Saved -> {out_path}")
    plt.close()


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    # 1. Load manifest and sample days
    print("Loading manifest...")
    manifest = pd.read_csv(MANIFEST)
    queue = (manifest[manifest["status"] == "OK"]
             .sort_values(["year", "month", "day"])
             .reset_index(drop=True))
    queue = queue.iloc[::STREAM_STRIDE].reset_index(drop=True)
    print(f"  Targeting {len(queue)} days (stride={STREAM_STRIDE})")

    # 2. Load embeddings from PVC
    print("\nLoading embeddings from PVC...")
    meta_rows, vae_chunks, t2v_chunks = [], [], []
    n_loaded = 0
    for _, row in queue.iterrows():
        df = load_day(int(row["year"]), int(row["month"]), int(row["day"]))
        if df is None:
            continue
        if len(df) > MAX_PER_DAY:
            df = df.sample(MAX_PER_DAY, random_state=42).reset_index(drop=True)
        meta_rows.append(df[["lat", "lon", "time"]])
        vae_chunks.append(np.vstack(df["vae"].values))
        t2v_chunks.append(np.vstack(df["t2v"].values))
        n_loaded += 1
        if n_loaded % 100 == 0:
            print(f"  {n_loaded} days loaded, {sum(len(m) for m in meta_rows):,} tiles so far...")

    print(f"  Loaded {n_loaded} days")
    df_meta  = pd.concat(meta_rows, ignore_index=True)
    X_vae    = np.vstack(vae_chunks).astype("float32")
    X_t2v    = np.vstack(t2v_chunks).astype("float32")
    print(f"  VAE {X_vae.shape}   T2V {X_t2v.shape}")

    # 3. Match ERA5
    print("\nOpening ERA5...")
    ds_era5, era5_tree = open_era5()

    print("Matching SST...")
    sst = match_sst(ds_era5, era5_tree, df_meta)
    ocean = np.isfinite(sst)
    print(f"  Ocean tiles: {ocean.sum():,} / {len(sst):,}")

    df_ocean = df_meta[ocean].reset_index(drop=True)
    X_vae_oc = X_vae[ocean]
    X_t2v_oc = X_t2v[ocean]
    sst_oc   = sst[ocean]
    lat_oc   = df_ocean["lat"].values
    mon_oc   = df_ocean["time"].dt.month.values

    print("Matching EIS (this loads several pressure-level fields)...")
    try:
        eis = match_eis(ds_era5, era5_tree, df_ocean)
        eis_ok = np.isfinite(eis).sum()
        print(f"  EIS valid: {eis_ok:,}")
    except Exception as e:
        print(f"  EIS matching failed ({e}) — skipping EIS runs")
        eis = None

    # 4. Run CCA
    results = []

    print("\n" + "=" * 60)
    print("Run 1 — Tile2Vec x SST (20-year, deconfounded)")
    print("=" * 60)
    pipe_t2v = run_cca(X_t2v_oc, sst_oc, lat_oc, mon_oc, N_PCA_T2V, tag="T2V-SST")
    results.append(("T2V", "SST", pipe_t2v["r_pearson"], pipe_t2v["r_spearman"]))

    print("\n" + "=" * 60)
    print("Run 2 — VAE (unblinded) x SST (20-year, deconfounded)")
    print("=" * 60)
    pipe_vae_sst = run_cca(X_vae_oc, sst_oc, lat_oc, mon_oc, N_PCA_VAE, tag="VAE-SST")
    results.append(("VAE", "SST", pipe_vae_sst["r_pearson"], pipe_vae_sst["r_spearman"]))

    if eis is not None:
        print("\n" + "=" * 60)
        print("Run 3 — VAE (unblinded) x EIS (20-year, deconfounded)")
        print("=" * 60)
        pipe_vae_eis = run_cca(X_vae_oc, eis, lat_oc, mon_oc, N_PCA_VAE, tag="VAE-EIS")
        results.append(("VAE", "EIS", pipe_vae_eis["r_pearson"], pipe_vae_eis["r_spearman"]))

    # 5. Decoded walks (needs checkpoint)
    has_ckpt = os.path.exists(CHECKPOINT)
    if not has_ckpt:
        print(f"\nCheckpoint not found at {CHECKPOINT} — skipping decoded walks.")
        print("To enable: copy lightning_model_50_transform.pt to that path.")
    else:
        print("\nDecoding VAE walk along SST direction...")
        decoded_walk(
            pipe        = pipe_vae_sst,
            var_vals    = sst_oc,
            tag         = "(20-year Pelican, unblinded)",
            phys_label  = "SST (°C)",
            out_path    = os.path.join(OUT_DIR, "walk_vae_sst.png"),
        )
        if eis is not None and pipe_vae_eis["r_pearson"] > 0.10:
            print("Decoding VAE walk along EIS direction...")
            decoded_walk(
                pipe        = pipe_vae_eis,
                var_vals    = eis[np.isfinite(eis)],
                tag         = "(20-year Pelican, unblinded)",
                phys_label  = "EIS (K)",
                out_path    = os.path.join(OUT_DIR, "walk_vae_eis.png"),
            )

    # 6. Save r values
    results_path = os.path.join(OUT_DIR, "cca_results.txt")
    with open(results_path, "w") as f:
        f.write("20-year MODIS cloud embedding CCA results\n")
        f.write(f"Embeddings: {n_loaded} days, {len(df_ocean):,} ocean tiles\n")
        f.write(f"Deconfounded by: lat, lat^2, sin(month), cos(month)\n")
        f.write(f"r from 20% held-out test set\n\n")
        f.write(f"{'Embedding':<8} {'Target':<6} {'Pearson r':>10} {'Spearman rho':>13}\n")
        f.write("-" * 42 + "\n")
        for emb, tgt, pr, sr in results:
            f.write(f"{emb:<8} {tgt:<6} {pr:>10.4f} {sr:>13.4f}\n")
    print(f"\nResults saved -> {results_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
