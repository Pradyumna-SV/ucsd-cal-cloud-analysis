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

import hashlib
import json
import os
import re
import warnings
from datetime import datetime
from pathlib import Path

import wandb
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
CACHE_DIR     = os.environ.get("CACHE_DIR",     OUT_DIR)
CHECKPOINT    = os.environ.get("CHECKPOINT",    "/workspace/vae_checkpoint/lightning_model_50_transform.pt")
MANIFEST      = os.environ.get("MANIFEST",      "/workspace/repo/manifest.csv")
STREAM_STRIDE = int(os.environ.get("STREAM_STRIDE", 11))
MAX_PER_DAY   = int(os.environ.get("MAX_PER_DAY",   200))

# Analysis hyperparameters — all overridable via env vars.
N_PCA_VAE       = int(os.environ.get("N_PCA_VAE",       50))
N_PCA_T2V       = int(os.environ.get("N_PCA_T2V",       49))
CCA_N_COMPONENTS = int(os.environ.get("CCA_N_COMPONENTS", 1))
N_WALK_STEPS    = int(os.environ.get("N_WALK_STEPS",     9))
WALK_SIGMA      = float(os.environ.get("WALK_SIGMA",     1.5))
APPLY_CLAHE     = os.environ.get("APPLY_CLAHE", "0").strip().lower() in ("1", "true", "yes")

# W&B
WANDB_PROJECT  = os.environ.get("WANDB_PROJECT",  "ucsd-cal-cloud-cca")
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME", None)   # auto-generated if not set

# ARCO-ERA5 on Google Cloud Storage — public, no auth needed.
# Contains: sea_surface_temperature, temperature (pressure levels),
#           specific_humidity, geopotential, vertical_velocity,
#           total_column_water_vapour, 2m_temperature
ARCO_ERA5 = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

os.makedirs(OUT_DIR, exist_ok=True)

# ── MODIS timestamp parsing ─────────────────────────────────────────────────
# JSON keys in centers files are MODIS granule names like:
#   "MOD35_L2.A2002362.1005.061.2017321034257.hdf"
# Pattern: .A{YYYY}{DOY}.{HHMM}
_MODIS_RE = re.compile(r'\.A(\d{7})\.(\d{4})')

def parse_modis_ts(key: str) -> pd.Timestamp:
    """Parse a MODIS granule name / JSON key into a pandas Timestamp.
    Falls back to pd.to_datetime for ISO-formatted keys."""
    m = _MODIS_RE.search(key)
    if m:
        year_doy, hhmm = m.group(1), m.group(2)
        year = int(year_doy[:4])
        doy  = int(year_doy[4:])
        hh, mm = int(hhmm[:2]), int(hhmm[2:])
        try:
            return (pd.Timestamp(datetime(year, 1, 1))
                    + pd.Timedelta(days=doy - 1, hours=hh, minutes=mm))
        except Exception:
            return pd.NaT
    # fallback for ISO strings
    try:
        return pd.Timestamp(key)
    except Exception:
        return pd.NaT


# ── ERA5 setup ─────────────────────────────────────────────────────────────
# All variables fetched in a single loop over unique days.
_ERA5_SPECS = [
    # (xarray varname,           pressure level,  output key)
    ("sea_surface_temperature",  None,            "sst_raw"),
    ("temperature",              700,             "T700"),
    ("temperature",              850,             "T850"),
    ("temperature",              1000,            "T1000"),
    ("specific_humidity",        850,             "q850"),
    ("specific_humidity",        1000,            "q1000"),
    ("geopotential",             700,             "Phi700"),
    ("2m_temperature",           None,            "T2m"),
    ("vertical_velocity",        500,             "omega500"),
]


def open_era5():
    print("Opening ARCO-ERA5 (GCS, anonymous)...")
    gcs = gcsfs.GCSFileSystem(token="anon")
    ds  = xr.open_zarr(gcs.get_mapper(ARCO_ERA5), chunks=None)
    lats = ds["latitude"].values
    lons = ds["longitude"].values
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    tree = cKDTree(np.column_stack([lat_grid.ravel(), lon_grid.ravel()]))
    print(f"  ERA5 grid: {len(lons)} lon x {len(lats)} lat")
    print(f"  ERA5 time: {str(ds['time'].values[0])[:10]} -> {str(ds['time'].values[-1])[:10]}")
    return ds, tree


def _match_all_batched(ds, tree, df):
    """
    Single-pass ERA5 match: loop over unique days once, load all required
    variables together per day, extract values for every tile.
    Returns dict of output_key -> np.ndarray (len = len(df), NaN where missing).
    """
    lats     = df["lat"].values
    lons     = df["lon"].values
    times    = pd.to_datetime(df["time"], errors="coerce")
    lons_pos = np.where(lons < 0, lons + 360, lons)   # ERA5 uses 0-360
    _, sp_idx = tree.query(np.column_stack([lats, lons_pos]))

    outputs = {key: np.full(len(df), np.nan) for _, _, key in _ERA5_SPECS}

    # Filter to valid, in-era timestamps (same guards as the old _match_var).
    valid_mask = times.notna()
    if not valid_mask.any():
        return outputs
    valid_idx   = np.where(valid_mask)[0]
    valid_times = times.iloc[valid_idx]
    yr  = valid_times.dt.year.values
    mo  = valid_times.dt.month.values
    dy  = valid_times.dt.day.values
    era_mask    = (yr >= 2000) & (yr <= 2023)
    valid_idx   = valid_idx[era_mask]
    yr, mo, dy  = yr[era_mask], mo[era_mask], dy[era_mask]
    valid_times = times.iloc[valid_idx]

    date_strs    = np.array([f"{y:04d}-{m:02d}-{d:02d}" for y, m, d in zip(yr, mo, dy)])
    unique_dates = np.unique(date_strs)

    # Only request variables that actually exist in this ERA5 dataset.
    avail_vars   = set(ds.data_vars)
    active_specs = [(v, l, k) for v, l, k in _ERA5_SPECS if v in avail_vars]
    unique_vars  = list(dict.fromkeys(v for v, _, _ in active_specs))
    skipped      = [k for v, _, k in _ERA5_SPECS if v not in avail_vars]
    if skipped:
        print(f"  [batch] variables not in dataset, skipped: {skipped}")

    n_ok, n_fail, first_err = 0, 0, None
    for date_str in unique_dates:
        in_date  = date_strs == date_str
        row_idxs = valid_idx[in_date]
        try:
            day_ds    = ds[unique_vars].sel(time=date_str).load()
            day_times = pd.DatetimeIndex(day_ds["time"].values)
            # Compute nearest time indices once per tile; reuse across all variables.
            tile_tis  = day_times.get_indexer(times.iloc[row_idxs], method="nearest")
            for varname, level, key in active_specs:
                try:
                    da   = day_ds[varname]
                    if level is not None:
                        da = da.sel(level=level)
                    flat = da.values.reshape(len(day_times), -1)
                    for j, (i, ti) in enumerate(zip(row_idxs, tile_tis)):
                        if ti >= 0:
                            outputs[key][i] = float(flat[ti, sp_idx[i]])
                except Exception:
                    pass  # leave NaN for this variable/day
            n_ok += 1
        except Exception as e:
            n_fail += 1
            if first_err is None:
                first_err = f"{date_str}: {type(e).__name__}: {e}"

    print(f"  [batch] days OK={n_ok} FAIL={n_fail}  valid_ts={len(valid_idx)}  "
          f"sst_matched={np.isfinite(outputs['sst_raw']).sum()}")
    if first_err:
        print(f"  [batch] first failure: {first_err}")
    return outputs


def _compute_sst(raw):
    """Convert raw ERA5 SST (K) to Celsius; NaN over land."""
    r = raw["sst_raw"]
    return np.where(r > 200, r - 273.15, np.nan)


def _compute_eis(raw):
    """Estimated Inversion Strength (Wood & Bretherton 2006), in K."""
    Lv = 2.5e6; Rd = 287.0; Rv = 461.0; cp = 1005.0; g = 9.81
    T700,  T850,  T1000 = raw["T700"],  raw["T850"],  raw["T1000"]
    q850,  q1000         = raw["q850"],  raw["q1000"]
    Phi700, T2m          = raw["Phi700"], raw["T2m"]

    lts    = T700 * (1000 / 700) ** 0.286 - T1000
    e_s    = 6.112e2 * np.exp(17.67 * (T850 - 273.15) / (T850 - 29.65))
    qs     = 0.622 * e_s / (85000 - 0.378 * e_s)
    gm     = (g / cp) * (1 + Lv * qs / (Rd * T850)) / (1 + Lv**2 * qs / (cp * Rv * T850**2))
    z700_km = Phi700 / (g * 1000)
    e_sfc  = np.clip(q1000 * 1013.25e2 / (0.622 + q1000), 10, 5000)
    T_D    = (243.5 * np.log(e_sfc / 611.2) / (17.67 - np.log(e_sfc / 611.2))) + 273.15
    z_lcl  = np.clip(0.125 * (T2m - T_D), 0, 3)
    return lts - gm * 1000 * (z700_km - z_lcl)


# ── ERA5 disk cache ─────────────────────────────────────────────────────────
def _era5_cache_key(df_meta: pd.DataFrame, stride: int) -> str:
    """MD5 fingerprint of the loaded tile set (unique days + stride)."""
    times = pd.to_datetime(df_meta["time"], errors="coerce").dropna()
    dates = sorted({f"{t.year:04d}-{t.month:02d}-{t.day:02d}" for t in times})
    raw   = f"stride={stride}|" + "|".join(dates)
    return hashlib.md5(raw.encode()).hexdigest()


def _save_era5_cache(path: Path, key: str, sst, eis, omega500):
    np.savez_compressed(
        str(path),
        _key       = np.array([key]),
        sst        = sst,
        eis        = eis      if eis      is not None else np.array([np.nan]),
        omega500   = omega500 if omega500 is not None else np.array([np.nan]),
        _has_eis   = np.array([eis      is not None]),
        _has_omega = np.array([omega500 is not None]),
    )
    print(f"  ERA5 cache saved -> {path}")


def _load_era5_cache(path: Path, key: str):
    """Returns (sst, eis, omega500) on cache hit, None on miss or key mismatch."""
    if not path.exists():
        return None
    try:
        c = np.load(str(path), allow_pickle=False)
        if str(c["_key"][0]) != key:
            print("  ERA5 cache key mismatch — re-matching from GCS.")
            return None
        sst      = c["sst"]
        eis      = c["eis"]      if bool(c["_has_eis"][0])   else None
        omega500 = c["omega500"] if bool(c["_has_omega"][0]) else None
        print(f"  ERA5 cache HIT  ({path.name})")
        return sst, eis, omega500
    except Exception as e:
        print(f"  ERA5 cache load error ({e}) — re-matching from GCS.")
        return None


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
        # Keys are HHMM strings (e.g. '0000', '0005') — combine with the
        # directory date to get the full acquisition timestamp.
        try:
            hh, mm = int(ts[:2]), int(ts[2:])
            t = pd.Timestamp(datetime(year, month, day, hh, mm))
        except Exception:
            t = parse_modis_ts(ts)   # fallback for any other format
        for lat, lon in meta[ts]:
            rows.append({"lat": float(lat), "lon": float(lon), "time": t})
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
                physics_scores=Xc_all.flatten(),
                valid_mask=valid)


# ── Multivariate CCA pipeline ───────────────────────────────────────────────
def run_cca_multi(X, Y_matrix, target_names, lat, months, n_pca, tag=""):
    """
    Deconfound X and each Y column by (lat, lat², sin/cos month), then run
    CCA(n_components = min(CCA_N_COMPONENTS, n_targets)).

    Returns one canonical direction per component, canonical scores for the
    full dataset, and Pearson/Spearman r for each canonical pair on the 20%
    held-out test set.

    Also returns valid_mask so the caller can filter X_vae to matching rows
    before passing it to cca_component_walk.
    """
    valid = np.all(np.isfinite(Y_matrix), axis=1)
    X  = X[valid].astype("float32")
    Y  = Y_matrix[valid].astype("float32")
    lat    = lat[valid]
    months = months[valid]

    C = np.column_stack([lat, lat**2,
                         np.sin(2 * np.pi * months / 12),
                         np.cos(2 * np.pi * months / 12)])

    reg_x    = LinearRegression(fit_intercept=True).fit(C, X)
    X_deconf = X - reg_x.predict(C)

    Y_deconf = np.zeros_like(Y)
    for j in range(Y.shape[1]):
        reg_y = LinearRegression(fit_intercept=True).fit(C, Y[:, j])
        Y_deconf[:, j] = Y[:, j] - reg_y.predict(C)

    sx = StandardScaler(); sy = StandardScaler()
    X_sc = sx.fit_transform(X_deconf)
    Y_sc = sy.fit_transform(Y_deconf)

    n_pca_act = min(n_pca, X.shape[1] - 1, X.shape[0] // 10)
    pca = PCA(n_components=n_pca_act, random_state=42)
    X_pca = pca.fit_transform(X_sc)
    print(f"  [{tag}] PCA({n_pca_act}): {pca.explained_variance_ratio_.sum()*100:.1f}% variance retained")

    n_comp = min(CCA_N_COMPONENTS, Y.shape[1])
    X_tr, X_te, Y_tr, Y_te = train_test_split(X_pca, Y_sc, test_size=0.2, random_state=42)
    cca = CCA(n_components=n_comp, max_iter=2000).fit(X_tr, Y_tr)
    Xc_te, Yc_te = cca.transform(X_te, Y_te)

    rs_pearson, rs_spearman = [], []
    for k in range(n_comp):
        pr, _ = pearsonr(Xc_te[:, k], Yc_te[:, k])
        sr, _ = spearmanr(Xc_te[:, k], Yc_te[:, k])
        rs_pearson.append(pr)
        rs_spearman.append(sr)
        print(f"  [{tag}] CCA{k+1}: Pearson r = {pr:.4f}  Spearman rho = {sr:.4f}  (20% test)")

    phys_dirs = []
    for k in range(n_comp):
        d = pca.components_.T @ cca.x_weights_[:, k]
        d /= np.linalg.norm(d)
        phys_dirs.append(d)

    Xc_all, _ = cca.transform(X_pca, Y_sc)

    return dict(
        rs_pearson=rs_pearson,
        rs_spearman=rs_spearman,
        phys_dirs=phys_dirs,
        scaler_X=sx,
        X_deconf=X_deconf,
        X_raw=X,
        cca_scores=Xc_all,        # (N_valid, n_comp)
        n_comp=n_comp,
        target_names=target_names,
        valid_mask=valid,
    )


# ── Bin-composite walk ───────────────────────────────────────────────────────
def bin_composite_walk(X_vae, var_vals, lat, months, tag, phys_label, out_path,
                       n_bins=9, batch_size=256):
    """
    Bin-composite decoded walk — the only approach that reliably shows visual
    signal when CCA r is modest.

    Strategy
    --------
    1. Deconfound *var_vals* (SST or EIS) by (lat, lat², sin/cos month).
       This residual is used only for binning so lat/season don't dominate.
    2. Divide tiles into quantile bins by the residual.
    3. Decode the *original* (on-manifold) VAE embeddings for every tile in
       each bin, then average those decoded images.
       Averaging ~N/n_bins real tiles cancels decoder noise and reveals any
       systematic visual pattern without ever leaving the trained manifold.
    4. Label each panel with the actual (non-residual) median of var_vals.
    """
    import torch
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from vae import VAELightningModule

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = VAELightningModule.load_from_checkpoint(CHECKPOINT)
    model.to(device).eval()
    print(f"  VAE loaded on {device}")

    valid = np.isfinite(var_vals)
    X_v   = X_vae[valid].astype("float32")
    y_v   = var_vals[valid]
    lat_v = lat[valid]
    mon_v = months[valid]
    print(f"  [{phys_label}] {valid.sum():,} valid tiles for composite walk")

    # Deconfound var_vals for binning only; decode original embeddings.
    C = np.column_stack([lat_v, lat_v**2,
                         np.sin(2 * np.pi * mon_v / 12),
                         np.cos(2 * np.pi * mon_v / 12)])
    y_res = y_v - LinearRegression(fit_intercept=True).fit(C, y_v).predict(C)

    edges = np.percentile(y_res, np.linspace(0, 100, n_bins + 1))
    edges[0] -= 1e-6; edges[-1] += 1e-6

    composite_imgs, bin_medians, bin_ns = [], [], []
    for i in range(n_bins):
        mask  = (y_res >= edges[i]) & (y_res < edges[i + 1])
        X_bin = X_v[mask]
        decoded = []
        with torch.no_grad():
            for start in range(0, len(X_bin), batch_size):
                z   = torch.tensor(X_bin[start:start + batch_size],
                                   dtype=torch.float32).to(device)
                out = model.decoder(z).cpu().numpy()  # (B, 3, H, W)
                decoded.append(np.mean(out, axis=1))   # greyscale: (B, H, W)
        composite = np.mean(np.concatenate(decoded, axis=0), axis=0)
        composite_imgs.append(composite)
        bin_medians.append(float(np.median(y_v[mask])))
        bin_ns.append(int(mask.sum()))
        print(f"  bin {i+1}/{n_bins}: n={bin_ns[-1]:,}  median={bin_medians[-1]:.2f}")

    all_vals = np.concatenate([img.ravel() for img in composite_imgs])
    lo, hi   = np.percentile(all_vals, [2, 98])
    imgs = [np.clip((img - lo) / (hi - lo + 1e-8), 0, 1) for img in composite_imgs]

    fig, axes = plt.subplots(1, n_bins,
                             figsize=(n_bins * 2.4, 3.4),
                             facecolor="#1a1a1a")
    fig.suptitle(
        f"VAE bin-composite walk — {phys_label}  {tag}\n"
        f"(each panel = mean of n decoded tiles; binned by lat/season-deconfounded residual)",
        color="white", fontsize=9, fontweight="bold")
    for col, (img, med, n) in enumerate(zip(imgs, bin_medians, bin_ns)):
        ax = axes[col]
        ax.imshow(img, cmap="bone", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")
        ax.set_title(f"{med:.1f}", color="white", fontsize=9, fontweight="bold", pad=3)
        ax.set_xlabel(f"n={n:,}", color="#aaa", fontsize=7)

    plt.tight_layout(rect=[0, 0.04, 1, 0.84])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
    print(f"  Saved -> {out_path}")
    plt.close()


# ── Bin-mosaic walk ──────────────────────────────────────────────────────────
def bin_mosaic_walk(X_vae, var_vals, lat, months, tag, phys_label, out_path,
                    n_bins=9, n_samples=3, seed=42):
    """
    Same binning as bin_composite_walk but shows n_samples individual decoded
    tiles per bin instead of their mean.  Averaging destroys cloud texture;
    individual tiles show what the VAE actually learned.
    """
    import torch
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from vae import VAELightningModule

    rng    = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = VAELightningModule.load_from_checkpoint(CHECKPOINT)
    model.to(device).eval()
    print(f"  VAE loaded on {device}")

    valid = np.isfinite(var_vals)
    X_v   = X_vae[valid].astype("float32")
    y_v   = var_vals[valid]
    lat_v = lat[valid]
    mon_v = months[valid]
    print(f"  [{phys_label}] {valid.sum():,} valid tiles for mosaic walk")

    C     = np.column_stack([lat_v, lat_v**2,
                             np.sin(2 * np.pi * mon_v / 12),
                             np.cos(2 * np.pi * mon_v / 12)])
    y_res = y_v - LinearRegression(fit_intercept=True).fit(C, y_v).predict(C)

    edges = np.percentile(y_res, np.linspace(0, 100, n_bins + 1))
    edges[0] -= 1e-6; edges[-1] += 1e-6

    all_tiles, bin_medians, bin_ns = [], [], []
    for i in range(n_bins):
        mask  = (y_res >= edges[i]) & (y_res < edges[i + 1])
        X_bin = X_v[mask]
        k     = min(n_samples, len(X_bin))
        idx   = rng.choice(len(X_bin), size=k, replace=False)
        with torch.no_grad():
            z   = torch.tensor(X_bin[idx], dtype=torch.float32).to(device)
            out = model.decoder(z).cpu().numpy()          # (k, 3, H, W) in [-1, 1]
        tiles = np.clip((out + 1) / 2, 0, 1).transpose(0, 2, 3, 1)   # (k, H, W, 3)
        all_tiles.append(tiles)
        bin_medians.append(float(np.median(y_v[mask])))
        bin_ns.append(int(mask.sum()))
        print(f"  bin {i+1}/{n_bins}: n={bin_ns[-1]:,}  median={bin_medians[-1]:.2f}")

    fig, axes = plt.subplots(n_samples, n_bins,
                             figsize=(n_bins * 2.4, n_samples * 2.5),
                             facecolor="#1a1a1a")
    fig.suptitle(
        f"VAE bin-mosaic walk — {phys_label}  {tag}\n"
        f"(each panel = individual decoded tile; binned by lat/season-deconfounded residual)",
        color="white", fontsize=9, fontweight="bold")
    for col in range(n_bins):
        for row in range(n_samples):
            ax = axes[row, col]
            ax.imshow(all_tiles[col][row])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#555")
            if row == 0:
                ax.set_title(f"{bin_medians[col]:.1f}", color="white",
                             fontsize=9, fontweight="bold", pad=3)
            if row == n_samples - 1:
                ax.set_xlabel(f"n={bin_ns[col]:,}", color="#aaa", fontsize=7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.87])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
    print(f"  Saved -> {out_path}")
    plt.close()


# ── PCA-component composite walk ─────────────────────────────────────────────
def pca_component_walk(X_vae, target_vals, lat, months, tag, phys_label, out_path,
                       n_components=3, n_bins=9, batch_size=256):
    """
    Walk along the top-n_components PCA directions of the deconfounded VAE
    embedding space.  Each row is one PC; panels are composites of tiles
    binned by their score on that PC.  Each row is labelled with variance
    explained and Pearson r vs the deconfounded target — revealing which
    structural mode of the VAE correlates with the physical variable.

    CCA(n_components=1) captures the single direction maximally correlated
    with a scalar target.  Additional variance may live in PC2/PC3
    that CCA can't expose when Y is 1-D.
    """
    import torch
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from vae import VAELightningModule

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = VAELightningModule.load_from_checkpoint(CHECKPOINT)
    model.to(device).eval()
    print(f"  VAE loaded on {device}")

    valid   = np.isfinite(target_vals)
    X_v     = X_vae[valid].astype("float32")
    y_v     = target_vals[valid]
    lat_v   = lat[valid]
    mon_v   = months[valid]

    C       = np.column_stack([lat_v, lat_v**2,
                               np.sin(2 * np.pi * mon_v / 12),
                               np.cos(2 * np.pi * mon_v / 12)])
    X_deconf = X_v - LinearRegression(fit_intercept=True).fit(C, X_v).predict(C)
    y_res    = y_v - LinearRegression(fit_intercept=True).fit(C, y_v).predict(C)

    X_sc    = StandardScaler().fit_transform(X_deconf)
    pca     = PCA(n_components=n_components, random_state=42)
    X_pca   = pca.fit_transform(X_sc)          # (N, n_components)

    pc_r    = [pearsonr(X_pca[:, k], y_res)[0] for k in range(n_components)]

    fig, axes = plt.subplots(n_components, n_bins,
                             figsize=(n_bins * 2.4, n_components * 2.8),
                             facecolor="#1a1a1a")
    fig.suptitle(
        f"PCA-component composite walk — {phys_label}  {tag}\n"
        f"(each row = top-K PCA direction of deconfounded VAE space; "
        f"composites binned by PC score)",
        color="white", fontsize=9, fontweight="bold")

    for k in range(n_components):
        scores = X_pca[:, k]
        edges  = np.percentile(scores, np.linspace(0, 100, n_bins + 1))
        edges[0] -= 1e-6; edges[-1] += 1e-6

        var_pct  = pca.explained_variance_ratio_[k] * 100
        row_label = f"PC{k+1}  {var_pct:.1f}% var  r_{phys_label.split()[0]}={pc_r[k]:.3f}"
        print(f"  {row_label}")

        row_composites = []
        for i in range(n_bins):
            mask   = (scores >= edges[i]) & (scores < edges[i + 1])
            X_bin  = X_v[mask]
            decoded = []
            with torch.no_grad():
                for start in range(0, len(X_bin), batch_size):
                    z   = torch.tensor(X_bin[start:start + batch_size],
                                       dtype=torch.float32).to(device)
                    out = model.decoder(z).cpu().numpy()
                    decoded.append(np.mean(out, axis=1))
            row_composites.append(np.mean(np.concatenate(decoded, axis=0), axis=0))

        all_vals = np.concatenate([c.ravel() for c in row_composites])
        lo, hi   = np.percentile(all_vals, [2, 98])
        for i, composite in enumerate(row_composites):
            img = np.clip((composite - lo) / (hi - lo + 1e-8), 0, 1)
            ax  = axes[k, i]
            ax.imshow(img, cmap="bone", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#555")
            if i == 0:
                ax.set_ylabel(row_label, color="white", fontsize=7,
                              rotation=90, labelpad=4)
            if k == 0:
                ax.set_title(f"bin {i+1}", color="#aaa", fontsize=8, pad=3)

    plt.tight_layout(rect=[0, 0.02, 1, 0.91])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
    print(f"  Saved -> {out_path}")
    plt.close()


# ── CCA-component composite walk ─────────────────────────────────────────────
def cca_component_walk(X_vae, cca_scores, target_names, canonical_rs, tag, out_path,
                       n_bins=9, batch_size=256):
    """
    One composite row per CCA component from run_cca_multi.

    Bins tiles by their canonical score Xc_k (not raw variable values), so
    each row is a pure walk along that CCA direction in latent space.
    CCA1 will likely be the brightness/EIS axis; CCA2/CCA3 should capture
    structural variance orthogonal to it (driven by omega500, SST, etc.).

    X_vae must already be filtered to the same valid rows as cca_scores.
    """
    import torch
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from vae import VAELightningModule

    n_comp     = cca_scores.shape[1]
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model      = VAELightningModule.load_from_checkpoint(CHECKPOINT)
    model.to(device).eval()
    print(f"  VAE loaded on {device}")

    X_v        = X_vae.astype("float32")
    targets_str = " + ".join(target_names)

    fig, axes = plt.subplots(n_comp, n_bins,
                             figsize=(n_bins * 2.4, n_comp * 2.8),
                             facecolor="#1a1a1a")
    if n_comp == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f"Multivariate CCA walk — [{targets_str}]  {tag}\n"
        f"(each row = CCA direction k; composites binned by canonical score Xc_k)",
        color="white", fontsize=9, fontweight="bold")

    for k in range(n_comp):
        scores = cca_scores[:, k]
        edges  = np.percentile(scores, np.linspace(0, 100, n_bins + 1))
        edges[0] -= 1e-6; edges[-1] += 1e-6

        print(f"  CCA{k+1} (r={canonical_rs[k]:.3f}): computing composites...")
        row_composites = []
        for i in range(n_bins):
            mask  = (scores >= edges[i]) & (scores < edges[i + 1])
            X_bin = X_v[mask]
            decoded = []
            with torch.no_grad():
                for start in range(0, len(X_bin), batch_size):
                    z   = torch.tensor(X_bin[start:start + batch_size],
                                       dtype=torch.float32).to(device)
                    out = model.decoder(z).cpu().numpy()
                    decoded.append(np.mean(out, axis=1))
            row_composites.append(np.mean(np.concatenate(decoded, axis=0), axis=0))

        all_vals = np.concatenate([c.ravel() for c in row_composites])
        lo, hi   = np.percentile(all_vals, [2, 98])
        for i, composite in enumerate(row_composites):
            img = np.clip((composite - lo) / (hi - lo + 1e-8), 0, 1)
            ax  = axes[k, i]
            ax.imshow(img, cmap="bone", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#555")
            if i == 0:
                ax.set_ylabel(f"CCA{k+1}\nr={canonical_rs[k]:.3f}",
                              color="white", fontsize=8, rotation=90, labelpad=4)
            if k == 0:
                ax.set_title(f"bin {i+1}", color="#aaa", fontsize=8, pad=3)

    plt.tight_layout(rect=[0, 0.02, 1, 0.91])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
    print(f"  Saved -> {out_path}")
    plt.close()


# ── Figure 1: UMAP of VAE embeddings coloured by EIS / SST ───────────────────
def plot_umap(X_vae, sst, eis, lat, months, tag, out_path):
    """
    PCA(50) → UMAP(2D) on deconfounded VAE embeddings.
    Two panels: left coloured by deconfounded EIS residual,
                right coloured by deconfounded SST residual.
    Shows whether EIS/SST structure is geometrically organised in
    the embedding space — no decoder, no CCA direction needed.
    """
    from umap import UMAP

    valid = np.isfinite(eis) & np.isfinite(sst)
    X_v   = X_vae[valid].astype("float32")
    eis_v = eis[valid]
    sst_v = sst[valid]
    lat_v = lat[valid]
    mon_v = months[valid]
    print(f"  [UMAP] {valid.sum():,} tiles with valid EIS+SST")

    C        = np.column_stack([lat_v, lat_v**2,
                                np.sin(2 * np.pi * mon_v / 12),
                                np.cos(2 * np.pi * mon_v / 12)])
    X_deconf = X_v - LinearRegression(fit_intercept=True).fit(C, X_v).predict(C)
    eis_res  = eis_v - LinearRegression(fit_intercept=True).fit(C, eis_v).predict(C)
    sst_res  = sst_v - LinearRegression(fit_intercept=True).fit(C, sst_v).predict(C)

    X_sc  = StandardScaler().fit_transform(X_deconf)
    X_pca = PCA(n_components=50, random_state=42).fit_transform(X_sc)
    print(f"  [UMAP] running UMAP on {len(X_pca):,} x 50D embeddings...")
    embed_2d = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                    random_state=42, low_memory=True).fit_transform(X_pca)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#111")
    fig.suptitle(f"VAE embedding UMAP — coloured by deconfounded EIS and SST  {tag}",
                 color="white", fontsize=11, fontweight="bold")

    for ax, residuals, label, cmap in zip(
            axes,
            [eis_res,         sst_res],
            ["EIS residual (K)", "SST residual (°C)"],
            ["RdBu_r",        "RdYlBu_r"]):
        vmax = np.percentile(np.abs(residuals), 95)
        sc   = ax.scatter(embed_2d[:, 0], embed_2d[:, 1],
                          c=residuals, cmap=cmap, s=0.8, alpha=0.35,
                          vmin=-vmax, vmax=vmax, rasterized=True,
                          linewidths=0)
        cb = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
        cb.set_label(label, color="white", fontsize=9)
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
        ax.set_title(label, color="white", fontsize=10, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#111")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#111")
    print(f"  Saved -> {out_path}")
    plt.close()


# ── Figure 2: Geographic hexbin of CCA scores ────────────────────────────────
def plot_geo_cca(lon, lat, scores_sst, mask_sst, scores_eis, mask_eis, tag, out_path):
    """
    Two-panel hexbin world map.
    Left:  SST CCA canonical score per tile.
    Right: EIS CCA canonical score per tile.
    Tiles are aggregated by mean CCA score within each hex cell so that
    dense regions (Sc decks) don't visually dominate through sheer dot count.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 6), facecolor="#111")
    fig.suptitle(f"Geographic distribution of VAE CCA scores  {tag}",
                 color="white", fontsize=11, fontweight="bold")

    pairs = [
        (scores_sst, mask_sst, "SST CCA score"),
        (scores_eis, mask_eis, "EIS CCA score"),
    ]
    for ax, (scores, mask, label) in zip(axes, pairs):
        lo_v = lon[mask]
        la_v = lat[mask]
        lo_180 = np.where(lo_v > 180, lo_v - 360, lo_v)

        vmax = np.percentile(np.abs(scores), 98)
        hb   = ax.hexbin(lo_180, la_v, C=scores,
                         reduce_C_function=np.mean,
                         gridsize=120, cmap="RdBu_r",
                         vmin=-vmax, vmax=vmax,
                         extent=[-180, 180, -60, 60],
                         linewidths=0.1)
        cb = plt.colorbar(hb, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label(label, color="white", fontsize=9)
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

        ax.set_xlim(-180, 180); ax.set_ylim(-60, 60)
        ax.set_facecolor("#111")
        ax.set_title(label, color="white", fontsize=10, pad=6)
        ax.set_xlabel("Longitude", color="#aaa", fontsize=8)
        ax.set_ylabel("Latitude",  color="#aaa", fontsize=8)
        ax.tick_params(colors="#aaa", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        ax.set_xticks(range(-180, 181, 60))
        ax.set_yticks(range(-60, 61, 30))
        ax.grid(color="#333", linewidth=0.4, linestyle="--")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#111")
    print(f"  Saved -> {out_path}")
    plt.close()


# ── Figure 3: Per-regime CCA r values ────────────────────────────────────────
_REGIMES = [
    (-30, -10, -100, -70,  "SE Pacific\n(Sc deck)"),
    (-30, -10,  -20,   5,  "SE Atlantic\n(Sc deck)"),
    ( 15,  35, -140, -110, "NE Pacific\n(Sc deck)"),
    (-20,  20,  -60,  20,  "Tropical\nAtlantic"),
    (-20,  20,   40, 100,  "Tropical\nIndian"),
    (-20,  20,  120, 180,  "Tropical\nW Pacific"),
]

def plot_regime_r(X_vae, sst, eis, lat, lon, months, n_pca,
                  global_r_sst, global_r_eis, tag, out_path,
                  min_tiles=500):
    """
    Run CCA(1) vs SST and vs EIS independently within 6 pre-defined ocean
    cloud-regime boxes.  Grouped bar chart with global r as reference lines.
    Shows whether the VAE-physics correlation is concentrated in the Sc decks.
    """
    lon_180 = np.where(lon > 180, lon - 360, lon)

    names, rs_sst, rs_eis, ns = [], [], [], []
    for lat_min, lat_max, lon_min, lon_max, name in _REGIMES:
        mask = ((lat  >= lat_min) & (lat  <= lat_max) &
                (lon_180 >= lon_min) & (lon_180 <= lon_max))
        n = int(mask.sum())
        if n < min_tiles:
            print(f"  [regime] {name!r}: only {n} tiles — skip")
            continue
        X_r   = X_vae[mask]
        sst_r = sst[mask]
        eis_r = eis[mask]
        lat_r = lat[mask]
        mon_r = months[mask]
        try:
            pr_sst = run_cca(X_r, sst_r, lat_r, mon_r, n_pca,
                             tag=f"regime-{name[:8]}-SST")["r_pearson"]
            pr_eis = run_cca(X_r, eis_r, lat_r, mon_r, n_pca,
                             tag=f"regime-{name[:8]}-EIS")["r_pearson"]
            names.append(name); rs_sst.append(pr_sst)
            rs_eis.append(pr_eis); ns.append(n)
            print(f"  [regime] {name!r}: SST r={pr_sst:.3f}  EIS r={pr_eis:.3f}  n={n:,}")
        except Exception as e:
            print(f"  [regime] {name!r}: failed ({e})")

    if not names:
        print("  [regime] no valid regimes — skipping figure")
        return

    x     = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 2), 5), facecolor="#111")
    ax.set_facecolor("#111")

    ax.bar(x - width / 2, rs_sst, width, label="SST", color="#4FC3F7", alpha=0.85)
    ax.bar(x + width / 2, rs_eis, width, label="EIS", color="#FF8A65", alpha=0.85)

    ax.axhline(global_r_sst, color="#4FC3F7", linewidth=1.2,
               linestyle="--", alpha=0.6, label=f"Global SST r={global_r_sst:.3f}")
    ax.axhline(global_r_eis, color="#FF8A65", linewidth=1.2,
               linestyle="--", alpha=0.6, label=f"Global EIS r={global_r_eis:.3f}")
    ax.axhline(0, color="#666", linewidth=0.6)

    for i, n in enumerate(ns):
        ax.text(i, ax.get_ylim()[0] * 0.95 if ax.get_ylim()[0] < 0 else -0.01,
                f"n={n:,}", ha="center", color="#888", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(names, color="white", fontsize=9)
    ax.set_ylabel("Pearson r (20% held-out test)", color="white", fontsize=9)
    ax.set_title(f"VAE-CCA Pearson r by ocean cloud regime  {tag}",
                 color="white", fontsize=11, fontweight="bold")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    legend = ax.legend(facecolor="#222", labelcolor="white", fontsize=8,
                       framealpha=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#111")
    print(f"  Saved -> {out_path}")
    plt.close()


# ── Decoded walk (CCA direction — kept for reference) ────────────────────────
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

    # Convert direction from standardised X_sc space back to raw embedding space
    # (phys_dir was computed in the space where each feature has unit variance)
    sx       = pipe["scaler_X"]
    raw_dir  = pipe["physics_dir"] * sx.scale_
    raw_dir /= np.linalg.norm(raw_dir)

    # Spread = std of raw embeddings projected onto the direction
    spread = np.std(pipe["X_raw"] @ raw_dir)

    alphas = np.linspace(-WALK_SIGMA, WALK_SIGMA, N_WALK_STEPS) * spread
    # Bin var_vals by physics score to get expected label at each walk step
    score_pct = np.percentile(pipe["physics_scores"], np.linspace(0, 100, N_WALK_STEPS))
    step_labels = np.interp(score_pct,
                            np.sort(pipe["physics_scores"]),
                            np.sort(var_vals[np.isfinite(var_vals)]))

    raw_imgs = []
    with torch.no_grad():
        for alpha in alphas:
            z   = torch.tensor(mean_z + alpha * raw_dir, dtype=torch.float32).unsqueeze(0).to(device)
            out = model.decoder(z).squeeze(0).cpu().numpy()   # (3, H, W)
            raw_imgs.append(np.mean(out, axis=0))             # greyscale mean across channels

    # Global contrast stretch across all walk steps so brightness changes are visible
    all_vals = np.concatenate([img.ravel() for img in raw_imgs])
    lo, hi   = np.percentile(all_vals, [2, 98])
    imgs = [np.clip((img - lo) / (hi - lo + 1e-8), 0, 1) for img in raw_imgs]

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
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config=dict(
            stream_stride=STREAM_STRIDE,
            max_per_day=MAX_PER_DAY,
            n_pca_vae=N_PCA_VAE,
            n_pca_t2v=N_PCA_T2V,
            cca_n_components=CCA_N_COMPONENTS,
            n_walk_steps=N_WALK_STEPS,
            walk_sigma=WALK_SIGMA,
            apply_clahe=APPLY_CLAHE,
            checkpoint=os.path.basename(CHECKPOINT),
        ),
    )

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

    # 3. Match ERA5 — batched (all variables in one pass per day) + disk cache
    print("\nChecking ERA5 cache...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = Path(CACHE_DIR) / "era5_matched.npz"
    cache_key  = _era5_cache_key(df_meta, STREAM_STRIDE)
    cached     = _load_era5_cache(cache_path, cache_key)

    if cached is not None:
        sst, eis, omega500 = cached
    else:
        print("Opening ERA5...")
        ds_era5, era5_tree = open_era5()
        print("Matching all ERA5 variables (batched, single pass per day)...")
        raw = _match_all_batched(ds_era5, era5_tree, df_meta)

        sst = _compute_sst(raw)

        try:
            eis = _compute_eis(raw)
            print(f"  EIS valid: {np.isfinite(eis).sum():,}")
        except Exception as e:
            print(f"  EIS computation failed ({e}) — skipping EIS runs")
            eis = None

        try:
            omega500 = raw["omega500"].copy()
            print(f"  omega500 valid: {np.isfinite(omega500).sum():,}")
        except Exception as e:
            print(f"  omega500 failed ({e}) — skipping multivariate CCA")
            omega500 = None

        _save_era5_cache(cache_path, cache_key, sst, eis, omega500)

    ocean = np.isfinite(sst)
    print(f"  Ocean tiles: {ocean.sum():,} / {len(sst):,}")

    df_ocean = df_meta[ocean].reset_index(drop=True)
    X_vae_oc = X_vae[ocean]
    X_t2v_oc = X_t2v[ocean]
    sst_oc   = sst[ocean]
    # Filter EIS and omega500 to ocean tiles so downstream code is unchanged.
    if eis is not None:
        eis = eis[ocean]
    if omega500 is not None:
        omega500 = omega500[ocean]
    lat_oc   = df_ocean["lat"].values
    lon_oc   = df_ocean["lon"].values
    mon_oc   = df_ocean["time"].dt.month.values

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

    # 4b. Embedding-space figures (no decoder needed)
    print("\n" + "=" * 60)
    print("Figure 1 — UMAP")
    print("=" * 60)
    if eis is not None:
        plot_umap(
            X_vae  = X_vae_oc,
            sst    = sst_oc,
            eis    = eis,
            lat    = lat_oc,
            months = mon_oc,
            tag    = "(20-year Pelican, unblinded)",
            out_path = os.path.join(OUT_DIR, "fig_umap.png"),
        )

    print("\n" + "=" * 60)
    print("Figure 2 — Geographic CCA scores")
    print("=" * 60)
    plot_geo_cca(
        lon        = lon_oc,
        lat        = lat_oc,
        scores_sst = pipe_vae_sst["physics_scores"],
        mask_sst   = pipe_vae_sst["valid_mask"],
        scores_eis = pipe_vae_eis["physics_scores"] if eis is not None else np.array([]),
        mask_eis   = pipe_vae_eis["valid_mask"]     if eis is not None else np.zeros(len(lon_oc), dtype=bool),
        tag        = "(20-year Pelican, unblinded)",
        out_path   = os.path.join(OUT_DIR, "fig_geo_cca.png"),
    )

    print("\n" + "=" * 60)
    print("Figure 3 — Per-regime r values")
    print("=" * 60)
    if eis is not None:
        plot_regime_r(
            X_vae       = X_vae_oc,
            sst         = sst_oc,
            eis         = eis,
            lat         = lat_oc,
            lon         = lon_oc,
            months      = mon_oc,
            n_pca       = N_PCA_VAE,
            global_r_sst = pipe_vae_sst["r_pearson"],
            global_r_eis = pipe_vae_eis["r_pearson"],
            tag         = "(20-year Pelican, unblinded)",
            out_path    = os.path.join(OUT_DIR, "fig_regime_r.png"),
        )

    pipe_multi = None
    if eis is not None and omega500 is not None:
        print("\n" + "=" * 60)
        print("Run 4 — VAE x [EIS, SST, omega500] multivariate CCA")
        print("=" * 60)
        Y_multi = np.column_stack([eis, sst_oc, omega500])
        pipe_multi = run_cca_multi(
            X_vae_oc, Y_multi,
            target_names=["EIS", "SST", "omega500"],
            lat=lat_oc, months=mon_oc,
            n_pca=N_PCA_VAE, tag="VAE-multi",
        )
        for k, (pr, sr) in enumerate(zip(pipe_multi["rs_pearson"], pipe_multi["rs_spearman"])):
            results.append((f"VAE-CCA{k+1}", "[EIS,SST,w500]", pr, sr))

    # 5. Bin-composite walks (needs checkpoint)
    has_ckpt = os.path.exists(CHECKPOINT)
    if not has_ckpt:
        print(f"\nCheckpoint not found at {CHECKPOINT} — skipping decoded walks.")
        print("To enable: copy lightning_model_50_transform.pt to that path.")
    else:
        print("\nBin-composite VAE walk — SST...")
        bin_composite_walk(
            X_vae      = X_vae_oc,
            var_vals   = sst_oc,
            lat        = lat_oc,
            months     = mon_oc,
            tag        = "(20-year Pelican, unblinded)",
            phys_label = "SST (°C)",
            out_path   = os.path.join(OUT_DIR, "walk_composite_sst.png"),
        )
        print("\nBin-mosaic VAE walk — SST...")
        bin_mosaic_walk(
            X_vae      = X_vae_oc,
            var_vals   = sst_oc,
            lat        = lat_oc,
            months     = mon_oc,
            tag        = "(20-year Pelican, unblinded)",
            phys_label = "SST (°C)",
            out_path   = os.path.join(OUT_DIR, "walk_mosaic_sst.png"),
        )
        print("\nPCA-component walk — SST...")
        pca_component_walk(
            X_vae        = X_vae_oc,
            target_vals  = sst_oc,
            lat          = lat_oc,
            months       = mon_oc,
            tag          = "(20-year Pelican, unblinded)",
            phys_label   = "SST (°C)",
            out_path     = os.path.join(OUT_DIR, "walk_pca_sst.png"),
        )
        if eis is not None:
            print("\nBin-composite VAE walk — EIS...")
            bin_composite_walk(
                X_vae      = X_vae_oc,
                var_vals   = eis,
                lat        = lat_oc,
                months     = mon_oc,
                tag        = "(20-year Pelican, unblinded)",
                phys_label = "EIS (K)",
                out_path   = os.path.join(OUT_DIR, "walk_composite_eis.png"),
            )
            print("\nBin-mosaic VAE walk — EIS...")
            bin_mosaic_walk(
                X_vae      = X_vae_oc,
                var_vals   = eis,
                lat        = lat_oc,
                months     = mon_oc,
                tag        = "(20-year Pelican, unblinded)",
                phys_label = "EIS (K)",
                out_path   = os.path.join(OUT_DIR, "walk_mosaic_eis.png"),
            )
            print("\nPCA-component walk — EIS...")
            pca_component_walk(
                X_vae       = X_vae_oc,
                target_vals = eis,
                lat         = lat_oc,
                months      = mon_oc,
                tag         = "(20-year Pelican, unblinded)",
                phys_label  = "EIS (K)",
                out_path    = os.path.join(OUT_DIR, "walk_pca_eis.png"),
            )

        if pipe_multi is not None:
            print("\nMultivariate CCA component walk — [EIS, SST, omega500]...")
            X_multi_valid = X_vae_oc[pipe_multi["valid_mask"]]
            cca_component_walk(
                X_vae        = X_multi_valid,
                cca_scores   = pipe_multi["cca_scores"],
                target_names = pipe_multi["target_names"],
                canonical_rs = pipe_multi["rs_pearson"],
                tag          = "(20-year Pelican, unblinded)",
                out_path     = os.path.join(OUT_DIR, "walk_cca_multi.png"),
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

    # 7. Log to W&B
    wandb_metrics = {
        "n_days":        n_loaded,
        "n_ocean_tiles": len(df_ocean),
        "T2V_SST_pearson_r":  pipe_t2v["r_pearson"],
        "T2V_SST_spearman_r": pipe_t2v["r_spearman"],
        "VAE_SST_pearson_r":  pipe_vae_sst["r_pearson"],
        "VAE_SST_spearman_r": pipe_vae_sst["r_spearman"],
    }
    if eis is not None:
        wandb_metrics["VAE_EIS_pearson_r"]  = pipe_vae_eis["r_pearson"]
        wandb_metrics["VAE_EIS_spearman_r"] = pipe_vae_eis["r_spearman"]
    if pipe_multi is not None:
        for k, (pr, sr) in enumerate(zip(pipe_multi["rs_pearson"], pipe_multi["rs_spearman"])):
            wandb_metrics[f"VAE_multi_CCA{k+1}_pearson_r"]  = pr
            wandb_metrics[f"VAE_multi_CCA{k+1}_spearman_r"] = sr

    png_keys = [
        "walk_composite_sst", "walk_mosaic_sst", "walk_pca_sst",
        "walk_composite_eis", "walk_mosaic_eis", "walk_pca_eis",
        "walk_cca_multi",
        "fig_umap", "fig_geo_cca", "fig_regime_r",
    ]
    for key in png_keys:
        p = os.path.join(OUT_DIR, f"{key}.png")
        if os.path.exists(p):
            wandb_metrics[key] = wandb.Image(p)

    wandb.log(wandb_metrics)
    wandb.save(results_path)
    wandb.finish()

    print("\nDone.")


if __name__ == "__main__":
    main()
