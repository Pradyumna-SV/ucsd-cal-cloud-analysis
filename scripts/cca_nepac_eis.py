
#!/usr/bin/env python3
"""
NE-Pacific curated tile CCA — Nautilus job script.

Pipeline:
  1. SSH into the data server, upload remote_filter_extract.py, run it.
     The remote script filters MODIS tiles (NE Pacific, ocean, daytime,
     cloud fraction >= 0.4) and extracts raw anchor images (3x128x128).
  2. SCP nepac_tiles.npy + nepac_meta.npz back to /workspace.
  3. Run VAE encoder inference (GPU) to get 2048-D embeddings.
  4. Match ERA5 EIS via ARCO-ERA5 on GCS (batched + disk-cached).
  5. Deconfounded PCA -> CCA(1) against EIS.
  6. Save bin-mosaic latent walk figure.

SSH placeholders (fill via env vars or Kubernetes secret):
  SSH_HOST        hostname of the data server
  SSH_USER        login username
  SSH_KEY_PATH    path to the private key file (inside the container)

Remote data paths (set to match the data server layout):
  REMOTE_COORD_DIR   path to coordinates_data_*.json files
  REMOTE_MYD06_DIR   path to MYD06_L2 NetCDF files
  REMOTE_IMG_DIR     path to orig_memmap*.memmap files
  REMOTE_TMP_DIR     writable scratch dir on the remote server
"""

import hashlib
import json
import os
import re
import warnings
from datetime import datetime
from pathlib import Path

import gcsfs
import matplotlib
import wandb
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
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────────────
OUT_DIR    = os.environ.get("OUT_DIR",    "/workspace/results/cca_nepac_eis")
CACHE_DIR  = os.environ.get("CACHE_DIR",  OUT_DIR)
CHECKPOINT = os.environ.get("CHECKPOINT", "/workspace/vae_checkpoint/lightning_model_50_transform.pt")
WORK_DIR   = os.environ.get("WORK_DIR",   "/workspace/nepac_scratch")

N_PCA_VAE    = int(os.environ.get("N_PCA_VAE",    50))
N_WALK_STEPS = int(os.environ.get("N_WALK_STEPS", 9))
N_SAMPLES    = int(os.environ.get("N_SAMPLES",    3))
WALK_SIGMA   = float(os.environ.get("WALK_SIGMA", 1.5))
VAE_BATCH    = int(os.environ.get("VAE_BATCH",    64))

# ── SSH config ───────────────────────────────────────────────────────────────
SSH_HOST     = os.environ.get("SSH_HOST",     "ds-serv8.ucsd.edu")
SSH_PORT     = int(os.environ.get("SSH_PORT", "31849"))
SSH_USER     = os.environ.get("SSH_USER",     "sukhanna")
SSH_PASSWORD = os.environ.get("SSH_PASSWORD", "")   # injected via Kubernetes secret

# ── Remote data paths (on the data server) ───────────────────────────────────
REMOTE_COORD_DIR = os.environ.get("REMOTE_COORD_DIR", "FILL_IN_COORD_DIR")
REMOTE_MYD06_DIR = os.environ.get("REMOTE_MYD06_DIR", "FILL_IN_MYD06_DIR")
REMOTE_IMG_DIR   = os.environ.get("REMOTE_IMG_DIR",   "FILL_IN_IMG_DIR")
REMOTE_TMP_DIR   = os.environ.get("REMOTE_TMP_DIR",   "FILL_IN_TMP_DIR")

REMOTE_N_FILES   = int(os.environ.get("REMOTE_N_FILES",   100))
REMOTE_N_PER_FILE = int(os.environ.get("REMOTE_N_PER_FILE", 10_000))

LAT_MIN = float(os.environ.get("LAT_MIN", 20.0))
LAT_MAX = float(os.environ.get("LAT_MAX", 65.0))
LON_MIN = float(os.environ.get("LON_MIN", -180.0))
LON_MAX = float(os.environ.get("LON_MAX", -110.0))
CF_THRESH = float(os.environ.get("CF_THRESH", 0.4))

WANDB_PROJECT  = os.environ.get("WANDB_PROJECT",  "ucsd-cal-cloud-cca")
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME", None)

ARCO_ERA5 = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(WORK_DIR,  exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# ── SSH: filter + extract tiles on remote server ─────────────────────────────
def ssh_extract_tiles():
    """
    Upload remote_filter_extract.py to the data server, run it,
    and SCP nepac_tiles.npy + nepac_meta.npz back to WORK_DIR.
    Returns (local_tiles_path, local_meta_path).
    """
    import paramiko

    local_tiles = os.path.join(WORK_DIR, "nepac_tiles.npy")
    local_meta  = os.path.join(WORK_DIR, "nepac_meta.npz")

    if os.path.exists(local_tiles) and os.path.exists(local_meta):
        print("  Tile cache found locally — skipping SSH extraction.")
        return local_tiles, local_meta

    print(f"Connecting to {SSH_USER}@{SSH_HOST}:{SSH_PORT} ...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASSWORD)

    # Ensure the remote tmp dir exists before transferring anything.
    _, stdout_mk, _ = client.exec_command(f"mkdir -p {REMOTE_TMP_DIR}")
    stdout_mk.channel.recv_exit_status()

    # Install required packages on the remote server if not already present.
    REMOTE_PYTHON = os.environ.get("REMOTE_PYTHON", "/home/sukhanna/miniconda3/bin/python")
    print("  Installing remote dependencies...")
    _, stdout_pip, stderr_pip = client.exec_command(
        f"{REMOTE_PYTHON} -m pip install --quiet netCDF4 pyhdf global-land-mask scipy numpy 2>&1"
    )
    pip_exit = stdout_pip.channel.recv_exit_status()
    if pip_exit != 0:
        for line in stderr_pip:
            print(f"  [remote pip] {line.rstrip()}")
        raise RuntimeError(f"Remote pip install failed with exit code {pip_exit}")
    print("  Remote dependencies ready.")

    sftp = client.open_sftp()

    # Upload the remote filter script.
    remote_script = f"{REMOTE_TMP_DIR}/remote_filter_extract.py"
    local_script  = str(Path(__file__).parent / "remote_filter_extract.py")
    print(f"  Uploading {local_script} -> {remote_script}")
    sftp.put(local_script, remote_script)

    # Print the top-level MYD06 directory layout so we can see the structure.
    print("  Checking remote MYD06 directory structure...")
    _, stdout_ls, _ = client.exec_command(
        f"ls {REMOTE_MYD06_DIR} 2>&1 | head -20"
    )
    for line in stdout_ls:
        print(f"  [remote ls] {line.rstrip()}")
    stdout_ls.channel.recv_exit_status()

    # Build the remote command.
    cmd = (
        f"{REMOTE_PYTHON} {remote_script} "
        f"--coord_dir  {REMOTE_COORD_DIR} "
        f"--myd06_dir  {REMOTE_MYD06_DIR} "
        f"--img_dir    {REMOTE_IMG_DIR} "
        f"--output_dir {REMOTE_TMP_DIR} "
        f"--n_files    {REMOTE_N_FILES} "
        f"--n_per_file {REMOTE_N_PER_FILE} "
        f"--lat_min    {LAT_MIN} "
        f"--lat_max    {LAT_MAX} "
        f"--lon_min    {LON_MIN} "
        f"--lon_max    {LON_MAX} "
        f"--cf_thresh  {CF_THRESH}"
    )
    print(f"  Running remote command:\n    {cmd}")
    _, stdout, stderr = client.exec_command(cmd)

    remote_tiles_path = None
    remote_meta_path  = None
    for line in stdout:
        line = line.rstrip()
        print(f"  [remote] {line}")
        if line.startswith("SAVED_TILES="):
            remote_tiles_path = line.split("=", 1)[1]
        elif line.startswith("SAVED_META="):
            remote_meta_path = line.split("=", 1)[1]
    exit_code = stdout.channel.recv_exit_status()

    for line in stderr:
        print(f"  [remote stderr] {line.rstrip()}")

    if exit_code != 0:
        client.close()
        raise RuntimeError(f"Remote filter script failed with exit code {exit_code}")
    if not remote_tiles_path or not remote_meta_path:
        client.close()
        raise RuntimeError("Remote script did not print SAVED_TILES / SAVED_META paths.")

    # SCP files back.
    print(f"  SCP {remote_tiles_path} -> {local_tiles}")
    sftp.get(remote_tiles_path, local_tiles)
    print(f"  SCP {remote_meta_path} -> {local_meta}")
    sftp.get(remote_meta_path, local_meta)

    sftp.close()
    client.close()
    print("  SSH extraction complete.")
    return local_tiles, local_meta


# ── VAE inference ─────────────────────────────────────────────────────────────
def run_vae_inference(tiles: np.ndarray) -> np.ndarray:
    """
    Run the VAE encoder on (N, 3, 128, 128) float32 tiles.
    Returns (N, 2048) mean vectors.
    """
    import sys
    import torch
    sys.path.insert(0, str(Path(__file__).parent))
    from vae import VAELightningModule

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = VAELightningModule.load_from_checkpoint(CHECKPOINT)
    model.to(device).eval()
    print(f"  VAE loaded on {device}")

    embeddings = []
    n = len(tiles)
    for start in range(0, n, VAE_BATCH):
        end    = min(start + VAE_BATCH, n)
        batch  = torch.tensor(tiles[start:end], dtype=torch.float32).to(device)
        with torch.no_grad():
            mean, _ = model.encoder(batch)
        embeddings.append(mean.cpu().numpy())
        if (start // VAE_BATCH) % 20 == 0:
            print(f"  [{end}/{n}] inference...")

    X = np.vstack(embeddings).astype("float32")
    print(f"  Embeddings: {X.shape}")
    return X


# ── ERA5 (copied verbatim from cca_20yr.py) ──────────────────────────────────
_ERA5_SPECS = [
    ("sea_surface_temperature",  None,  "sst_raw"),
    ("temperature",              700,   "T700"),
    ("temperature",              850,   "T850"),
    ("temperature",              1000,  "T1000"),
    ("specific_humidity",        850,   "q850"),
    ("specific_humidity",        1000,  "q1000"),
    ("geopotential",             700,   "Phi700"),
    ("2m_temperature",           None,  "T2m"),
    ("vertical_velocity",        500,   "omega500"),
]


def open_era5():
    print("Opening ARCO-ERA5 (GCS, anonymous)...")
    gcs  = gcsfs.GCSFileSystem(token="anon")
    ds   = xr.open_zarr(gcs.get_mapper(ARCO_ERA5), chunks=None)
    lats = ds["latitude"].values
    lons = ds["longitude"].values
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    tree = cKDTree(np.column_stack([lat_grid.ravel(), lon_grid.ravel()]))
    print(f"  ERA5 grid: {len(lons)} lon x {len(lats)} lat")
    print(f"  ERA5 time: {str(ds['time'].values[0])[:10]} -> {str(ds['time'].values[-1])[:10]}")
    return ds, tree


def _match_all_batched(ds, tree, df):
    lats      = df["lat"].values
    lons      = df["lon"].values
    times     = pd.to_datetime(df["time"], errors="coerce")
    lons_pos  = np.where(lons < 0, lons + 360, lons)
    _, sp_idx = tree.query(np.column_stack([lats, lons_pos]))

    outputs    = {key: np.full(len(df), np.nan) for _, _, key in _ERA5_SPECS}
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
                    pass
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


def _compute_eis(raw):
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


def _era5_cache_key(df_meta: pd.DataFrame) -> str:
    times = pd.to_datetime(df_meta["time"], errors="coerce").dropna()
    dates = sorted({f"{t.year:04d}-{t.month:02d}-{t.day:02d}" for t in times})
    return hashlib.md5("|".join(dates).encode()).hexdigest()


def _save_era5_cache(path: Path, key: str, eis):
    np.savez_compressed(str(path), _key=np.array([key]), eis=eis)
    print(f"  ERA5 cache saved -> {path}")


def _load_era5_cache(path: Path, key: str):
    if not path.exists():
        return None
    try:
        c = np.load(str(path), allow_pickle=False)
        if str(c["_key"][0]) != key:
            print("  ERA5 cache key mismatch — re-matching from GCS.")
            return None
        print(f"  ERA5 cache HIT  ({path.name})")
        return c["eis"]
    except Exception as e:
        print(f"  ERA5 cache load error ({e}) — re-matching from GCS.")
        return None


# ── CCA (copied verbatim from cca_20yr.py) ───────────────────────────────────
def run_cca(X, target, lat, months, n_pca, tag=""):
    valid  = np.isfinite(target)
    X, target, lat, months = X[valid], target[valid], lat[valid], months[valid]

    C = np.column_stack([lat, lat**2,
                         np.sin(2 * np.pi * months / 12),
                         np.cos(2 * np.pi * months / 12)])
    reg      = LinearRegression(fit_intercept=True).fit(C, X)
    X_deconf = X - reg.predict(C)

    sx = StandardScaler(); sy = StandardScaler()
    X_sc = sx.fit_transform(X_deconf)
    Y_sc = sy.fit_transform(target.reshape(-1, 1))

    n_pca_act = min(n_pca, X.shape[1] - 1, X.shape[0] // 10)
    pca  = PCA(n_components=n_pca_act, random_state=42)
    X_pca = pca.fit_transform(X_sc)
    print(f"  [{tag}] PCA({n_pca_act}): {pca.explained_variance_ratio_.sum()*100:.1f}% variance retained")

    X_tr, X_te, Y_tr, Y_te = train_test_split(X_pca, Y_sc, test_size=0.2, random_state=42)
    cca = CCA(n_components=1, max_iter=1000).fit(X_tr, Y_tr)
    Xc, Yc = cca.transform(X_te, Y_te)
    pr, _ = pearsonr(Xc.flatten(), Yc.flatten())
    sr, _ = spearmanr(Xc.flatten(), Yc.flatten())
    print(f"  [{tag}] Pearson r = {pr:.4f}   Spearman rho = {sr:.4f}  (20% test set)")

    phys_dir  = pca.components_.T @ cca.x_weights_.flatten()
    phys_dir /= np.linalg.norm(phys_dir)
    Xc_all, _ = cca.transform(X_pca, Y_sc)

    return dict(r_pearson=pr, r_spearman=sr,
                physics_dir=phys_dir, scaler_X=sx,
                X_deconf=X_deconf, X_raw=X,
                physics_scores=Xc_all.flatten(),
                valid_mask=valid)


# ── Bin-mosaic walk (copied verbatim from cca_20yr.py) ───────────────────────
def bin_mosaic_walk(X_vae, var_vals, lat, months, tag, phys_label, out_path,
                    n_bins=9, n_samples=3, seed=42):
    import sys
    import torch
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
            out = model.decoder(z).cpu().numpy()
        tiles = np.clip((out + 1) / 2, 0, 1).transpose(0, 2, 3, 1)
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


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config=dict(
            lat_min=LAT_MIN, lat_max=LAT_MAX,
            lon_min=LON_MIN, lon_max=LON_MAX,
            cf_thresh=CF_THRESH,
            n_pca_vae=N_PCA_VAE,
            n_walk_steps=N_WALK_STEPS,
            n_samples=N_SAMPLES,
            checkpoint=os.path.basename(CHECKPOINT),
        ),
    )

    # 1. SSH: filter tiles on remote server and SCP back.
    print("=" * 60)
    print("Step 1 — Remote tile filtering + extraction (SSH)")
    print("=" * 60)
    tiles_path, meta_path = ssh_extract_tiles()

    tiles = np.load(tiles_path)                   # (N, 3, 128, 128) float32
    meta  = np.load(meta_path, allow_pickle=True)
    lat   = meta["lat"].astype("float32")
    lon   = meta["lon"].astype("float32")
    times = pd.to_datetime(meta["time"])
    months = times.month.values.astype("float32")

    print(f"  Loaded {len(tiles):,} tiles  lat[{lat.min():.1f}, {lat.max():.1f}]  "
          f"lon[{lon.min():.1f}, {lon.max():.1f}]")

    # 2. VAE inference.
    print("\n" + "=" * 60)
    print("Step 2 — VAE encoder inference")
    print("=" * 60)
    X_vae = run_vae_inference(tiles)

    # 3. ERA5 EIS matching (batched + cached).
    print("\n" + "=" * 60)
    print("Step 3 — ERA5 EIS matching")
    print("=" * 60)
    df_meta    = pd.DataFrame({"lat": lat, "lon": lon, "time": times})
    cache_key  = _era5_cache_key(df_meta)
    cache_path = Path(CACHE_DIR) / "era5_nepac.npz"
    eis        = _load_era5_cache(cache_path, cache_key)

    if eis is None:
        print("Opening ERA5...")
        ds_era5, era5_tree = open_era5()
        print("Matching all ERA5 variables (batched, single pass per day)...")
        raw = _match_all_batched(ds_era5, era5_tree, df_meta)
        try:
            eis = _compute_eis(raw)
            print(f"  EIS valid: {np.isfinite(eis).sum():,} / {len(eis):,}")
        except Exception as e:
            raise RuntimeError(f"EIS computation failed: {e}")
        _save_era5_cache(cache_path, cache_key, eis)

    # 4. CCA.
    print("\n" + "=" * 60)
    print("Step 4 — VAE x EIS CCA (NE Pacific, deconfounded)")
    print("=" * 60)
    pipe = run_cca(X_vae, eis, lat, months, N_PCA_VAE, tag="VAE-EIS-NePac")

    results_path = os.path.join(OUT_DIR, "cca_results.txt")
    with open(results_path, "w") as f:
        f.write("NE-Pacific curated tile CCA results\n")
        f.write(f"Tiles: {len(tiles):,}  (NE Pac, ocean, daytime, CF>=0.4)\n")
        f.write(f"Deconfounded by: lat, lat^2, sin(month), cos(month)\n")
        f.write(f"r from 20% held-out test set\n\n")
        f.write(f"{'Embedding':<8} {'Target':<6} {'Pearson r':>10} {'Spearman rho':>13}\n")
        f.write("-" * 42 + "\n")
        f.write(f"{'VAE':<8} {'EIS':<6} {pipe['r_pearson']:>10.4f} {pipe['r_spearman']:>13.4f}\n")
    print(f"\nResults saved -> {results_path}")

    # 5. Bin-mosaic walk.
    print("\n" + "=" * 60)
    print("Step 5 — Bin-mosaic latent walk (EIS)")
    print("=" * 60)
    if not os.path.exists(CHECKPOINT):
        print(f"Checkpoint not found at {CHECKPOINT} — skipping mosaic walk.")
    else:
        bin_mosaic_walk(
            X_vae      = X_vae,
            var_vals   = eis,
            lat        = lat,
            months     = months,
            tag        = "(NE Pacific, curated, ocean+daytime+CF≥0.4)",
            phys_label = "EIS (K)",
            out_path   = os.path.join(OUT_DIR, "walk_mosaic_eis_nepac.png"),
            n_bins     = N_WALK_STEPS,
            n_samples  = N_SAMPLES,
        )

    # 6. Log to W&B.
    wandb_metrics = {
        "n_tiles":            len(tiles),
        "n_eis_valid":        int(np.isfinite(eis).sum()),
        "VAE_EIS_pearson_r":  pipe["r_pearson"],
        "VAE_EIS_spearman_r": pipe["r_spearman"],
    }
    mosaic_path = os.path.join(OUT_DIR, "walk_mosaic_eis_nepac.png")
    if os.path.exists(mosaic_path):
        wandb_metrics["walk_mosaic_eis_nepac"] = wandb.Image(mosaic_path)
    wandb.log(wandb_metrics)
    wandb.save(results_path)
    wandb.finish()

    print("\nDone.")


if __name__ == "__main__":
    main()
