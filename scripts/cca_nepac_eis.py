
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

# Regress SST out of embeddings before VAE×EIS CCA (and EIS mosaic bins).
SST_FIXED_EIS = os.environ.get("SST_FIXED_EIS", "0").strip().lower() in ("1", "true", "yes")
# Bin mosaic by CCA latent score along EIS direction (else by EIS residual vs lat/season[/SST]).
BIN_BY_CCA_SCORE = os.environ.get("BIN_BY_CCA_SCORE", "0").strip().lower() in ("1", "true", "yes")
# CPU-only mode: populate ERA5 cache, then exit before VAE/GPU work.
ERA5_PREP_ONLY = os.environ.get("ERA5_PREP_ONLY", "0").strip().lower() in ("1", "true", "yes")
# GPU-safety mode: fail instead of matching ERA5 while holding a GPU.
REQUIRE_ERA5_CACHE = os.environ.get("REQUIRE_ERA5_CACHE", "0").strip().lower() in ("1", "true", "yes")
# Publication figure: controlled decoded traversal along learned CCA direction.
MAKE_HERO_TRAVERSAL = os.environ.get("MAKE_HERO_TRAVERSAL", "1").strip().lower() in ("1", "true", "yes")
HERO_N_STEPS = int(os.environ.get("HERO_N_STEPS", 7))
HERO_N_BASES = int(os.environ.get("HERO_N_BASES", 6))
MAKE_SST_HERO = os.environ.get("MAKE_SST_HERO", "0").strip().lower() in ("1", "true", "yes")
# On-manifold decoder figure: interpolate between real environmental prototypes.
MAKE_PROTO_INTERP = os.environ.get("MAKE_PROTO_INTERP", "1").strip().lower() in ("1", "true", "yes")
PROTO_N_STEPS = int(os.environ.get("PROTO_N_STEPS", 9))

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


def _compute_sst(raw):
    r = raw["sst_raw"]
    return np.where(r > 200, r - 273.15, np.nan)


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
    raw = (f"bbox={LAT_MIN},{LAT_MAX},{LON_MIN},{LON_MAX}|" + "|".join(dates))
    return hashlib.md5(raw.encode()).hexdigest()


def _save_era5_cache(path: Path, key: str, eis, sst=None):
    np.savez_compressed(
        str(path),
        _key     = np.array([key]),
        eis      = eis,
        sst      = sst if sst is not None else np.array([np.nan]),
        _has_sst = np.array([sst is not None]),
    )
    print(f"  ERA5 cache saved -> {path}")


def _load_era5_cache(path: Path, key: str):
    if not path.exists():
        return None
    try:
        c = np.load(str(path), allow_pickle=False)
        if str(c["_key"][0]) != key:
            print("  ERA5 cache key mismatch — re-matching from GCS.")
            return None
        eis = c["eis"]
        if "_has_sst" in c.files:
            sst = c["sst"] if bool(c["_has_sst"][0]) else None
        else:
            sst = None
        print(f"  ERA5 cache HIT  ({path.name})")
        return eis, sst
    except Exception as e:
        print(f"  ERA5 cache load error ({e}) — re-matching from GCS.")
        return None


# ── CCA (copied from cca_20yr.py; SST optional confound) ─────────────────────
def run_cca(X, target, lat, months, n_pca, tag="", sst_for_confound=None):
    valid = np.isfinite(target)
    if sst_for_confound is not None:
        valid &= np.isfinite(sst_for_confound)
    X, target, lat, months = X[valid], target[valid], lat[valid], months[valid]
    if sst_for_confound is not None:
        sst_c = sst_for_confound[valid].astype("float64")
    else:
        sst_c = None

    C = np.column_stack([lat, lat**2,
                         np.sin(2 * np.pi * months / 12),
                         np.cos(2 * np.pi * months / 12)])
    if sst_c is not None:
        C = np.column_stack([C, sst_c])
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
                    n_bins=9, n_samples=3, seed=42, sst_for_residual=None,
                    bin_by_cca=False):
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
    if not bin_by_cca and sst_for_residual is not None:
        valid &= np.isfinite(sst_for_residual)
    X_v   = X_vae[valid].astype("float32")
    y_v   = var_vals[valid]
    lat_v = lat[valid]
    mon_v = months[valid]
    print(f"  [{phys_label}] {valid.sum():,} valid tiles for mosaic walk")

    if bin_by_cca:
        y_bin = y_v
        bin_label = "CCA score (EIS direction)"
    else:
        C = np.column_stack([lat_v, lat_v**2,
                             np.sin(2 * np.pi * mon_v / 12),
                             np.cos(2 * np.pi * mon_v / 12)])
        if sst_for_residual is not None:
            C = np.column_stack([C, sst_for_residual[valid].astype("float64")])
        y_bin = y_v - LinearRegression(fit_intercept=True).fit(C, y_v).predict(C)
        bin_label = ("lat/season/SST-deconfounded residual"
                     if sst_for_residual is not None
                     else "lat/season-deconfounded residual")

    edges = np.percentile(y_bin, np.linspace(0, 100, n_bins + 1))
    edges[0] -= 1e-6; edges[-1] += 1e-6

    all_tiles, bin_medians, bin_ns = [], [], []
    for i in range(n_bins):
        mask  = (y_bin >= edges[i]) & (y_bin < edges[i + 1])
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
        f"(each panel = individual decoded tile; binned by {bin_label})",
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


def _tiles_to_rgb(tile_batch):
    """Convert stored CHW tiles to display-ready RGB in [0, 1]."""
    arr = np.asarray(tile_batch, dtype="float32")
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.shape[1] == 3:
        arr = arr.transpose(0, 2, 3, 1)
    finite = arr[np.isfinite(arr)]
    if finite.size and np.nanmax(finite) > 2.0:
        arr = arr / 255.0
    elif finite.size and np.nanmin(finite) < -0.05:
        arr = (arr + 1.0) / 2.0
    return np.clip(arr, 0, 1)


def _decode_latents(model, device, z, batch_size=64):
    import torch

    outs = []
    z = np.asarray(z, dtype="float32")
    for start in range(0, len(z), batch_size):
        with torch.no_grad():
            z_t = torch.tensor(z[start:start + batch_size], dtype=torch.float32).to(device)
            out = model.decoder(z_t).cpu().numpy()
        outs.append(out)
    decoded = np.concatenate(outs, axis=0)
    return np.clip((decoded + 1) / 2, 0, 1).transpose(0, 2, 3, 1)


def _corr_or_nan(a, b):
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def cca_latent_traversal_hero(X_vae, tiles, pipe, target_vals, aux_sst, lat, months,
                              tag, target_name, target_units, out_path,
                              candidates_path, n_steps=7, n_bases=6):
    """
    Decode a controlled traversal along the learned CCA direction.

    The direction returned by run_cca lives in standardized deconfounded VAE
    coordinates. We convert a unit step in that space back to raw VAE latent
    coordinates before decoding, then ground the generated path with nearest
    real MODIS tiles.
    """
    import sys
    import torch
    sys.path.insert(0, str(Path(__file__).parent))
    from vae import VAELightningModule

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vm = pipe["valid_mask"]
    X_raw = pipe["X_raw"].astype("float64")
    X_deconf = pipe["X_deconf"].astype("float64")
    X_masked = X_vae[vm]
    target_v = target_vals[vm].astype("float64")
    sst_v = aux_sst[vm].astype("float64") if aux_sst is not None else None
    tiles_v = tiles[vm]

    if len(X_raw) != len(X_masked) or len(X_raw) != len(target_v) or len(X_raw) != len(pipe["physics_scores"]):
        raise ValueError("Hero traversal inputs are not aligned with pipe['valid_mask']")

    scaler = pipe["scaler_X"]
    X_sc = scaler.transform(X_deconf)
    direction = np.asarray(pipe["physics_dir"], dtype="float64")
    direction = direction / np.linalg.norm(direction)
    scores = np.asarray(pipe["physics_scores"], dtype="float64")

    # Orient the displayed walk so columns move toward higher target values.
    direction_coord = X_sc @ direction
    if _corr_or_nan(direction_coord, target_v) < 0:
        direction *= -1
        direction_coord *= -1
    if _corr_or_nan(scores, target_v) < 0:
        scores *= -1

    n_steps = max(3, int(n_steps))
    n_bases = max(1, int(n_bases))
    percentiles = np.array([5, 15, 30, 50, 70, 85, 95], dtype="float64")
    if n_steps != len(percentiles):
        percentiles = np.linspace(5, 95, n_steps)
    target_coords = np.percentile(direction_coord, percentiles)
    raw_direction = scaler.scale_.astype("float64") * direction

    mid_lo, mid_hi = np.percentile(scores, [40, 60])
    mid_mask = (scores >= mid_lo) & (scores <= mid_hi)
    if mid_mask.sum() < n_bases:
        mid_lo, mid_hi = np.percentile(scores, [30, 70])
        mid_mask = (scores >= mid_lo) & (scores <= mid_hi)
    if not np.any(mid_mask):
        mid_mask = np.ones_like(scores, dtype=bool)

    center = np.median(X_sc[mid_mask], axis=0)
    pool = np.where(mid_mask)[0]
    pool_dist = np.linalg.norm(X_sc[pool] - center, axis=1)
    candidate_idxs = pool[np.argsort(pool_dist)[:min(n_bases, len(pool))]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAELightningModule.load_from_checkpoint(CHECKPOINT)
    model.to(device).eval()
    print(f"  Hero traversal VAE loaded on {device}")

    tree = cKDTree(X_raw)
    candidate_decoded, candidate_walks, candidate_nearest = [], [], []
    candidate_metrics = []
    for cand_idx in candidate_idxs:
        base_coord = float(X_sc[cand_idx] @ direction)
        deltas = target_coords - base_coord
        z_walk = X_raw[cand_idx] + np.outer(deltas, raw_direction)
        nearest_dist, nearest_idx = tree.query(z_walk, k=1)
        decoded = _decode_latents(model, device, z_walk, batch_size=VAE_BATCH)
        decoded_base = _decode_latents(model, device, X_raw[cand_idx:cand_idx + 1],
                                       batch_size=1)[0]
        real_base = _tiles_to_rgb(tiles_v[cand_idx])[0]
        recon_mse = float(np.mean((decoded_base - real_base) ** 2))
        score = float(np.median(nearest_dist) + 0.5 * np.max(nearest_dist) + 50.0 * recon_mse)

        candidate_decoded.append(decoded)
        candidate_walks.append(z_walk)
        candidate_nearest.append((nearest_dist, nearest_idx))
        candidate_metrics.append(dict(
            cand_idx=int(cand_idx),
            selection_score=score,
            recon_mse=recon_mse,
            nearest_median=float(np.median(nearest_dist)),
            nearest_max=float(np.max(nearest_dist)),
        ))

    best_pos = int(np.argmin([m["selection_score"] for m in candidate_metrics]))
    best_idx = int(candidate_idxs[best_pos])
    decoded_best = candidate_decoded[best_pos]
    nearest_dist, nearest_idx = candidate_nearest[best_pos]

    edges = np.percentile(direction_coord, np.linspace(0, 100, n_steps + 1))
    medoid_idx, bin_target_median, bin_sst_median, bin_n = [], [], [], []
    for col in range(n_steps):
        lo, hi = edges[col], edges[col + 1]
        if col == n_steps - 1:
            mask = (direction_coord >= lo) & (direction_coord <= hi)
        else:
            mask = (direction_coord >= lo) & (direction_coord < hi)
        if not np.any(mask):
            idx = int(np.argmin(np.abs(direction_coord - target_coords[col])))
            mask = np.zeros_like(direction_coord, dtype=bool)
            mask[idx] = True
        group = np.where(mask)[0]
        group_center = np.median(X_sc[group], axis=0)
        idx = int(group[np.argmin(np.linalg.norm(X_sc[group] - group_center, axis=1))])
        medoid_idx.append(idx)
        bin_target_median.append(float(np.nanmedian(target_v[group])))
        bin_sst_median.append(float(np.nanmedian(sst_v[group])) if sst_v is not None else np.nan)
        bin_n.append(int(len(group)))
    medoid_idx = np.array(medoid_idx, dtype=int)

    nearest_tiles = _tiles_to_rgb(tiles_v[nearest_idx])
    medoid_tiles = _tiles_to_rgb(tiles_v[medoid_idx])

    fig, axes = plt.subplots(3, n_steps, figsize=(n_steps * 2.15, 7.3),
                             facecolor="#111111")
    axes = np.asarray(axes).reshape(3, n_steps)
    row_labels = ["Decoded\nCCA walk", "Nearest\nobserved", "Bin\nmedoid"]
    for col in range(n_steps):
        imgs = [decoded_best[col], nearest_tiles[col], medoid_tiles[col]]
        for row in range(3):
            ax = axes[row, col]
            ax.imshow(imgs[row])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#555")
            if col == 0:
                ax.set_ylabel(row_labels[row], color="white", fontsize=9,
                              fontweight="bold", rotation=0, labelpad=34, va="center")
        sst_txt = f"\nSST {bin_sst_median[col]:.1f}C" if np.isfinite(bin_sst_median[col]) else ""
        axes[0, col].set_title(
            f"p{percentiles[col]:.0f}\n{target_name} {bin_target_median[col]:.1f}{target_units}{sst_txt}",
            color="white", fontsize=8, pad=4)
        axes[2, col].set_xlabel(f"n={bin_n[col]:,}", color="#aaa", fontsize=7)

    corr_txt = _corr_or_nan(direction_coord, target_v)
    fig.suptitle(
        f"VAE morphology traversal along learned {target_name} direction  {tag}\n"
        f"base={best_idx}  r(direction,{target_name})={corr_txt:.2f}  "
        f"filters: ocean, daytime, CF≥{CF_THRESH}"
        + ("  SST-controlled" if SST_FIXED_EIS else ""),
        color="white", fontsize=10, fontweight="bold")
    plt.tight_layout(rect=[0, 0.02, 1, 0.88])
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="#111111")
    print(f"  Saved hero traversal -> {out_path}")
    plt.close()

    fig, axes = plt.subplots(len(candidate_idxs), n_steps,
                             figsize=(n_steps * 1.8, len(candidate_idxs) * 1.75),
                             facecolor="#111111")
    axes = np.asarray(axes).reshape(len(candidate_idxs), n_steps)
    for row, metric in enumerate(candidate_metrics):
        for col in range(n_steps):
            ax = axes[row, col]
            ax.imshow(candidate_decoded[row][col])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#555")
            if row == 0:
                ax.set_title(f"p{percentiles[col]:.0f}", color="white", fontsize=8)
            if col == 0:
                ax.set_ylabel(
                    f"base {metric['cand_idx']}\nscore {metric['selection_score']:.1f}",
                    color="white", fontsize=7, rotation=0, labelpad=36, va="center")
    fig.suptitle(
        f"Candidate decoded {target_name} traversals  {tag}\n"
        "lower score = closer to observed latent manifold + better base reconstruction",
        color="white", fontsize=9, fontweight="bold")
    plt.tight_layout(rect=[0, 0.02, 1, 0.88])
    plt.savefig(candidates_path, dpi=160, bbox_inches="tight", facecolor="#111111")
    print(f"  Saved hero candidates -> {candidates_path}")
    plt.close()

    metrics = {
        "hero_best_base_index": best_idx,
        "hero_best_selection_score": candidate_metrics[best_pos]["selection_score"],
        "hero_base_recon_mse": candidate_metrics[best_pos]["recon_mse"],
        "hero_nearest_latent_distance_median": float(np.median(nearest_dist)),
        "hero_nearest_latent_distance_max": float(np.max(nearest_dist)),
        "hero_direction_target_corr": corr_txt,
        "hero_score_percentile_min": float(percentiles.min()),
        "hero_score_percentile_max": float(percentiles.max()),
    }
    return dict(
        hero_path=out_path,
        candidates_path=candidates_path,
        metrics=metrics,
    )


def _visual_quality_scores(tile_batch):
    """Rank tiles for visible cloud texture/contrast using only the displayed pixels."""
    rgb = _tiles_to_rgb(tile_batch)
    lum = rgb.mean(axis=3)
    mean = lum.mean(axis=(1, 2))
    std = lum.std(axis=(1, 2))
    p05 = np.percentile(lum, 5, axis=(1, 2))
    p95 = np.percentile(lum, 95, axis=(1, 2))
    contrast = p95 - p05
    grad_x = np.abs(np.diff(lum, axis=2)).mean(axis=(1, 2))
    grad_y = np.abs(np.diff(lum, axis=1)).mean(axis=(1, 2))
    texture = grad_x + grad_y
    saturation = (rgb.max(axis=3) - rgb.min(axis=3)).mean(axis=(1, 2))

    score = 1.8 * std + 1.3 * contrast + 8.0 * texture + 0.5 * saturation
    score -= np.maximum(0, 0.18 - mean) * 2.0
    score -= np.maximum(0, mean - 0.86) * 2.0
    score -= np.maximum(0, 0.08 - std) * 3.0
    return score.astype("float64")


def _choose_quality_medoid(X_sc, tiles_rgb, idxs):
    idxs = np.asarray(idxs, dtype=int)
    center = np.median(X_sc[idxs], axis=0)
    dist = np.linalg.norm(X_sc[idxs] - center, axis=1)
    quality = _visual_quality_scores(tiles_rgb[idxs])

    dist_z = (dist - np.median(dist)) / (np.std(dist) + 1e-6)
    quality_z = (quality - np.median(quality)) / (np.std(quality) + 1e-6)
    # Favor representative latents, but require visually informative cloud texture.
    rank = dist_z - 0.8 * quality_z
    return int(idxs[np.argmin(rank)])


def prototype_interpolation_hero(X_vae, tiles, score_vals, target_vals, aux_sst,
                                 tag, score_name, score_units, target_name,
                                 target_units, out_path, n_steps=9):
    """
    Decode a continuous interpolation between real prototype cloud latents.

    Unlike directional extrapolation, every point lies between real observed
    latents selected from low/mid/high environmental bins.
    """
    import sys
    import torch
    sys.path.insert(0, str(Path(__file__).parent))
    from vae import VAELightningModule

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    valid = np.isfinite(score_vals) & np.isfinite(target_vals)
    if aux_sst is not None:
        valid &= np.isfinite(aux_sst)
    X = X_vae[valid].astype("float64")
    score = score_vals[valid].astype("float64")
    target = target_vals[valid].astype("float64")
    sst_v = aux_sst[valid].astype("float64") if aux_sst is not None else None
    tiles_v = tiles[valid]

    if len(X) < 30:
        raise ValueError(f"Not enough valid tiles for {score_name} prototype interpolation")

    if _corr_or_nan(score, target) < 0:
        score *= -1

    X_sc = StandardScaler().fit_transform(X)
    bins = [(5, 25), (40, 60), (75, 95)]
    anchor_idxs, anchor_stats = [], []
    for lo, hi in bins:
        qlo, qhi = np.percentile(score, [lo, hi])
        mask = (score >= qlo) & (score <= qhi)
        if mask.sum() < 5:
            center_q = (lo + hi) / 2
            qlo, qhi = np.percentile(score, [max(0, center_q - 20), min(100, center_q + 20)])
            mask = (score >= qlo) & (score <= qhi)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            idxs = np.array([int(np.argmin(np.abs(score - np.percentile(score, (lo + hi) / 2))))])
        anchor = _choose_quality_medoid(X_sc, tiles_v, idxs)
        anchor_idxs.append(anchor)
        anchor_stats.append(dict(
            q_lo=lo, q_hi=hi, n=int(len(idxs)),
            score=float(np.nanmedian(score[idxs])),
            target=float(np.nanmedian(target[idxs])),
            sst=float(np.nanmedian(sst_v[idxs])) if sst_v is not None else np.nan,
        ))

    low, mid, high = anchor_idxs
    n_steps = max(5, int(n_steps))
    left_n = n_steps // 2 + 1
    right_n = n_steps - left_n + 1
    left = np.array([(1 - a) * X[low] + a * X[mid] for a in np.linspace(0, 1, left_n)])
    right = np.array([(1 - a) * X[mid] + a * X[high] for a in np.linspace(0, 1, right_n)[1:]])
    z_path = np.vstack([left, right]).astype("float32")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAELightningModule.load_from_checkpoint(CHECKPOINT)
    model.to(device).eval()
    print(f"  Prototype interpolation VAE loaded on {device}")

    decoded = _decode_latents(model, device, z_path, batch_size=VAE_BATCH)
    tree = cKDTree(X)
    nearest_dist, nearest_idx = tree.query(z_path, k=1)
    nearest_tiles = _tiles_to_rgb(tiles_v[nearest_idx])
    anchor_tiles = _tiles_to_rgb(tiles_v[anchor_idxs])

    interp_pos = np.r_[np.linspace(0, 0.5, left_n), np.linspace(0.5, 1, right_n)[1:]]
    interp_score = np.interp(interp_pos, [0, 0.5, 1],
                             [anchor_stats[0]["score"], anchor_stats[1]["score"], anchor_stats[2]["score"]])
    interp_target = np.interp(interp_pos, [0, 0.5, 1],
                              [anchor_stats[0]["target"], anchor_stats[1]["target"], anchor_stats[2]["target"]])

    fig, axes = plt.subplots(3, n_steps, figsize=(n_steps * 2.05, 7.1),
                             facecolor="#111111")
    axes = np.asarray(axes).reshape(3, n_steps)
    row_labels = ["Decoded\ncontinuum", "Nearest\nobserved", "Observed\nanchors"]
    anchor_cols = {0: 0, left_n - 1: 1, n_steps - 1: 2}
    for col in range(n_steps):
        imgs = [decoded[col], nearest_tiles[col], None]
        for row in range(3):
            ax = axes[row, col]
            if row == 2:
                if col in anchor_cols:
                    ax.imshow(anchor_tiles[anchor_cols[col]])
                    for sp in ax.spines.values():
                        sp.set_edgecolor("#dddddd")
                        sp.set_linewidth(1.5)
                else:
                    ax.set_facecolor("#111111")
                    ax.text(0.5, 0.5, "interpolation", color="#777777",
                            fontsize=7, ha="center", va="center", transform=ax.transAxes)
                    for sp in ax.spines.values():
                        sp.set_edgecolor("#333333")
            else:
                ax.imshow(imgs[row])
                for sp in ax.spines.values():
                    sp.set_edgecolor("#555555")
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(row_labels[row], color="white", fontsize=9,
                              fontweight="bold", rotation=0, labelpad=34, va="center")

        sst_txt = ""
        if sst_v is not None:
            nearest_sst = sst_v[nearest_idx[col]]
            sst_txt = f"\nSST {nearest_sst:.1f}C"
        axes[0, col].set_title(
            f"{score_name} {interp_score[col]:.2f}{score_units}\n"
            f"{target_name} {interp_target[col]:.1f}{target_units}{sst_txt}",
            color="white", fontsize=7, pad=4)
        axes[1, col].set_xlabel(f"d={nearest_dist[col]:.1f}", color="#aaaaaa", fontsize=7)

    fig.suptitle(
        f"VAE-decoded cloud morphology continuum ordered by {score_name}  {tag}\n"
        "low prototype -> mid prototype -> high prototype; anchors are real cloudy ocean MODIS tiles",
        color="white", fontsize=10, fontweight="bold")
    plt.tight_layout(rect=[0, 0.02, 1, 0.88])
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="#111111")
    print(f"  Saved prototype interpolation -> {out_path}")
    plt.close()

    metric_prefix = re.sub(r"[^a-z0-9]+", "_", score_name.lower()).strip("_")
    return dict(
        path=out_path,
        metrics={
            f"proto_{metric_prefix}_nearest_distance_median": float(np.median(nearest_dist)),
            f"proto_{metric_prefix}_nearest_distance_max": float(np.max(nearest_dist)),
            f"proto_{metric_prefix}_low_anchor": int(low),
            f"proto_{metric_prefix}_mid_anchor": int(mid),
            f"proto_{metric_prefix}_high_anchor": int(high),
        },
    )


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config=dict(
            lat_min=LAT_MIN, lat_max=LAT_MAX,
            lon_min=LON_MIN, lon_max=LON_MAX,
            cf_thresh=CF_THRESH,
            sst_fixed_eis=SST_FIXED_EIS,
            bin_by_cca_score=BIN_BY_CCA_SCORE,
            era5_prep_only=ERA5_PREP_ONLY,
            require_era5_cache=REQUIRE_ERA5_CACHE,
            make_hero_traversal=MAKE_HERO_TRAVERSAL,
            hero_n_steps=HERO_N_STEPS,
            hero_n_bases=HERO_N_BASES,
            make_sst_hero=MAKE_SST_HERO,
            make_proto_interp=MAKE_PROTO_INTERP,
            proto_n_steps=PROTO_N_STEPS,
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

    meta  = np.load(meta_path, allow_pickle=True)
    lat   = meta["lat"].astype("float32")
    lon   = meta["lon"].astype("float32")
    times = pd.to_datetime(meta["time"])
    months = times.month.values.astype("float32")

    print(f"  Loaded metadata for {len(lat):,} tiles  lat[{lat.min():.1f}, {lat.max():.1f}]  "
          f"lon[{lon.min():.1f}, {lon.max():.1f}]")

    # 2. ERA5 EIS matching (batched + cached).
    print("\n" + "=" * 60)
    print("Step 2 — ERA5 EIS matching")
    print("=" * 60)
    df_meta    = pd.DataFrame({"lat": lat, "lon": lon, "time": times})
    cache_key  = _era5_cache_key(df_meta)
    cache_path = Path(CACHE_DIR) / "era5_curated_tiles.npz"
    cached     = _load_era5_cache(cache_path, cache_key)
    eis, sst   = (None, None)
    if cached is not None:
        eis, sst = cached
    if SST_FIXED_EIS and sst is None and eis is not None:
        print("  SST required for SST_FIXED_EIS but missing from cache — re-matching ERA5.")
        eis, sst = None, None

    if eis is None:
        if REQUIRE_ERA5_CACHE:
            raise RuntimeError(
                f"ERA5 cache miss at {cache_path}; run ERA5_PREP_ONLY=1 without a GPU first."
            )
        print("Opening ERA5...")
        ds_era5, era5_tree = open_era5()
        print("Matching all ERA5 variables (batched, single pass per day)...")
        raw = _match_all_batched(ds_era5, era5_tree, df_meta)
        sst = _compute_sst(raw)
        try:
            eis = _compute_eis(raw)
            print(f"  EIS valid: {np.isfinite(eis).sum():,} / {len(eis):,}")
        except Exception as e:
            raise RuntimeError(f"EIS computation failed: {e}")
        _save_era5_cache(cache_path, cache_key, eis, sst)

    if ERA5_PREP_ONLY:
        metrics = {
            "n_tiles":     len(lat),
            "n_eis_valid": int(np.isfinite(eis).sum()),
            "n_sst_valid": int(np.isfinite(sst).sum()) if sst is not None else 0,
            "era5_prep_only": 1,
        }
        wandb.log(metrics)
        wandb.finish()
        print("\nERA5 prep complete; exiting before VAE/GPU steps.")
        return

    # 3. VAE inference.
    print("\n" + "=" * 60)
    print("Step 3 — VAE encoder inference")
    print("=" * 60)
    tiles = np.load(tiles_path)                   # (N, 3, 128, 128) float32
    print(f"  Loaded {len(tiles):,} tiles from {tiles_path}")
    X_vae = run_vae_inference(tiles)

    # 4. CCA.
    print("\n" + "=" * 60)
    print("Step 4 — VAE x EIS CCA (curated tiles, "
          + ("SST partialled from embeddings" if SST_FIXED_EIS else "lat/season deconfounded")
          + ")")
    print("=" * 60)
    sst_conf = sst if SST_FIXED_EIS else None
    pipe = run_cca(X_vae, eis, lat, months, N_PCA_VAE, tag="VAE-EIS-tiles",
                   sst_for_confound=sst_conf)

    results_path = os.path.join(OUT_DIR, "cca_results.txt")
    deconf_txt = (
        "lat, lat^2, sin(month), cos(month); VAE×EIS additionally partials SST from embeddings"
        if SST_FIXED_EIS else "lat, lat^2, sin(month), cos(month)"
    )
    with open(results_path, "w") as f:
        f.write("Curated-tile MODIS cloud embedding CCA (regional bbox)\n")
        f.write(f"Tiles: {len(tiles):,}  (bbox, ocean, daytime, CF>={CF_THRESH})\n")
        f.write(f"Bbox: lat [{LAT_MIN},{LAT_MAX}]  lon [{LON_MIN},{LON_MAX}]\n")
        f.write(f"Deconfounded by: {deconf_txt}\n")
        f.write(f"r from 20% held-out test set\n\n")
        f.write(f"{'Embedding':<8} {'Target':<6} {'Pearson r':>10} {'Spearman rho':>13}\n")
        f.write("-" * 42 + "\n")
        f.write(f"{'VAE':<8} {'EIS':<6} {pipe['r_pearson']:>10.4f} {pipe['r_spearman']:>13.4f}\n")
    print(f"\nResults saved -> {results_path}")

    hero_images = {}
    hero_metrics = {}

    # 5. Controlled hero traversal.
    print("\n" + "=" * 60)
    print("Step 5 — Controlled CCA latent traversal hero")
    print("=" * 60)
    if MAKE_HERO_TRAVERSAL and os.path.exists(CHECKPOINT):
        hero_eis = cca_latent_traversal_hero(
            X_vae           = X_vae,
            tiles           = tiles,
            pipe            = pipe,
            target_vals     = eis,
            aux_sst         = sst,
            lat             = lat,
            months          = months,
            tag             = f"(curated tiles, CF≥{CF_THRESH})",
            target_name     = "EIS",
            target_units    = " K",
            out_path        = os.path.join(OUT_DIR, "hero_cca_traversal_eis.png"),
            candidates_path = os.path.join(OUT_DIR, "hero_cca_traversal_candidates_eis.png"),
            n_steps         = HERO_N_STEPS,
            n_bases         = HERO_N_BASES,
        )
        hero_images["hero_cca_traversal_eis"] = hero_eis["hero_path"]
        hero_images["hero_cca_traversal_candidates_eis"] = hero_eis["candidates_path"]
        hero_metrics.update(hero_eis["metrics"])

        if MAKE_SST_HERO and sst is not None and np.isfinite(sst).sum() > 10:
            print("\nStep 5b — Optional SST latent traversal companion")
            pipe_sst = run_cca(X_vae, sst, lat, months, N_PCA_VAE, tag="VAE-SST-tiles",
                               sst_for_confound=None)
            hero_sst = cca_latent_traversal_hero(
                X_vae           = X_vae,
                tiles           = tiles,
                pipe            = pipe_sst,
                target_vals     = sst,
                aux_sst         = None,
                lat             = lat,
                months          = months,
                tag             = f"(curated tiles, CF≥{CF_THRESH})",
                target_name     = "SST",
                target_units    = " C",
                out_path        = os.path.join(OUT_DIR, "hero_cca_traversal_sst.png"),
                candidates_path = os.path.join(OUT_DIR, "hero_cca_traversal_candidates_sst.png"),
                n_steps         = HERO_N_STEPS,
                n_bases         = HERO_N_BASES,
            )
            hero_images["hero_cca_traversal_sst"] = hero_sst["hero_path"]
            hero_images["hero_cca_traversal_candidates_sst"] = hero_sst["candidates_path"]
            hero_metrics.update({f"sst_{k}": v for k, v in hero_sst["metrics"].items()})
    elif MAKE_HERO_TRAVERSAL:
        print(f"Checkpoint not found at {CHECKPOINT} — skipping hero traversal.")
    else:
        print("MAKE_HERO_TRAVERSAL disabled — skipping hero traversal.")

    # 5c. On-manifold prototype interpolation hero candidates.
    print("\n" + "=" * 60)
    print("Step 5c — Prototype latent interpolation heroes")
    print("=" * 60)
    if MAKE_PROTO_INTERP and os.path.exists(CHECKPOINT):
        proto_eis = prototype_interpolation_hero(
            X_vae        = X_vae,
            tiles        = tiles,
            score_vals   = eis,
            target_vals  = eis,
            aux_sst      = sst,
            tag          = f"(curated tiles, CF≥{CF_THRESH})",
            score_name   = "EIS",
            score_units  = " K",
            target_name  = "EIS",
            target_units = " K",
            out_path     = os.path.join(OUT_DIR, "hero_proto_interp_eis.png"),
            n_steps      = PROTO_N_STEPS,
        )
        hero_images["hero_proto_interp_eis"] = proto_eis["path"]
        hero_metrics.update(proto_eis["metrics"])

        vm = pipe["valid_mask"]
        scores = pipe["physics_scores"]
        if len(X_vae[vm]) != len(scores):
            raise ValueError("Prototype CCA-score inputs are not aligned with pipe['valid_mask']")
        proto_cca = prototype_interpolation_hero(
            X_vae        = X_vae[vm],
            tiles        = tiles[vm],
            score_vals   = scores,
            target_vals  = eis[vm],
            aux_sst      = sst[vm] if sst is not None else None,
            tag          = f"(curated tiles, CF≥{CF_THRESH})",
            score_name   = "CCA score",
            score_units  = "",
            target_name  = "EIS",
            target_units = " K",
            out_path     = os.path.join(OUT_DIR, "hero_proto_interp_ccascore.png"),
            n_steps      = PROTO_N_STEPS,
        )
        hero_images["hero_proto_interp_ccascore"] = proto_cca["path"]
        hero_metrics.update(proto_cca["metrics"])
    elif MAKE_PROTO_INTERP:
        print(f"Checkpoint not found at {CHECKPOINT} — skipping prototype interpolation heroes.")
    else:
        print("MAKE_PROTO_INTERP disabled — skipping prototype interpolation heroes.")

    # 6. Bin-mosaic walk.
    print("\n" + "=" * 60)
    print("Step 6 — Bin-mosaic latent walk (EIS)")
    print("=" * 60)
    if BIN_BY_CCA_SCORE:
        mosaic_name = ("walk_mosaic_eis_ccascore_sstfixed.png" if SST_FIXED_EIS
                       else "walk_mosaic_eis_ccascore.png")
    else:
        mosaic_name = "walk_mosaic_eis_sstfixed.png" if SST_FIXED_EIS else "walk_mosaic_eis.png"
    mosaic_path = os.path.join(OUT_DIR, mosaic_name)
    if not os.path.exists(CHECKPOINT):
        print(f"Checkpoint not found at {CHECKPOINT} — skipping mosaic walk.")
    elif BIN_BY_CCA_SCORE:
        vm = pipe["valid_mask"]
        X_cca = X_vae[vm]
        assert len(X_cca) == len(pipe["physics_scores"]), \
            "Mismatch between valid_mask and physics_scores length"
        bin_mosaic_walk(
            X_vae           = X_cca,
            var_vals        = pipe["physics_scores"],
            lat             = lat[vm],
            months          = months[vm],
            tag             = f"(curated tiles, CF≥{CF_THRESH})",
            phys_label      = "EIS (K)",
            out_path        = mosaic_path,
            n_bins          = N_WALK_STEPS,
            n_samples       = N_SAMPLES,
            sst_for_residual = None,
            bin_by_cca      = True,
        )
    else:
        bin_mosaic_walk(
            X_vae      = X_vae,
            var_vals   = eis,
            lat        = lat,
            months     = months,
            tag        = f"(curated tiles, CF≥{CF_THRESH})",
            phys_label = "EIS (K)",
            out_path   = mosaic_path,
            n_bins     = N_WALK_STEPS,
            n_samples  = N_SAMPLES,
            sst_for_residual = sst if SST_FIXED_EIS else None,
            bin_by_cca = False,
        )

    # 7. Log to W&B.
    wandb_metrics = {
        "n_tiles":            len(tiles),
        "n_eis_valid":        int(np.isfinite(eis).sum()),
        "n_sst_valid":        int(np.isfinite(sst).sum()) if sst is not None else 0,
        "VAE_EIS_pearson_r":  pipe["r_pearson"],
        "VAE_EIS_spearman_r": pipe["r_spearman"],
        "sst_fixed_eis":      int(SST_FIXED_EIS),
        "bin_by_cca_score":   int(BIN_BY_CCA_SCORE),
    }
    wandb_metrics.update(hero_metrics)
    for key, path in hero_images.items():
        if os.path.exists(path):
            wandb_metrics[key] = wandb.Image(path)
    if os.path.exists(mosaic_path):
        if BIN_BY_CCA_SCORE:
            wandb_key = ("walk_mosaic_eis_ccascore_sstfixed" if SST_FIXED_EIS
                         else "walk_mosaic_eis_ccascore")
        else:
            wandb_key = "walk_mosaic_eis_sstfixed" if SST_FIXED_EIS else "walk_mosaic_eis"
        wandb_metrics[wandb_key] = wandb.Image(mosaic_path)
    wandb.log(wandb_metrics)
    wandb.save(results_path)
    wandb.finish()

    print("\nDone.")


if __name__ == "__main__":
    main()
