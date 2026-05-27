#!/usr/bin/env python3
"""
Unified CCF pipeline for NE‑Pacific MODIS tiles.

Environment variable PREP_ONLY:
  PREP_ONLY=1  →  CPU‑only preprocessing: extract MERRA‑2 CCFs + IR means,
                   save to /workspace/ccf_preprocessed/preprocessed.npz,
                   then exit.
  otherwise    →  GPU pipeline: load preprocessed data, run VAE inference,
                   perform IR‑removal + CCF regression + PCA, build atlas.
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ───────────────────────────────────────────────────────────────
PREP_ONLY = os.environ.get("PREP_ONLY", "0").strip().lower() in ("1", "true", "yes")

TILES_PATH = "/workspace/nepac_scratch/nepac_tiles.npy"
META_PATH  = "/workspace/nepac_scratch/nepac_meta.npz"
CCF_PATH   = "/workspace/merra2_2011_CCFs.nc"
CKPT_PATH  = "/workspace/vae_checkpoint/lightning_model_50_transform.pt"
OUT_DIR    = "/workspace/results/ccf_pipeline"
PREP_DIR   = "/workspace/ccf_preprocessed"
VAE_BATCH  = 64

# GPU imports only if needed
if not PREP_ONLY:
    import torch
    from sklearn.linear_model import LinearRegression
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from scipy.spatial import cKDTree
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # to import vae

warnings.filterwarnings("ignore")

# ── CCF extraction (used in PREP_ONLY mode) ──────────────────────────────
def extract_ccf_matrix(lat_arr, lon_arr, times_arr, ccf_nc_path):
    ds = xr.open_dataset(ccf_nc_path)
    var_names = ["TS", "EIS", "TS_adv", "RH700", "w700", "WS"]
    N = len(lat_arr)
    C = np.full((N, 6), np.nan, dtype=np.float64)
    if not isinstance(times_arr, pd.DatetimeIndex):
        times_arr = pd.to_datetime(times_arr)
    month_starts = times_arr.strftime('%Y-%m-01')
    for i in range(N):
        try:
            point = ds.sel(time=month_starts[i], lat=lat_arr[i], lon=lon_arr[i], method="nearest")
            for j, vn in enumerate(var_names):
                C[i, j] = float(point[vn].values)
        except Exception:
            continue
    ds.close()
    return C

# ── PREP ONLY: extract CCFs + IR means, save, exit ───────────────────────
def run_prep():
    print("=== PREP ONLY: extracting CCFs and IR means ===")
    os.makedirs(PREP_DIR, exist_ok=True)

    tiles = np.load(TILES_PATH)
    meta  = np.load(META_PATH, allow_pickle=True)
    lat   = meta["lat"].astype("float32")
    lon   = meta["lon"].astype("float32")
    times = pd.to_datetime(meta["time"])
    print(f"Loaded {len(lat):,} tiles")

    C = extract_ccf_matrix(lat, lon, times, CCF_PATH)
    valid = np.isfinite(C).all(axis=1)
    print(f"Complete CCFs: {valid.sum():,} / {len(lat):,}")

    tiles   = tiles[valid]
    C       = C[valid]
    lat     = lat[valid]
    lon     = lon[valid]
    times   = times[valid]

    ir_mean = tiles[:, 2, :, :].mean(axis=(1, 2))
    print("IR means computed.")

    out_file = os.path.join(PREP_DIR, "preprocessed.npz")
    np.savez_compressed(out_file,
                        tiles   = tiles,
                        C       = C,
                        ir_mean = ir_mean,
                        lat     = lat,
                        lon     = lon,
                        times   = times.to_numpy())
    print(f"Saved preprocessed data -> {out_file}")

# ── GPU pipeline (after prep) ────────────────────────────────────────────
def run_vae_inference(tiles):
    from vae import VAELightningModule
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAELightningModule.load_from_checkpoint(CKPT_PATH)
    model.to(device).eval()
    print(f"VAE loaded on {device}")
    embeddings = []
    n = len(tiles)
    for start in range(0, n, VAE_BATCH):
        end = min(start + VAE_BATCH, n)
        batch = torch.tensor(tiles[start:end], dtype=torch.float32).to(device)
        with torch.no_grad():
            mean, _ = model.encoder(batch)
        embeddings.append(mean.cpu().numpy())
        if (start // VAE_BATCH) % 20 == 0:
            print(f"  [{end}/{n}] inference...")
    X = np.vstack(embeddings).astype("float32")
    print(f"Embeddings: {X.shape}")
    return X

def decode_latents(z, batch_size=64):
    from vae import VAELightningModule
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAELightningModule.load_from_checkpoint(CKPT_PATH)
    model.to(device).eval()
    outs = []
    z = np.asarray(z, dtype="float32")
    for start in range(0, len(z), batch_size):
        with torch.no_grad():
            z_t = torch.tensor(z[start:start + batch_size], dtype=torch.float32).to(device)
            out = model.decoder(z_t).cpu().numpy()
        outs.append(out)
    decoded = np.concatenate(outs, axis=0)
    decoded = np.clip((decoded + 1) / 2, 0, 1).transpose(0, 2, 3, 1)
    return decoded

def tile_to_rgb(tile):
    arr = np.asarray(tile, dtype="float32")
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = arr.transpose(1,2,0)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.empty_like(arr)
    for c in range(arr.shape[2]):
        ch = arr[..., c]
        lo, hi = np.percentile(ch, [1, 99])
        if hi <= lo + 1e-8:
            out[..., c] = 0
        else:
            out[..., c] = np.clip((ch - lo) / (hi - lo), 0, 1)
    return out

def run_ccf_pipeline(X_vae, ir_mean, C):
    # Step 0
    ir_reg = LinearRegression().fit(ir_mean.reshape(-1,1), X_vae)
    X_ir_free = X_vae - ir_reg.predict(ir_mean.reshape(-1,1))
    # Step 1
    scaler_C = StandardScaler().fit(C)
    C_scaled = scaler_C.transform(C)
    env_reg = LinearRegression().fit(C_scaled, X_ir_free)
    X_env_explained = env_reg.predict(C_scaled)
    X_residual = X_ir_free - X_env_explained
    # Steps 2-3
    n_comp = min(2, X_env_explained.shape[1] - 1, X_env_explained.shape[0] // 10)
    pca_env = PCA(n_components=n_comp, random_state=42).fit(X_env_explained)
    env_pcs = pca_env.transform(X_env_explained)
    pca_res = PCA(n_components=n_comp, random_state=42).fit(X_residual)
    res_pcs = pca_res.transform(X_residual)
    # Variance
    var_total = np.var(X_vae, axis=0).sum()
    var_ir_free = np.var(X_ir_free, axis=0).sum()
    var_env = np.var(X_env_explained, axis=0).sum()
    var_res = np.var(X_residual, axis=0).sum()
    print(f"Var total={var_total:.1f}")
    print(f"After IR: {var_ir_free:.1f} ({var_ir_free/var_total:.1%})")
    print(f"Env expl: {var_env:.1f} ({var_env/var_ir_free:.1%})")
    print(f"Residual: {var_res:.1f} ({var_res/var_ir_free:.1%})")
    return dict(
        X_ir_free=X_ir_free, X_env_explained=X_env_explained, X_residual=X_residual,
        env_pcs=env_pcs, res_pcs=res_pcs, pca_env=pca_env, pca_res=pca_res,
        C_scaler=scaler_C, env_regression_coef=env_reg.coef_,
        var_total=var_total, var_ir_free=var_ir_free, var_env=var_env, var_res=var_res,
    )

def build_atlas(X_vae, env_pc1, res_pc1, n_cols=6, n_rows=5):
    coords = np.column_stack([env_pc1, res_pc1])
    tree = cKDTree(coords)
    env_edges = np.percentile(env_pc1, np.linspace(0,100,n_cols+1))
    res_edges = np.percentile(res_pc1, np.linspace(0,100,n_rows+1))
    env_centers = np.percentile(env_pc1, np.linspace(50/n_cols,100-50/n_cols,n_cols))
    res_centers = np.percentile(res_pc1, np.linspace(50/n_rows,100-50/n_rows,n_rows))
    chosen_idxs = np.zeros((n_rows, n_cols), dtype=int)
    cell_medians = np.zeros((n_rows, n_cols, 2))
    cell_counts = np.zeros((n_rows, n_cols), dtype=int)
    for r in range(n_rows):
        res_lo, res_hi = res_edges[r], res_edges[r+1]
        for c in range(n_cols):
            env_lo, env_hi = env_edges[c], env_edges[c+1]
            mask = (env_pc1>=env_lo)&(env_pc1<env_hi if c<n_cols-1 else env_pc1<=env_hi)& \
                   (res_pc1>=res_lo)&(res_pc1<res_hi if r<n_rows-1 else res_pc1<=res_hi)
            if mask.sum()>0:
                center=np.array([np.median(env_pc1[mask]), np.median(res_pc1[mask])])
                dist,idx=tree.query(center.reshape(1,-1),k=1)
                chosen_idxs[r,c]=idx[0]; cell_medians[r,c]=center; cell_counts[r,c]=mask.sum()
            else:
                center=np.array([env_centers[c], res_centers[r]])
                dist,idx=tree.query(center.reshape(1,-1),k=1)
                chosen_idxs[r,c]=idx[0]; cell_medians[r,c]=center; cell_counts[r,c]=1
    z = X_vae[chosen_idxs.ravel()].astype("float32")
    decoded = decode_latents(z).reshape(n_rows, n_cols, 128, 128, 3)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2.2, n_rows*2.3), facecolor="#111111")
    axes = np.asarray(axes).reshape(n_rows, n_cols)
    for r in range(n_rows):
        for c in range(n_cols):
            ax=axes[r,c]; ax.imshow(tile_to_rgb(decoded[r,c]))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_edgecolor("#444444")
            if r==0: ax.set_title(f"envPC1={cell_medians[r,c,0]:.2f}\nn={cell_counts[r,c]}",color="white",fontsize=7,pad=4)
            if c==0: ax.set_ylabel(f"resPC1\n{cell_medians[r,c,1]:.2f}",color="white",fontsize=8,fontweight="bold",rotation=0,labelpad=32,va="center")
    fig.suptitle("VAE‑decoded morphology atlas: envPC1 × resPC1",color="white",fontsize=10,fontweight="bold")
    plt.tight_layout(rect=[0,0.02,1,0.91])
    out_path=os.path.join(OUT_DIR,"morphology_atlas.png")
    plt.savefig(out_path,dpi=150,bbox_inches="tight",facecolor="#111111")
    plt.close()
    print(f"Atlas saved -> {out_path}")

def run_gpu_pipeline():
    print("=== GPU pipeline ===")
    os.makedirs(OUT_DIR, exist_ok=True)
    prep = np.load(os.path.join(PREP_DIR, "preprocessed.npz"), allow_pickle=True)
    tiles, C, ir_mean = prep["tiles"], prep["C"], prep["ir_mean"]
    print(f"Loaded {len(tiles)} tiles")
    X_vae = run_vae_inference(tiles)
    pipe = run_ccf_pipeline(X_vae, ir_mean, C)
    np.savez_compressed(os.path.join(OUT_DIR,"pipeline_results.npz"),
                        X_ir_free=pipe["X_ir_free"],
                        X_env_explained=pipe["X_env_explained"],
                        X_residual=pipe["X_residual"],
                        env_pcs=pipe["env_pcs"],
                        res_pcs=pipe["res_pcs"],
                        var_total=pipe["var_total"],
                        var_ir_free=pipe["var_ir_free"],
                        var_env=pipe["var_env"],
                        var_res=pipe["var_res"],
                        env_reg_coef=pipe["env_regression_coef"],
                        CCF_vars=np.array(["TS","EIS","TS_adv","RH700","w700","WS"]))
    build_atlas(X_vae, pipe["env_pcs"][:,0], pipe["res_pcs"][:,0])

# ── Entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    if PREP_ONLY:
        run_prep()
    else:
        run_gpu_pipeline()