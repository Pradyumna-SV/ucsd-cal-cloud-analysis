#!/usr/bin/env python3
"""
Unified CCF pipeline for NE-Pacific MODIS tiles.

Environment variable PREP_ONLY:
  PREP_ONLY=1  -> CPU-only preprocessing: extract MERRA-2 CCF anomalies + IR means,
                   save to /workspace/ccf_preprocessed/preprocessed.npz, then exit.
  otherwise    -> GPU pipeline: load preprocessed data, run VAE inference,
                   perform IR-removal + CCF regression + PCA, build morphology
                   atlases (default: |EIS| x residual morphology; optional PC
                   comparison via ATLAS_MAKE_PC), log to Weights & Biases.
"""

import os
import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _env_flag(name, default="0"):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


# ── Config ───────────────────────────────────────────────────────────────────
PREP_ONLY = _env_flag("PREP_ONLY")

TILES_PATH = os.environ.get("TILES_PATH", "/workspace/nepac_scratch/nepac_tiles.npy")
META_PATH  = os.environ.get("META_PATH",  "/workspace/nepac_scratch/nepac_meta.npz")
CCF_PATH   = os.environ.get("CCF_PATH",   "/workspace/ccf_2011_fixed.nc")
CHECKPOINT = os.environ.get("CHECKPOINT", "/workspace/vae_checkpoint/lightning_model_50_transform.pt")
OUT_DIR    = os.environ.get("OUT_DIR",    "/workspace/results/ccf_pipeline")
PREP_DIR   = os.environ.get("PREP_DIR",   "/workspace/ccf_preprocessed")

VAE_BATCH    = int(os.environ.get("VAE_BATCH",    64))
ATLAS_N_COLS = int(os.environ.get("ATLAS_N_COLS", 6))
ATLAS_N_ROWS = int(os.environ.get("ATLAS_N_ROWS", 5))
CCF_VARS     = tuple(os.environ.get("CCF_VARS", "TS,EIS,TS_adv,RH700,w700,WS").split(","))

# Atlas ordering: physical CCF magnitude by default (PI request).
# ATLAS_COL_VAR / ATLAS_ROW_VAR: CCF var name (TS=SST), envPC1, resPC1, or morph_resid.
# morph_resid = dominant latent morphology after regressing out the column score.
ATLAS_COL_VAR  = os.environ.get("ATLAS_COL_VAR",  "EIS").strip()
ATLAS_ROW_VAR  = os.environ.get("ATLAS_ROW_VAR",  "morph_resid").strip()
ATLAS_COL_ABS  = _env_flag("ATLAS_COL_ABS", "1")
ATLAS_ROW_ABS  = _env_flag("ATLAS_ROW_ABS", "0")
ATLAS_MAKE_PC  = _env_flag("ATLAS_MAKE_PC",  "1")   # also emit envPC1 x resPC1 comparison
ATLAS_MAKE_TS  = _env_flag("ATLAS_MAKE_TS",  "0")   # also emit |TS| x morph_resid atlas

# WandB will be initialised inside each mode if WANDB_PROJECT is set
WANDB_PROJECT  = os.environ.get("WANDB_PROJECT", None)
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME", None)
WANDB_MODE     = os.environ.get("WANDB_MODE", "online")

# GPU imports only if needed
if not PREP_ONLY:
    import torch
    from sklearn.linear_model import LinearRegression
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from scipy.spatial import cKDTree
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PREP_DIR, exist_ok=True)


# ── CCF extraction (works for both raw and anomaly files) ─────────────────────
def extract_ccf_matrix(lat_arr, lon_arr, times_arr, ccf_nc_path):
    ds = xr.open_dataset(ccf_nc_path)
    var_names = CCF_VARS
    N = len(lat_arr)
    C = np.full((N, len(var_names)), np.nan, dtype=np.float64)
    if not isinstance(times_arr, pd.DatetimeIndex):
        times_arr = pd.to_datetime(times_arr)
    month_starts = times_arr.strftime("%Y-%m-01")
    for i in range(N):
        try:
            point = ds.sel(time=month_starts[i], lat=lat_arr[i], lon=lon_arr[i], method="nearest")
            for j, vn in enumerate(var_names):
                C[i, j] = float(point[vn].values)
        except Exception:
            continue
    ds.close()
    return C


# ── PREP ONLY: extract CCFs + IR means, save, exit ────────────────────────────
def run_prep():
    print("=== PREP ONLY: extracting CCF anomalies and IR means ===")
    os.makedirs(PREP_DIR, exist_ok=True)

    if WANDB_PROJECT:
        import wandb
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME or "ccf-prep",
            mode=WANDB_MODE,
            config=dict(
                prep_only=True,
                tiles_path=TILES_PATH,
                meta_path=META_PATH,
                ccf_path=CCF_PATH,
                prep_dir=PREP_DIR,
                ccf_vars=CCF_VARS,
            ),
        )

    tiles = np.load(TILES_PATH)
    meta  = np.load(META_PATH, allow_pickle=True)
    lat   = meta["lat"].astype("float32")
    lon   = meta["lon"].astype("float32")
    times = pd.to_datetime(meta["time"])
    n_total = len(lat)
    print(f"Loaded {n_total:,} tiles")

    C = extract_ccf_matrix(lat, lon, times, CCF_PATH)
    valid = np.isfinite(C).all(axis=1)
    print(f"Complete CCFs: {valid.sum():,} / {n_total:,}")

    tiles   = tiles[valid]
    C       = C[valid]
    lat     = lat[valid]
    lon     = lon[valid]
    times   = times[valid]

    ir_mean = tiles[:, 2, :, :].mean(axis=(1, 2))
    print("IR means computed.")

    out_file = os.path.join(PREP_DIR, "preprocessed.npz")
    np.savez_compressed(
        out_file,
        tiles    = tiles,
        C        = C,
        ir_mean  = ir_mean,
        lat      = lat,
        lon      = lon,
        times    = times.to_numpy(),
        ccf_vars = np.array(CCF_VARS),
    )
    print(f"Saved preprocessed data -> {out_file}")

    if WANDB_PROJECT:
        wandb.log({"n_tiles_kept": len(tiles), "n_tiles_total": n_total})
        wandb.finish()


# ── GPU pipeline (after prep) ─────────────────────────────────────────────────
def run_vae_inference(tiles):
    from vae import VAELightningModule
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAELightningModule.load_from_checkpoint(CHECKPOINT)
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
    model = VAELightningModule.load_from_checkpoint(CHECKPOINT)
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
        arr = arr.transpose(1, 2, 0)
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


def _normalize_ccf_var(name):
    name = name.strip().upper()
    if name == "SST":
        name = "TS"
    return name


def _ccf_var_index(name):
    name = _normalize_ccf_var(name)
    if name not in CCF_VARS:
        raise ValueError(f"Unknown CCF var {name!r}; choices: {CCF_VARS}")
    return CCF_VARS.index(name)


def _morph_resid_score(X_vae, col_score):
    """Dominant morphology axis after removing the column environmental score."""
    X_sc = StandardScaler().fit_transform(X_vae)
    col = np.asarray(col_score, dtype="float64")
    col_z = (col - np.nanmean(col)) / (np.nanstd(col) + 1e-6)
    design = np.column_stack([np.ones(len(col_z)), col_z])
    coef, *_ = np.linalg.lstsq(design, X_sc, rcond=None)
    X_resid = X_sc - design @ coef
    return PCA(n_components=1, random_state=42).fit_transform(X_resid).flatten()


def resolve_atlas_axis(spec, use_abs, C, pipe, X_vae, col_score=None):
    """Return (score_array, axis_label) for one atlas dimension."""
    key = spec.strip().lower()
    if key in ("envpc1", "env_pc1"):
        return pipe["env_pcs"][:, 0].astype("float64"), "envPC1"
    if key in ("respc1", "res_pc1"):
        return pipe["res_pcs"][:, 0].astype("float64"), "resPC1"
    if key in ("morph_resid", "morph-resid", "morphology"):
        if col_score is None:
            raise ValueError("morph_resid row axis requires col_score")
        return _morph_resid_score(X_vae, col_score), "morph resid"

    idx = _ccf_var_index(spec)
    var_name = CCF_VARS[idx]
    vals = C[:, idx].astype("float64")
    if use_abs:
        vals = np.abs(vals)
        label = f"|{var_name}|"
    else:
        label = var_name
    return vals, label


def run_ccf_pipeline(X_vae, ir_mean, C):
    ir_reg = LinearRegression().fit(ir_mean.reshape(-1, 1), X_vae)
    X_ir_free = X_vae - ir_reg.predict(ir_mean.reshape(-1, 1))

    scaler_C = StandardScaler().fit(C)
    C_scaled = scaler_C.transform(C)
    env_reg = LinearRegression().fit(C_scaled, X_ir_free)
    X_env_explained = env_reg.predict(C_scaled)
    X_residual = X_ir_free - X_env_explained

    n_comp = min(2, X_env_explained.shape[1] - 1, X_env_explained.shape[0] // 10)
    pca_env = PCA(n_components=n_comp, random_state=42).fit(X_env_explained)
    env_pcs = pca_env.transform(X_env_explained)
    pca_res = PCA(n_components=n_comp, random_state=42).fit(X_residual)
    res_pcs = pca_res.transform(X_residual)

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


def build_atlas(X_vae, tiles, col_score, row_score, col_label, row_label,
                n_cols=6, n_rows=5, file_tag=""):
    col_score = np.asarray(col_score, dtype="float64")
    row_score = np.asarray(row_score, dtype="float64")
    coords = np.column_stack([col_score, row_score])
    tree = cKDTree(coords)
    col_edges = np.percentile(col_score, np.linspace(0, 100, n_cols + 1))
    row_edges = np.percentile(row_score, np.linspace(0, 100, n_rows + 1))
    col_centers = np.percentile(col_score, np.linspace(50 / n_cols, 100 - 50 / n_cols, n_cols))
    row_centers = np.percentile(row_score, np.linspace(50 / n_rows, 100 - 50 / n_rows, n_rows))
    chosen_idxs = np.zeros((n_rows, n_cols), dtype=int)
    cell_medians = np.zeros((n_rows, n_cols, 2))
    cell_counts = np.zeros((n_rows, n_cols), dtype=int)
    for r in range(n_rows):
        row_lo, row_hi = row_edges[r], row_edges[r + 1]
        for c in range(n_cols):
            col_lo, col_hi = col_edges[c], col_edges[c + 1]
            col_mask = (col_score >= col_lo) & ((col_score < col_hi) if c < n_cols - 1 else (col_score <= col_hi))
            row_mask = (row_score >= row_lo) & ((row_score < row_hi) if r < n_rows - 1 else (row_score <= row_hi))
            mask = col_mask & row_mask
            if mask.sum() > 0:
                idxs = np.where(mask)[0]
                center = np.array([np.median(col_score[idxs]), np.median(row_score[idxs])])
                local_dist = np.linalg.norm(coords[idxs] - center, axis=1)
                chosen_idxs[r, c] = int(idxs[np.argmin(local_dist)])
                cell_medians[r, c] = center
                cell_counts[r, c] = len(idxs)
            else:
                center = np.array([col_centers[c], row_centers[r]])
                _, idx = tree.query(center.reshape(1, -1), k=1)
                chosen_idxs[r, c] = int(idx[0])
                cell_medians[r, c] = center
                cell_counts[r, c] = 1
    z = X_vae[chosen_idxs.ravel()].astype("float32")
    decoded = decode_latents(z).reshape(n_rows, n_cols, 128, 128, 3)
    observed = np.array([tile_to_rgb(tiles[i]) for i in chosen_idxs.ravel()]).reshape(n_rows, n_cols, 128, 128, 3)

    suffix = f"_{file_tag}" if file_tag else ""
    axis_title = f"{col_label} x {row_label}"

    def save_grid(imgs, out_name, title, decoded_grid=False):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.3),
                                 facecolor="#111111")
        axes = np.asarray(axes).reshape(n_rows, n_cols)
        for r in range(n_rows):
            for c in range(n_cols):
                ax = axes[r, c]
                ax.imshow(imgs[r, c] if decoded_grid else tile_to_rgb(imgs[r, c]))
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_edgecolor("#444444")
                if r == 0:
                    ax.set_title(
                        f"{col_label}={cell_medians[r, c, 0]:.2f}\nn={cell_counts[r, c]}",
                        color="white", fontsize=7, pad=4)
                if c == 0:
                    ax.set_ylabel(
                        f"{row_label}\n{cell_medians[r, c, 1]:.2f}",
                        color="white", fontsize=8, fontweight="bold",
                        rotation=0, labelpad=32, va="center")
        fig.suptitle(title, color="white", fontsize=10, fontweight="bold")
        plt.tight_layout(rect=[0, 0.02, 1, 0.91])
        out_path = os.path.join(OUT_DIR, out_name)
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#111111")
        plt.close()
        print(f"Saved atlas -> {out_path}")
        return out_path

    decoded_path = save_grid(
        decoded,
        f"morphology_atlas{suffix}_decoded.png",
        f"VAE-decoded morphology atlas: {axis_title}",
        decoded_grid=True,
    )
    observed_path = save_grid(
        observed,
        f"morphology_atlas{suffix}_observed.png",
        f"Observed MODIS medoids: {axis_title}",
        decoded_grid=True,
    )
    return dict(
        decoded_path=decoded_path,
        observed_path=observed_path,
        col_label=col_label,
        row_label=row_label,
        file_tag=file_tag or "default",
        cell_counts=cell_counts,
    )


def build_atlas_from_specs(X_vae, tiles, C, pipe, col_spec, row_spec,
                           col_abs, row_abs, file_tag, n_cols, n_rows):
    col_score, col_label = resolve_atlas_axis(col_spec, col_abs, C, pipe, X_vae)
    row_score, row_label = resolve_atlas_axis(
        row_spec, row_abs, C, pipe, X_vae, col_score=col_score,
    )
    print(f"  Atlas {file_tag or 'default'}: columns={col_label}, rows={row_label}")
    return build_atlas(
        X_vae, tiles, col_score, row_score, col_label, row_label,
        n_cols=n_cols, n_rows=n_rows, file_tag=file_tag,
    )

def run_gpu_pipeline():
    print("=== GPU pipeline ===")
    os.makedirs(OUT_DIR, exist_ok=True)

    if WANDB_PROJECT:
        import wandb
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME or "ccf-gpu",
            mode=WANDB_MODE,
            config=dict(
                prep_only=False,
                prep_dir=PREP_DIR,
                out_dir=OUT_DIR,
                checkpoint=os.path.basename(CHECKPOINT),
                vae_batch=VAE_BATCH,
                atlas_n_cols=ATLAS_N_COLS,
                atlas_n_rows=ATLAS_N_ROWS,
                ccf_vars=CCF_VARS,
                atlas_col_var=ATLAS_COL_VAR,
                atlas_row_var=ATLAS_ROW_VAR,
                atlas_col_abs=ATLAS_COL_ABS,
                atlas_row_abs=ATLAS_ROW_ABS,
                atlas_make_pc=ATLAS_MAKE_PC,
                atlas_make_ts=ATLAS_MAKE_TS,
            ),
        )

    prep = np.load(os.path.join(PREP_DIR, "preprocessed.npz"), allow_pickle=True)
    tiles, C, ir_mean = prep["tiles"], prep["C"], prep["ir_mean"]
    print(f"Loaded {len(tiles)} tiles")

    X_vae = run_vae_inference(tiles)
    pipe = run_ccf_pipeline(X_vae, ir_mean, C)

    # Save numerical results
    results_path = os.path.join(OUT_DIR, "pipeline_results.npz")
    np.savez_compressed(results_path,
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
                        CCF_vars=np.array(CCF_VARS))

    eis_idx = _ccf_var_index("EIS")
    ts_idx = _ccf_var_index("TS")
    corr_envpc1_abs_eis = float(np.corrcoef(
        pipe["env_pcs"][:, 0], np.abs(C[:, eis_idx]),
    )[0, 1])
    corr_envpc1_abs_ts = float(np.corrcoef(
        pipe["env_pcs"][:, 0], np.abs(C[:, ts_idx]),
    )[0, 1])
    print(f"corr(envPC1, |EIS|) = {corr_envpc1_abs_eis:.3f}")
    print(f"corr(envPC1, |TS|)  = {corr_envpc1_abs_ts:.3f}")

    atlases = []
    print("Building primary morphology atlas (physical ordering)...")
    atlases.append(build_atlas_from_specs(
        X_vae, tiles, C, pipe,
        col_spec=ATLAS_COL_VAR, row_spec=ATLAS_ROW_VAR,
        col_abs=ATLAS_COL_ABS, row_abs=ATLAS_ROW_ABS,
        file_tag="", n_cols=ATLAS_N_COLS, n_rows=ATLAS_N_ROWS,
    ))
    if ATLAS_MAKE_PC:
        print("Building PC comparison atlas (envPC1 x resPC1)...")
        atlases.append(build_atlas_from_specs(
            X_vae, tiles, C, pipe,
            col_spec="envPC1", row_spec="resPC1",
            col_abs=False, row_abs=False,
            file_tag="pc", n_cols=ATLAS_N_COLS, n_rows=ATLAS_N_ROWS,
        ))
    if ATLAS_MAKE_TS:
        print("Building |TS| morphology atlas...")
        atlases.append(build_atlas_from_specs(
            X_vae, tiles, C, pipe,
            col_spec="TS", row_spec="morph_resid",
            col_abs=True, row_abs=False,
            file_tag="abs_ts", n_cols=ATLAS_N_COLS, n_rows=ATLAS_N_ROWS,
        ))

    if WANDB_PROJECT:
        artifact = wandb.Artifact("ccf_pipeline_results", type="dataset", metadata={
            "description": "CCF pipeline PCA scores and variance fractions"
        })
        artifact.add_file(results_path)
        log_payload = {
            "n_tiles": len(tiles),
            "var_total": pipe["var_total"],
            "var_ir_free": pipe["var_ir_free"],
            "var_env": pipe["var_env"],
            "var_res": pipe["var_res"],
            "var_env_fraction": pipe["var_env"] / pipe["var_ir_free"],
            "var_res_fraction": pipe["var_res"] / pipe["var_ir_free"],
            "corr_envPC1_abs_EIS": corr_envpc1_abs_eis,
            "corr_envPC1_abs_TS": corr_envpc1_abs_ts,
        }
        for atlas in atlases:
            tag = atlas["file_tag"]
            log_payload[f"ccf_morphology_atlas_{tag}_decoded"] = wandb.Image(atlas["decoded_path"])
            log_payload[f"ccf_morphology_atlas_{tag}_observed"] = wandb.Image(atlas["observed_path"])
        wandb.log(log_payload)
        wandb.log_artifact(artifact)
        wandb.finish()

    print("Done.")

# ── Entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    if PREP_ONLY:
        run_prep()
    else:
        run_gpu_pipeline()