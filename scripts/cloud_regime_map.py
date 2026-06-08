#!/usr/bin/env python3
"""
Global cloud-regime map from Tile2Vec embeddings.

Uses the same PVC layout and manifest day queue as scripts/cca_20yr.py.

White-background maps use geographic grid aggregation (mean RGB per lat/lon
cell at full opacity). Scatter + alpha cannot reproduce the vivid black-bg
look on white — overlapping transparent points wash out.

Environment variables (all optional):
  EMBED_DIR         embeddings root on PVC       default: /workspace/embeddings
  MANIFEST          manifest.csv path            default: /workspace/repo/manifest.csv
  OUT_DIR           output directory             default: /workspace/results/cloud_regime_map
  STREAM_STRIDE     every Nth OK manifest day    default: 1
  SUBSAMPLE_RATE    fraction per day             default: 0.05
  MAX_PER_DAY       cap tiles/day; 0 = no cap    default: 5000
  RANDOM_SEED       subsample seed               default: 42
  OUT_NAME          output png filename          default: cloud_regime_map_20years.png
  PLOT_BACKGROUND   white or black               default: white
  PLOT_MODE         geo_grid or scatter          default: geo_grid if white else scatter
  GRID_N_LON        longitude bins for geo_grid  default: 3600
  GRID_N_LAT        latitude bins for geo_grid   default: 1800
  GRID_MIN_COUNT    min points to paint a cell   default: 1
  COLOR_NORM        global or percentile         default: global
  COLOR_GAMMA       <1 deepens colors on white   default: 0.8
  PLOT_MARKER_SIZE  scatter only                 default: 0.05
  PLOT_ALPHA        scatter only                 default: 0.8
  WANDB_PROJECT     W&B project                  default: unset
  WANDB_RUN_NAME    W&B run name                 default: cloud-regime-map
  WANDB_MODE        online/offline/disabled      default: online
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from sklearn.decomposition import PCA

EMBED_DIR = os.environ.get("EMBED_DIR", os.environ.get("BASE_DIR", "/workspace/embeddings"))
MANIFEST = os.environ.get("MANIFEST", "/workspace/repo/manifest.csv")
OUT_DIR = os.environ.get("OUT_DIR", "/workspace/results/cloud_regime_map")
STREAM_STRIDE = max(1, int(os.environ.get("STREAM_STRIDE", os.environ.get("DAY_STRIDE", "1"))))
SUBSAMPLE_RATE = float(os.environ.get("SUBSAMPLE_RATE", "0.05"))
_MAX_PER_DAY = os.environ.get("MAX_PER_DAY", "5000").strip()
MAX_PER_DAY = int(_MAX_PER_DAY) if _MAX_PER_DAY and _MAX_PER_DAY != "0" else None
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))
OUT_NAME = os.environ.get("OUT_NAME", "cloud_regime_map_20years.png")
PLOT_BACKGROUND = os.environ.get("PLOT_BACKGROUND", "white").strip().lower()
_COLOR_GAMMA = os.environ.get("COLOR_GAMMA", "0.8").strip()
COLOR_GAMMA = float(_COLOR_GAMMA) if _COLOR_GAMMA else 1.0
COLOR_NORM = os.environ.get("COLOR_NORM", "global").strip().lower()
GRID_N_LON = max(100, int(os.environ.get("GRID_N_LON", "3600")))
GRID_N_LAT = max(50, int(os.environ.get("GRID_N_LAT", "1800")))
GRID_MIN_COUNT = max(1, int(os.environ.get("GRID_MIN_COUNT", "1")))
PLOT_MARKER_SIZE = float(os.environ.get("PLOT_MARKER_SIZE", "0.05"))
PLOT_ALPHA = float(os.environ.get("PLOT_ALPHA", "0.8"))
_PLOT_MODE = os.environ.get("PLOT_MODE", "").strip().lower()
if _PLOT_MODE:
    PLOT_MODE = _PLOT_MODE
elif PLOT_BACKGROUND == "white":
    PLOT_MODE = "geo_grid"
else:
    PLOT_MODE = "scatter"
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", None)
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME", None)
WANDB_MODE = os.environ.get("WANDB_MODE", "online")


def check_memory(label=""):
    mem = psutil.virtual_memory()
    prefix = f"[RAM{': ' + label if label else ''}]"
    print(f"{prefix} Used: {mem.used / (1024**3):.2f} GiB / "
          f"Total: {mem.total / (1024**3):.2f} GiB ({mem.percent}%)")


def load_day_tile2vec(year, month, day):
    """Load one day's Tile2Vec embeddings + lat/lon (matches cca_20yr.py)."""
    prefix = f"{year}_{month:02d}_{day:02d}"
    day_dir = Path(EMBED_DIR) / str(year) / f"{month:02d}" / f"{day:02d}"
    t2v_p = day_dir / f"{prefix}_tile2vec.npy"
    meta_p = day_dir / f"{prefix}_centers.json"

    if not (t2v_p.exists() and meta_p.exists()):
        return None

    try:
        t2v_arr = np.load(t2v_p).squeeze()
    except Exception as exc:
        print(f"  [skip] {prefix}: corrupt tile2vec ({exc})")
        return None

    try:
        with open(meta_p, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as exc:
        print(f"  [skip] {prefix}: corrupt centers json ({exc})")
        return None

    lats, lons = [], []
    for ts in sorted(meta.keys()):
        for lat, lon in meta[ts]:
            lats.append(float(lat))
            lons.append(float(lon))

    if not lats:
        print(f"  [skip] {prefix}: empty centers json")
        return None

    n = min(len(lats), len(t2v_arr))
    if n < len(lats):
        print(f"  [warn] {prefix}: tile2vec shorter than centers ({n} vs {len(lats)}), truncating")

    return {
        "prefix": prefix,
        "t2v": np.asarray(t2v_arr[:n], dtype=np.float32),
        "lat": np.asarray(lats[:n], dtype=np.float64),
        "lon": np.asarray(lons[:n], dtype=np.float64),
    }


def pca_scores_to_rgb(rgb_pca):
    """Map PCA scores to RGB. 'global' matches the original notebook script."""
    if COLOR_NORM == "percentile":
        colors = np.empty_like(rgb_pca, dtype=np.float64)
        for j in range(rgb_pca.shape[1]):
            lo, hi = np.percentile(rgb_pca[:, j], [2, 98])
            colors[:, j] = (rgb_pca[:, j] - lo) / max(hi - lo, 1e-8)
        colors = np.clip(colors, 0.0, 1.0)
    else:
        rgb_min = rgb_pca.min(axis=0)
        rgb_max = rgb_pca.max(axis=0)
        colors = (rgb_pca - rgb_min) / np.maximum(rgb_max - rgb_min, 1e-8)
    return colors.astype(np.float32)


def build_geo_rgb_image(lons, lats, colors):
    """
    Average RGB into a lat/lon grid at full opacity.
    Empty cells stay white — this is what makes white-bg maps vivid.
    """
    lon_idx = np.floor((lons + 180.0) / 360.0 * GRID_N_LON).astype(np.int32)
    lat_idx = np.floor((lats + 90.0) / 180.0 * GRID_N_LAT).astype(np.int32)
    lon_idx = np.clip(lon_idx, 0, GRID_N_LON - 1)
    lat_idx = np.clip(lat_idx, 0, GRID_N_LAT - 1)

    sum_r = np.zeros((GRID_N_LAT, GRID_N_LON), dtype=np.float64)
    sum_g = np.zeros((GRID_N_LAT, GRID_N_LON), dtype=np.float64)
    sum_b = np.zeros((GRID_N_LAT, GRID_N_LON), dtype=np.float64)
    counts = np.zeros((GRID_N_LAT, GRID_N_LON), dtype=np.int32)

    np.add.at(sum_r, (lat_idx, lon_idx), colors[:, 0].astype(np.float64))
    np.add.at(sum_g, (lat_idx, lon_idx), colors[:, 1].astype(np.float64))
    np.add.at(sum_b, (lat_idx, lon_idx), colors[:, 2].astype(np.float64))
    np.add.at(counts, (lat_idx, lon_idx), 1)

    mask = counts >= GRID_MIN_COUNT
    img = np.ones((GRID_N_LAT, GRID_N_LON, 3), dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        img[..., 0] = np.where(mask, sum_r / np.maximum(counts, 1), 1.0)
        img[..., 1] = np.where(mask, sum_g / np.maximum(counts, 1), 1.0)
        img[..., 2] = np.where(mask, sum_b / np.maximum(counts, 1), 1.0)

    if COLOR_GAMMA > 0 and COLOR_GAMMA != 1.0:
        data_mask = np.broadcast_to(mask[..., None], img.shape)
        img[data_mask] = np.power(np.clip(img[data_mask], 0.0, 1.0), COLOR_GAMMA)

    img = np.clip(img, 0.0, 1.0)
    return img, int(mask.sum())


def load_manifest_queue():
    if not os.path.exists(MANIFEST):
        raise FileNotFoundError(
            f"Manifest not found at {MANIFEST}. "
            "Set MANIFEST to manifest.csv on the PVC or in the cloned repo."
        )
    manifest = pd.read_csv(MANIFEST)
    queue = (
        manifest[manifest["status"] == "OK"]
        .sort_values(["year", "month", "day"])
        .reset_index(drop=True)
    )
    queue = queue.iloc[::STREAM_STRIDE].reset_index(drop=True)
    return queue


def save_map(lons, lats, colors, pca, n_loaded):
    dark_bg = PLOT_BACKGROUND != "white"
    bg = "black" if dark_bg else "white"
    fg = "white" if dark_bg else "black"

    print(
        f"Plot style: mode={PLOT_MODE}, background={bg}, color_norm={COLOR_NORM}, "
        f"color_gamma={COLOR_GAMMA}"
    )

    fig, ax = plt.subplots(figsize=(20, 10), facecolor=bg)
    ax.set_facecolor(bg)
    grid_cells = None

    if PLOT_MODE == "geo_grid":
        print(f"Aggregating to geo grid {GRID_N_LON} x {GRID_N_LAT}...")
        img, grid_cells = build_geo_rgb_image(lons, lats, colors)
        print(f"Painted {grid_cells:,} grid cells")
        ax.imshow(
            img,
            origin="lower",
            extent=[-180, 180, -90, 90],
            aspect="auto",
            interpolation="nearest",
        )
    else:
        print(f"Scatter: marker_size={PLOT_MARKER_SIZE}, alpha={PLOT_ALPHA}")
        ax.scatter(
            lons, lats, c=colors, s=PLOT_MARKER_SIZE, alpha=PLOT_ALPHA, marker="s",
            linewidths=0,
        )
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)

    ax.set_title(
        "Global Cloud Regimes (2002-2022) - PCA Projected Tile2Vec Embeddings",
        color=fg,
        fontsize=20,
    )
    ax.set_xlabel("Longitude", color=fg)
    ax.set_ylabel("Latitude", color=fg)
    ax.grid(False)
    ax.tick_params(axis="both", colors=fg)

    out_path = os.path.join(OUT_DIR, OUT_NAME)
    fig.savefig(out_path, dpi=300, facecolor=bg, bbox_inches="tight")
    plt.close(fig)
    print(f"Map saved -> {out_path}")
    return out_path, grid_cells


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    if WANDB_PROJECT:
        import wandb
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME or "cloud-regime-map",
            mode=WANDB_MODE,
            config=dict(
                embed_dir=EMBED_DIR,
                manifest=MANIFEST,
                out_dir=OUT_DIR,
                stream_stride=STREAM_STRIDE,
                subsample_rate=SUBSAMPLE_RATE,
                max_per_day=MAX_PER_DAY,
                random_seed=RANDOM_SEED,
                out_name=OUT_NAME,
                plot_background=PLOT_BACKGROUND,
                plot_mode=PLOT_MODE,
                grid_n_lon=GRID_N_LON,
                grid_n_lat=GRID_N_LAT,
                color_norm=COLOR_NORM,
                color_gamma=COLOR_GAMMA,
            ),
        )

    print(f"Embeddings PVC path: {EMBED_DIR}")
    print(f"Manifest: {MANIFEST}")
    queue = load_manifest_queue()
    cap_msg = "no cap" if MAX_PER_DAY is None else str(MAX_PER_DAY)
    print(f"Targeting {len(queue)} OK days (stride={STREAM_STRIDE}, max_per_day={cap_msg})")

    all_vectors = []
    all_lats = []
    all_lons = []
    n_loaded = 0

    check_memory("start")
    for i, row in queue.iterrows():
        if i % 50 == 0:
            print(f"[{i}/{len(queue)}] Loading {int(row['year'])}-"
                  f"{int(row['month']):02d}-{int(row['day']):02d}...")

        day = load_day_tile2vec(int(row["year"]), int(row["month"]), int(row["day"]))
        if day is None:
            continue

        n = len(day["t2v"])
        n_samples = max(1, int(n * SUBSAMPLE_RATE))
        if MAX_PER_DAY is not None:
            n_samples = min(n_samples, MAX_PER_DAY)
        n_samples = min(n_samples, n)
        indices = rng.choice(n, size=n_samples, replace=False)

        all_vectors.append(day["t2v"][indices])
        all_lats.append(day["lat"][indices])
        all_lons.append(day["lon"][indices])
        n_loaded += 1

    if not all_vectors:
        raise RuntimeError(
            f"No embedding days loaded from {EMBED_DIR}. "
            "Check EMBED_DIR and manifest OK rows."
        )

    print(f"Loaded {n_loaded} days. Stacking arrays...")
    check_memory("pre-stack")

    X = np.vstack(all_vectors)
    lats = np.concatenate(all_lats)
    lons = np.concatenate(all_lons)
    print(f"Total points on map: {X.shape[0]:,}  (embedding dim={X.shape[1]})")
    del all_vectors

    print("Running PCA (RGB colors)...")
    pca = PCA(n_components=3, random_state=RANDOM_SEED)
    rgb_pca = pca.fit_transform(X)
    del X
    check_memory("post-pca")
    print(
        "PCA variance explained:",
        " ".join(f"PC{i+1}={v:.1%}" for i, v in enumerate(pca.explained_variance_ratio_)),
    )

    colors = pca_scores_to_rgb(rgb_pca)
    del rgb_pca

    out_path, grid_cells = save_map(lons, lats, colors, pca, n_loaded)

    if WANDB_PROJECT:
        import wandb
        artifact = wandb.Artifact(
            "cloud_regime_map",
            type="image",
            metadata=dict(
                n_days_loaded=n_loaded,
                n_points=int(lats.shape[0]),
                pca_explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
            ),
        )
        artifact.add_file(out_path, name=OUT_NAME)
        log_payload = {
            "n_days_loaded": n_loaded,
            "n_points": int(lats.shape[0]),
            "pca_var_ratio_pc1": float(pca.explained_variance_ratio_[0]),
            "pca_var_ratio_pc2": float(pca.explained_variance_ratio_[1]),
            "pca_var_ratio_pc3": float(pca.explained_variance_ratio_[2]),
            "plot_background": PLOT_BACKGROUND,
            "plot_mode": PLOT_MODE,
            "cloud_regime_map": wandb.Image(out_path),
        }
        if grid_cells is not None:
            log_payload["grid_cells_painted"] = grid_cells
        wandb.log(log_payload)
        wandb.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    main()
