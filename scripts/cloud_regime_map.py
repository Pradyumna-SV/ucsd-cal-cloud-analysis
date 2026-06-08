#!/usr/bin/env python3
"""
Global cloud-regime map from Tile2Vec embeddings.

Uses the same PVC layout and manifest day queue as scripts/cca_20yr.py:
  /workspace/embeddings/YYYY/MM/DD/YYYY_MM_DD_tile2vec.npy
  /workspace/embeddings/YYYY/MM/DD/YYYY_MM_DD_centers.json

Environment variables (all optional):
  EMBED_DIR       embeddings root on PVC     default: /workspace/embeddings
  MANIFEST        manifest.csv path          default: /workspace/repo/manifest.csv
  OUT_DIR         output directory           default: /workspace/results/cloud_regime_map
  STREAM_STRIDE   every Nth OK manifest day  default: 11
  SUBSAMPLE_RATE  fraction per day           default: 0.05
  MAX_PER_DAY     cap tiles kept per day     default: 2000
  RANDOM_SEED     subsample seed             default: 42
  OUT_NAME        output png filename        default: cloud_regime_map_20years.png
  WANDB_PROJECT   W&B project                default: unset (skip logging)
  WANDB_RUN_NAME  W&B run name               default: cloud-regime-map
  WANDB_MODE      online/offline/disabled    default: online
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
STREAM_STRIDE = max(1, int(os.environ.get("STREAM_STRIDE", os.environ.get("DAY_STRIDE", "11"))))
SUBSAMPLE_RATE = float(os.environ.get("SUBSAMPLE_RATE", "0.05"))
MAX_PER_DAY = max(1, int(os.environ.get("MAX_PER_DAY", "2000")))
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))
OUT_NAME = os.environ.get("OUT_NAME", "cloud_regime_map_20years.png")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", None)
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME", None)
WANDB_MODE = os.environ.get("WANDB_MODE", "online")


def check_memory(label=""):
    mem = psutil.virtual_memory()
    prefix = f"[RAM{': ' + label if label else ''}]"
    print(f"{prefix} Used: {mem.used / (1024**3):.2f} GiB / "
          f"Total: {mem.total / (1024**3):.2f} GiB ({mem.percent}%)")


def load_day_tile2vec(year, month, day):
    """
    Load one day's Tile2Vec embeddings + lat/lon.
    Mirrors scripts/cca_20yr.py::load_day (tile2vec path only).
    """
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
            ),
        )

    print(f"Embeddings PVC path: {EMBED_DIR}")
    print(f"Manifest: {MANIFEST}")
    queue = load_manifest_queue()
    print(f"Targeting {len(queue)} OK days (stride={STREAM_STRIDE})")

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
        n_samples = min(n_samples, n, MAX_PER_DAY)
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

    rgb_min = rgb_pca.min(axis=0)
    rgb_max = rgb_pca.max(axis=0)
    denom = np.maximum(rgb_max - rgb_min, 1e-8)
    colors = (rgb_pca - rgb_min) / denom
    del rgb_pca

    print("Plotting global cloud regimes...")
    fig, ax = plt.subplots(figsize=(20, 10), facecolor="black")
    ax.set_facecolor("black")
    ax.scatter(
        lons, lats, c=colors, s=0.05, alpha=0.8, marker="s",
        linewidths=0, rasterized=True,
    )

    ax.set_title(
        "Global Cloud Regimes (2002-2022) - PCA Projected Tile2Vec Embeddings",
        color="white",
        fontsize=20,
    )
    ax.set_xlabel("Longitude", color="white")
    ax.set_ylabel("Latitude", color="white")
    ax.grid(False)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.tick_params(axis="both", colors="white")

    out_path = os.path.join(OUT_DIR, OUT_NAME)
    fig.savefig(out_path, dpi=300, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    print(f"Map saved -> {out_path}")

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
        wandb.log({
            "n_days_loaded": n_loaded,
            "n_points": int(lats.shape[0]),
            "pca_var_ratio_pc1": float(pca.explained_variance_ratio_[0]),
            "pca_var_ratio_pc2": float(pca.explained_variance_ratio_[1]),
            "pca_var_ratio_pc3": float(pca.explained_variance_ratio_[2]),
            "cloud_regime_map": wandb.Image(out_path),
        })
        wandb.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    main()
