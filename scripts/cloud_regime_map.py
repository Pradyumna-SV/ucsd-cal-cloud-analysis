#!/usr/bin/env python3
"""
Global cloud-regime map from Tile2Vec embeddings.

Reads daily tile2vec.npy + centers.json files, subsamples, runs PCA(3) for RGB
colors, and saves a lat/lon scatter map.

Environment variables (all optional):
  BASE_DIR        embeddings root          default: /workspace/embeddings
  OUT_DIR         output directory         default: /workspace/results/cloud_regime_map
  START_DATE      YYYY-MM-DD               default: 2002-01-01
  END_DATE        YYYY-MM-DD               default: 2022-12-31
  SUBSAMPLE_RATE  fraction per day         default: 0.05
  DAY_STRIDE      process every Nth day      default: 11
  RANDOM_SEED     subsample seed           default: 42
  OUT_NAME        output png filename      default: cloud_regime_map_20years.png
  WANDB_PROJECT   W&B project name         default: unset (skip logging)
  WANDB_RUN_NAME  W&B run name             default: cloud-regime-map
  WANDB_MODE      online/offline/disabled  default: online
"""

import datetime
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil
from sklearn.decomposition import PCA

BASE_DIR = os.environ.get("BASE_DIR", "/workspace/embeddings")
OUT_DIR = os.environ.get("OUT_DIR", "/workspace/results/cloud_regime_map")
START_DATE = datetime.date.fromisoformat(os.environ.get("START_DATE", "2002-01-01"))
END_DATE = datetime.date.fromisoformat(os.environ.get("END_DATE", "2022-12-31"))
SUBSAMPLE_RATE = float(os.environ.get("SUBSAMPLE_RATE", "0.05"))
DAY_STRIDE = max(1, int(os.environ.get("DAY_STRIDE", "11")))
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


def load_day_coords(centers_dict):
    """Flatten centers.json in sorted timestamp order (matches cca_20yr.py)."""
    coords = []
    for ts in sorted(centers_dict.keys()):
        for lat, lon in centers_dict[ts]:
            coords.append((float(lat), float(lon)))
    return np.asarray(coords, dtype=np.float64)


def iter_day_paths():
    current = START_DATE
    day_index = 0
    while current <= END_DATE:
        if day_index % DAY_STRIDE == 0:
            year_str = current.strftime("%Y")
            month_str = current.strftime("%m")
            day_str = current.strftime("%d")
            filename_date = current.strftime("%Y_%m_%d")
            day_dir = os.path.join(BASE_DIR, year_str, month_str, day_str)
            yield (
                os.path.join(day_dir, f"{filename_date}_tile2vec.npy"),
                os.path.join(day_dir, f"{filename_date}_centers.json"),
                filename_date,
            )
        current += datetime.timedelta(days=1)
        day_index += 1


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
                base_dir=BASE_DIR,
                out_dir=OUT_DIR,
                start_date=str(START_DATE),
                end_date=str(END_DATE),
                subsample_rate=SUBSAMPLE_RATE,
                day_stride=DAY_STRIDE,
                random_seed=RANDOM_SEED,
                out_name=OUT_NAME,
            ),
        )

    paths = list(iter_day_paths())
    print(f"Configured {len(paths)} days (stride={DAY_STRIDE}, "
          f"subsample={SUBSAMPLE_RATE:.3f}) under {BASE_DIR}")

    all_vectors = []
    all_lats = []
    all_lons = []
    n_loaded = 0

    check_memory("start")
    for i, (t_file, c_file, day_str) in enumerate(paths):
        if i % 25 == 0:
            print(f"[{i}/{len(paths)}] Processing {day_str}...")

        if not os.path.exists(t_file) or not os.path.exists(c_file):
            continue

        try:
            vecs = np.load(t_file)
            with open(c_file, "r", encoding="utf-8") as f:
                centers_dict = json.load(f)

            day_coords = load_day_coords(centers_dict)
            n = min(len(vecs), len(day_coords))
            if n == 0:
                continue
            vecs = vecs[:n]
            day_coords = day_coords[:n]

            n_samples = max(1, int(n * SUBSAMPLE_RATE))
            n_samples = min(n_samples, n)
            indices = rng.choice(n, size=n_samples, replace=False)

            all_vectors.append(np.asarray(vecs[indices], dtype=np.float32))
            all_lats.append(day_coords[indices, 0])
            all_lons.append(day_coords[indices, 1])
            n_loaded += 1
        except Exception as exc:
            print(f"Error on {day_str}: {exc}")

    if not all_vectors:
        raise RuntimeError(
            f"No embedding days loaded from {BASE_DIR}. "
            "Check BASE_DIR and that tile2vec/centers files exist."
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
    fig, ax = plt.subplots(figsize=(20, 10), facecolor="white")
    ax.set_facecolor("white")
    ax.scatter(lons, lats, c=colors, s=0.05, alpha=0.8, marker="s", linewidths=0)

    ax.set_title(
        "Global Cloud Regimes (2002-2022) - PCA Projected Embeddings",
        color="black",
        fontsize=20,
    )
    ax.set_xlabel("Longitude", color="black")
    ax.set_ylabel("Latitude", color="black")
    ax.grid(False)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.tick_params(axis="both", colors="black")

    out_path = os.path.join(OUT_DIR, OUT_NAME)
    fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
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
