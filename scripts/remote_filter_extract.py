#!/usr/bin/env python3
"""
Runs on the REMOTE DATA SERVER (uploaded and executed via SSH by cca_nepac_eis.py).

Filters MODIS triplet tiles by:
  1. NE Pacific bounding box
  2. Ocean (global_land_mask)
  3. Daytime (local solar hour 06:00–18:00)
  4. Cloud fraction >= CF_THRESH (from MYD06 NetCDF)

Extracts the anchor image (channel 0) for each surviving tile,
saves tiles + metadata to OUTPUT_DIR, and prints the output paths
to stdout so the caller can SCP them back.
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime, timedelta
from itertools import groupby

import netCDF4 as nc
import numpy as np
from global_land_mask import globe
from scipy.spatial import cKDTree


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--coord_dir",  required=True, help="Path to coordinates_data_*.json files")
    p.add_argument("--myd06_dir",  required=True, help="Path to MYD06_L2 NetCDF files")
    p.add_argument("--img_dir",    required=True, help="Path to orig_memmap*.memmap files")
    p.add_argument("--output_dir", required=True, help="Where to write nepac_tiles.npy / nepac_meta.npz")
    p.add_argument("--n_files",    type=int,   default=100)
    p.add_argument("--n_per_file", type=int,   default=10_000)
    p.add_argument("--lat_min",    type=float, default=20.0)
    p.add_argument("--lat_max",    type=float, default=65.0)
    p.add_argument("--lon_min",    type=float, default=-180.0)
    p.add_argument("--lon_max",    type=float, default=-110.0)
    p.add_argument("--cf_thresh",  type=float, default=0.4)
    return p.parse_args()


def parse_modis_filename(nc_file):
    parts = os.path.basename(nc_file).split(".")
    year   = int(parts[1][1:5])
    doy    = int(parts[1][5:])
    hour   = int(parts[2][:2])
    minute = int(parts[2][2:])
    return datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hour, minutes=minute)


def is_daytime(lat, lon, utc_dt):
    local_hour = (utc_dt.hour + utc_dt.minute / 60.0 + lon / 15.0) % 24
    return 6.0 <= local_hour <= 18.0


def find_myd06(nc_file, myd06_dir):
    parts   = os.path.basename(nc_file).split(".")
    year    = parts[1][1:5]
    doy     = parts[1][5:]
    time    = parts[2]
    pattern = os.path.join(myd06_dir, f"MYD06_L2.A{year}{doy}.{time}.*.nc")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


_myd06_cache = {}


def get_cloud_fraction(lat, lon, nc_file, myd06_dir):
    myd06_path = find_myd06(nc_file, myd06_dir)
    if myd06_path is None:
        return None

    if myd06_path not in _myd06_cache:
        ds       = nc.Dataset(myd06_path, "r")
        cf_lats  = ds.variables["latitude"][:].ravel()
        cf_lons  = ds.variables["longitude"][:].ravel()
        cf_vals  = np.ma.filled(ds.variables["Cloud_Fraction"][:], np.nan).ravel()
        ds.close()

        valid = np.isfinite(cf_vals)
        tree  = cKDTree(np.column_stack([cf_lats[valid], cf_lons[valid]]))
        _myd06_cache[myd06_path] = (tree, cf_vals[valid])

        if len(_myd06_cache) > 20:
            del _myd06_cache[next(iter(_myd06_cache))]

    tree, cf_values = _myd06_cache[myd06_path]
    _, idx = tree.query([lat, lon], k=1)
    return float(cf_values[idx])


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    kept    = []   # (file_idx, local_idx, lat, lon, datetime)
    skipped = dict(bbox=0, land=0, night=0, cloud=0, no_myd06=0)

    # ── Pass 1: metadata scan ─────────────────────────────────────────────────
    print("Pass 1: filtering metadata...", flush=True)
    for file_idx in range(args.n_files):
        coord_path = os.path.join(args.coord_dir, f"coordinates_data_{file_idx}.json")
        if not os.path.exists(coord_path):
            continue

        with open(coord_path) as f:
            triplets = json.load(f)

        for local_idx, triplet in enumerate(triplets):
            lat, lon, nc_file = float(triplet[0][0]), float(triplet[0][1]), str(triplet[0][2])

            if not (args.lat_min <= lat <= args.lat_max and args.lon_min <= lon <= args.lon_max):
                skipped["bbox"] += 1
                continue
            if globe.is_land(lat, lon):
                skipped["land"] += 1
                continue
            dt = parse_modis_filename(nc_file)
            if not is_daytime(lat, lon, dt):
                skipped["night"] += 1
                continue
            cf = get_cloud_fraction(lat, lon, nc_file, args.myd06_dir)
            if cf is None:
                skipped["no_myd06"] += 1
                continue
            if cf < args.cf_thresh:
                skipped["cloud"] += 1
                continue

            kept.append((file_idx, local_idx, lat, lon, dt))

        if file_idx % 10 == 0:
            print(f"  [{file_idx:3d}/{args.n_files-1}] kept={len(kept):,}  skipped={skipped}",
                  flush=True)

    print(f"\nFiltering done: {len(kept):,} tiles kept,  skipped={skipped}", flush=True)

    if not kept:
        print("ERROR: no tiles passed all filters.", file=sys.stderr)
        sys.exit(1)

    # ── Pass 2: extract anchor images ─────────────────────────────────────────
    print("\nPass 2: extracting anchor images...", flush=True)
    kept_sorted = sorted(kept, key=lambda x: x[0])

    tiles_list, lats, lons, time_strs = [], [], [], []

    for file_idx, group in groupby(kept_sorted, key=lambda x: x[0]):
        group   = list(group)
        mm_path = os.path.join(args.img_dir, f"orig_memmap{file_idx}.memmap")
        mm      = np.memmap(mm_path, dtype="float64", mode="r",
                            shape=(args.n_per_file, 3, 3, 128, 128))
        for _, local_idx, lat, lon, dt in group:
            tiles_list.append(mm[local_idx, 0].astype(np.float32))   # (3, 128, 128)
            lats.append(lat)
            lons.append(lon)
            time_strs.append(dt.isoformat())
        del mm
        print(f"  file {file_idx:3d}: extracted {len(group)} tiles", flush=True)

    tiles = np.stack(tiles_list)   # (N, 3, 128, 128)
    print(f"\nTotal extracted: {tiles.shape}  ({tiles.nbytes / 1e9:.2f} GB)", flush=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    tiles_path = os.path.join(args.output_dir, "nepac_tiles.npy")
    meta_path  = os.path.join(args.output_dir, "nepac_meta.npz")

    np.save(tiles_path, tiles)
    np.savez(meta_path,
             lat  = np.array(lats,      dtype=np.float32),
             lon  = np.array(lons,      dtype=np.float32),
             time = np.array(time_strs))

    # These sentinel lines are parsed by the caller.
    print(f"SAVED_TILES={tiles_path}", flush=True)
    print(f"SAVED_META={meta_path}",  flush=True)


if __name__ == "__main__":
    main()
