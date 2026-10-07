"""Build the planning trajectory anchors by k-means over the training set.

The model embeds anchors at 10 Hz over the 4 s horizon, shape (num_anchors, 40, 3),
and subsamples `[:, 4::5]` to the 8 poses it actually predicts. Ground-truth
trajectories in the cache are those 8 poses, so each cluster centre is resampled to
40 poses before it is written out.

Default source is the navsim logs, which hold the trajectories without any of the
sensor data -- the feature cache stores ~4.5 MB per sample, so reading the whole
split from it would mean caching hundreds of GB for 192 bytes of trajectory each.

    python navsim/planning/script/run_trajectory_anchor_kmeans.py \
        --out trajectory_anchors/trajectory_anchors_256_kmeans.npy --num-anchors 256

    # or reuse a cache that already exists
    python navsim/planning/script/run_trajectory_anchor_kmeans.py --cache exp/safedrive_train_cache
"""
import argparse
import glob
import gzip
import os
import pickle
import time

import numpy as np
import yaml
from sklearn.cluster import KMeans

HORIZON_S = 4.0
ANCHOR_HZ = 10                      # anchors are stored at 10 Hz ...
GT_STRIDE = 5                       # ... and the model reads every 5th pose (2 Hz)


def load_gt_trajectories(cache_dir, limit=None):
    """Collect the ground-truth ego trajectories (N, 8, 3) from a feature cache."""
    files = sorted(glob.glob(os.path.join(cache_dir, "*", "*", "transfuser_target.gz")))
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"no cached targets under {cache_dir}")

    trajectories = []
    for i, path in enumerate(files, 1):
        with gzip.open(path, "rb") as f:
            target = pickle.load(f)
        trajectories.append(np.asarray(target["trajectory"], dtype=np.float32))
        if i % 5000 == 0:
            print(f"  read {i}/{len(files)}")
    return np.stack(trajectories)


def load_gt_trajectories_from_logs(split, num_poses, limit=None):
    """Read the ground-truth trajectories (N, 8, 3) straight from the navsim logs."""
    from pathlib import Path

    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import SceneLoader

    repo = Path(__file__).resolve().parents[3]
    filter_yaml = repo / f"navsim/planning/script/config/common/train_test_split/scene_filter/{split}.yaml"
    spec = yaml.safe_load(filter_yaml.read_text())
    data_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    data_split = "test" if split.startswith("navtest") else "trainval"

    scene_filter = SceneFilter(
        num_history_frames=spec["num_history_frames"],
        num_future_frames=spec["num_future_frames"],
        frame_interval=spec.get("frame_interval", 1),
        has_route=spec.get("has_route", True),
        max_scenes=limit,
        log_names=spec.get("log_names"),
        tokens=spec.get("tokens"),
    )
    loader = SceneLoader(
        sensor_blobs_path=data_root / f"sensor_blobs/{data_split}",
        data_path=data_root / f"navsim_logs/{data_split}",
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )

    tokens = loader.tokens
    print(f"[anchors] {len(tokens)} tokens in {split}")
    trajectories, started = [], time.time()
    for i, token in enumerate(tokens, 1):
        scene = loader.get_scene_from_token(token)
        trajectories.append(
            np.asarray(scene.get_future_trajectory(num_trajectory_frames=num_poses).poses,
                       dtype=np.float32))
        if i % 10000 == 0:
            rate = i / (time.time() - started)
            print(f"  {i}/{len(tokens)}  ({rate:.0f} tok/s, "
                  f"{(len(tokens) - i) / rate / 60:.1f} min left)")
    return np.stack(trajectories)


def resample_to_anchor_rate(trajectories):
    """(N, 8, 3) at 2 Hz -> (N, 40, 3) at 10 Hz, starting from the ego pose at t=0."""
    n, num_gt, dim = trajectories.shape
    gt_times = np.arange(1, num_gt + 1) * (HORIZON_S / num_gt)          # 0.5 .. 4.0
    anchor_times = np.arange(1, int(HORIZON_S * ANCHOR_HZ) + 1) / ANCHOR_HZ  # 0.1 .. 4.0

    # the ego is at the origin at t=0, which anchors the interpolation
    padded_times = np.concatenate([[0.0], gt_times])
    out = np.zeros((n, len(anchor_times), dim), dtype=np.float32)
    for i in range(n):
        padded = np.concatenate([np.zeros((1, dim), dtype=np.float32), trajectories[i]])
        for d in range(dim):
            out[i, :, d] = np.interp(anchor_times, padded_times, padded[:, d])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default=None,
                    help="read from this feature cache instead of the logs")
    ap.add_argument("--split", default="navtrain", help="scene filter to read the logs with")
    ap.add_argument("--out", default="trajectory_anchors/trajectory_anchors_256_kmeans.npy")
    ap.add_argument("--num-anchors", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N samples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.cache:
        trajectories = load_gt_trajectories(args.cache, args.limit)
    else:
        trajectories = load_gt_trajectories_from_logs(args.split, num_poses=8, limit=args.limit)
    print(f"[anchors] {trajectories.shape[0]} trajectories, shape {trajectories.shape[1:]}")

    flat = trajectories.reshape(len(trajectories), -1)
    kmeans = KMeans(n_clusters=args.num_anchors, n_init="auto", random_state=args.seed)
    kmeans.fit(flat)
    centres = kmeans.cluster_centers_.reshape(args.num_anchors, *trajectories.shape[1:])

    # snap every centre to its nearest real trajectory so the anchors stay drivable
    snapped = np.zeros_like(centres)
    for c in range(args.num_anchors):
        members = np.where(kmeans.labels_ == c)[0]
        if len(members) == 0:
            snapped[c] = centres[c]
            continue
        d = np.linalg.norm(flat[members] - kmeans.cluster_centers_[c], axis=1)
        snapped[c] = trajectories[members[np.argmin(d)]]

    anchors = resample_to_anchor_rate(snapped)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, anchors)

    check = anchors[:, GT_STRIDE - 1::GT_STRIDE]
    print(f"[anchors] wrote {args.out}  {anchors.shape}  ({anchors.nbytes / 1024:.0f} KB)")
    print(f"[anchors] subsampled [4::5] -> {check.shape}, "
          f"max |anchor - source| = {np.abs(check - snapped).max():.4f}")


if __name__ == "__main__":
    main()
