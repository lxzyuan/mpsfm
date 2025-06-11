"""Reproduce the depth uncertainty analysis from Figure 7 of the paper.

The script loads Metric3Dv2 predictions and ground‑truth depth for all ETH3D
scenes and produces the sensitivity and calibration plots used in the paper.

By default it expects the ETH3D cache directory to be located at
``/mnt/data/0/cedar/datasets/mpsfm_local/benchmarks/eth3d/cache_dir``.  You can
override this path by either setting the ``ETH3D_CACHE_DIR`` environment
variable or passing ``--data_root`` on the command line.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Default directory containing all ETH3D scenes
DEFAULT_ROOT = Path("/mnt/data/0/cedar/datasets/mpsfm_local/benchmarks/eth3d/cache_dir")
DATA_ROOT = Path(os.environ.get("ETH3D_CACHE_DIR", DEFAULT_ROOT))
# All scenes that provide predictions and ground truth
SCENES = [
    "courtyard",
    "delivery_area",
    "electro",
    "facade",
    "kicker",
    "meadow",
    "office",
    "pipes",
    "playground",
    "relief",
    "relief_2",
    "terrace",
    "terrains",
]
H5_NAME = "metric3dv2.h5"
GT_SUBDIR = "gt_depth"
# Depth proportional uncertainty factor reported by Metric3Dv2
K_CONST = 1.6e-4


def resize_to_match(arr: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Resize array to the given height/width if necessary."""
    if arr.shape == hw:
        return arr
    return cv2.resize(arr.astype(np.float32), (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)


def load_dataset(data_root: Path, scenes: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all valid pixels across all scenes."""
    all_err: list[np.ndarray] = []
    all_depth: list[np.ndarray] = []
    all_sigma: list[np.ndarray] = []

    for scene in tqdm(scenes, desc="Scenes"):
        scene_dir = data_root / scene
        h5_path = scene_dir / H5_NAME
        gt_dir = scene_dir / GT_SUBDIR
        if not h5_path.exists() or not gt_dir.exists():
            continue
        with h5py.File(h5_path, "r") as f:
            for img_key in f:
                gt_path = gt_dir / f"{img_key}.npy"
                if not gt_path.exists():
                    continue
                gt = np.load(gt_path).astype(np.float32)
                pred = f[img_key]["depth"][()].astype(np.float32)
                if "depth_variance" in f[img_key]:
                    sigma = np.sqrt(f[img_key]["depth_variance"][()].astype(np.float32))
                else:
                    sigma = f[img_key]["uncertainty"][()].astype(np.float32)
                pred = resize_to_match(pred, gt.shape)
                sigma = resize_to_match(sigma, gt.shape)

                mask = np.isfinite(gt) & (gt > 0)
                if not np.any(mask):
                    continue
                # scale only alignment per image
                scale = np.median(gt[mask]) / np.median(pred[mask])
                pred *= scale
                sigma *= scale

                all_err.append(np.abs(pred - gt)[mask])
                all_depth.append(pred[mask])
                all_sigma.append(sigma[mask])

    if not all_err:
        raise RuntimeError(f"No valid data found under {data_root}. Check the path and content.")

    return (
        np.concatenate(all_err),
        np.concatenate(all_depth),
        np.concatenate(all_sigma),
    )


def make_sensitivity(err: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(sigma)
    rmse = np.sqrt(np.cumsum(err[order] ** 2) / np.arange(1, len(err) + 1))
    recall = np.arange(1, len(err) + 1) / len(err) * 100.0
    return recall, rmse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=Path,
        default=DATA_ROOT,
        help="Path to ETH3D cache directory",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    err, depth, sigma_prior = load_dataset(args.data_root, SCENES)

    sigma_depth = np.sqrt(K_CONST) * depth
    sigma_comb = np.maximum(sigma_prior, sigma_depth)

    # temperature scaling for calibration
    alpha = (sigma_comb * err).sum() / (sigma_comb**2).sum()
    sigma_comb_cal = sigma_comb * alpha

    fig, (ax_s, ax_c) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    # Sensitivity curves
    upper_sorted = np.sort(err)
    ax_s.plot(
        np.linspace(0, 100, len(upper_sorted)),
        np.sqrt(np.cumsum(upper_sorted**2) / np.arange(1, len(upper_sorted) + 1)),
        "k",
        label="Upper Bound",
    )
    for sigma, color, label in [
        (sigma_prior, "C0", "Prior"),
        (sigma_depth, "C1", "Depth"),
        (sigma_comb, "C2", "Combined"),
    ]:
        rec, rm = make_sensitivity(err, sigma)
        ax_s.plot(rec, rm, color, label=label)

    ax_s.set_xlim(0, 100)
    ax_s.set_ylim(ax_s.get_ylim()[1], ax_s.get_ylim()[0])
    ax_s.set_xlabel("Recall (%)")
    ax_s.set_ylabel("RMSE (m)")
    ax_s.set_title("Metric3Dv2 Depth Sensitivity Analysis")
    ax_s.legend()

    # Calibration plot (Combined)
    vmax = 0.4
    bins = np.linspace(0, vmax, 20)
    idx = np.clip(np.digitize(sigma_comb_cal, bins) - 1, 0, len(bins) - 2)
    rmse_bin = [np.sqrt((err[idx == b] ** 2).mean()) if np.any(idx == b) else np.nan for b in range(len(bins) - 1)]
    freq_bin = np.bincount(idx, minlength=len(bins) - 1) / len(err) * 100
    centers = 0.5 * (bins[:-1] + bins[1:])

    ax_c.plot(centers, rmse_bin, "C2", lw=2, label="Combined")
    ax_c.plot([0, vmax], [0, vmax], "k--", label="y=x")
    ax_c.set_xlim(0, vmax)
    ax_c.set_ylim(0, vmax)
    ax_c.set_xlabel("Depth StdDev (m)")
    ax_c.set_ylabel("RMSE (m)")
    ax_c.legend(loc="upper left")

    ax_h = ax_c.twinx()
    ax_h.bar(centers, freq_bin, width=0.02, color="C2", alpha=0.25)
    ax_h.set_ylabel("Bin Size (%)")

    ax_c.set_title("Metric3Dv2 Depth Uncertainty Calibration")

    plt.savefig("reproduce_fig7.png", dpi=300)
    print("Saved → reproduce_fig7.png")


if __name__ == "__main__":
    main(parse_args())
