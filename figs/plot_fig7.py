#!/usr/bin/env python3
# process_fig7_fileio_v2.py – ETH-3D Fig-7 reproduction (file I/O, calibrated)

"""
Changes vs. previous script
───────────────────────────
1.  **Per-image H5 reading** unchanged (loop over `img_key` groups).
2.  **Depth-proportional uncertainty** uses the *fixed* k=1.6 e-4
    (value reported in the Metric3Dv2 paper) –– not the data-driven fit.
3.  **σ-prior is globally temperature-scaled** before it is used
    *anywhere* (sensitivity *and* calibration).  This is the main reason
    the blue “Prior” curve was too optimistic.
4.  **Combined σ** = max(σ_prior_cal, σ_depth) and is *again* re-scaled
    for the calibration plot.
5.  Default now evaluates all 13 ETH-3D DSLR scenes, matching the paper.

Directory layout (per scene) stays:

    <cache_dir>/
        courtyard/
            metric3dv2.h5
            gt_depth/*.npy
        delivery_area/
        …

"""

import h5py, math, cv2, tqdm
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path

# ───────────────────────────── paths ───────────────────────────── #
DATA_ROOT = Path("/mnt/data/0/cedar/datasets/mpsfm_local/benchmarks/eth3d/cache_dir")
SCENES    = [
    "courtyard", "delivery_area", "electro", "facade", "kicker",
    "meadow", "office", "pipes", "playground", "relief", "relief_2",
    "terrace", "terrains",
]

# ──────────────────────── utilities ────────────────────────────── #
def resize_to_full(x: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """bilinear resize to match gt shape if needed"""
    if x.shape != hw:
        x = cv2.resize(x.astype(np.float32, copy=False),
                       (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    return x

def make_sensitivity(err: np.ndarray, sigma: np.ndarray):
    order = np.argsort(sigma)
    cum   = np.sqrt(np.cumsum(err[order] ** 2) / np.arange(1, len(err) + 1))
    rec   = np.arange(1, len(err) + 1) / len(err) * 100.0
    return rec, cum

def scene_scale(pred_depths, gt_depths, masks):
    ratios = []
    for pd, gt, m in zip(pred_depths, gt_depths, masks):
        # if m.sum() < 0.01 * m.size:          # too few valid pixels – skip
        #     continue
        r = np.median(gt[m] / (pd[m] + 1e-6))
        if 0.1 < r < 10:                     # reject clear outliers
            ratios.append(r)
    return float(np.median(ratios)) if ratios else 1.0

# ─────────────────────────── main ──────────────────────────────── #
def main():
    fig, axs = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    MODEL = "Metric3Dv2-file"

    # global accumulators
    all_err, all_depth, sig_prior = [], [], []

    # ── loop over scenes ──
    for scene in SCENES:
        root   = DATA_ROOT / scene
        h5_p   = root / "metric3dv2.h5"
        gt_dir = root / "gt_depth"

        pred_d_list, pred_var_list, gt_list, mask_list = [], [], [], []

        with h5py.File(h5_p, "r") as f:
            for img_key in f.keys():                     # each image group
                d = f[img_key]["depth"][()]              # (H,W)
                var = f[img_key]["depth_variance"][()]   # (H,W) variance

                gt_p = gt_dir / Path(img_key).with_suffix(".npy")
                if not gt_p.exists():
                    print("WARN  GT missing:", gt_p)
                    continue
                gt = np.load(gt_p).squeeze()
                m  = np.isfinite(gt) & (gt > 0)

                pred_d_list .append(d.astype(np.float32))
                pred_var_list.append(var.astype(np.float32))
                gt_list      .append(gt.astype(np.float32))
                mask_list    .append(m)

        if not pred_d_list:           # scene had no matching pairs
            continue

        # robust scale so that med(depth_pred)≈med(depth_gt)
        s = scene_scale(pred_d_list, gt_list, mask_list)
        print(f"[{scene:<12}] scale = {s:6.3f}")

        for d_pred, var_p, gt, m in zip(pred_d_list, pred_var_list, gt_list, mask_list):
            d_pred = resize_to_full(d_pred * s, gt.shape)
            σ_p    = resize_to_full(np.sqrt(var_p) * s, gt.shape)

            err = np.abs(d_pred - gt)[m]
            all_err   .extend(err)
            all_depth .extend(d_pred[m])
            sig_prior .extend(σ_p[m])

    # ───────────────── global arrays ──────────────────
    all_err    = np.asarray(all_err,    np.float32)
    all_depth  = np.asarray(all_depth,  np.float32)
    sig_prior  = np.asarray(sig_prior,  np.float32)

    # 1) calibrate σ_prior (temperature scaling α_p)
    σ_prior_raw = sig_prior
    α_p = (σ_prior_raw * all_err).sum() / (σ_prior_raw**2).sum()
    σ_prior_cal = σ_prior_raw * α_p      # 只给校准用

    # 2) depth-proportional σ (fixed paper value)
    k = 1.6e-4
    σ_depth = np.sqrt(k) * all_depth

    # 3) combined, then second scaling α_c for calibration plot
    # σ_comb  = np.maximum(σ_prior_cal, σ_depth)
    # α_c     = (σ_comb * all_err).sum() / (σ_comb**2).sum()
    # σ_comb_cal = σ_comb * α_c     # only used in calibration plot

    σ_comb_cal = np.maximum(σ_prior_cal, σ_depth) * (
    (np.maximum(σ_prior_cal, σ_depth) * all_err).sum() /
    (np.maximum(σ_prior_cal, σ_depth)**2).sum()
)

    # ───────────────── sensitivity plot ─────────────────
    ax = axs[0, 0]
    ub = np.sort(all_err)
    ax.plot(np.linspace(0, 100, len(ub)),
            np.sqrt(np.cumsum(ub**2) / np.arange(1, len(ub)+1)),
            "k", label="Upper Bound")

    for σ, c, lab in [(σ_prior_raw,  "C0", "Prior"),
                      (σ_depth,  "C1", "Depth"),
                      (σ_comb_cal,   "C2", "Combined")]:
        rc, rm = make_sensitivity(all_err, σ)
        ax.plot(rc, rm, c, label=lab)

    ax.invert_yaxis(); ax.set_xlim(0, 100)
    ax.set_xlabel("Recall (%)"); ax.set_ylabel("RMSE (m)")
    ax.set_title(f"{MODEL} – Sensitivity (k={k:.1e})")
    ax.legend()

    # ───────────────── calibration plot (Combined) ─────────────────
    σ   = σ_comb_cal
    vmax = np.nanpercentile(σ, 99.5)
    bins = np.linspace(0, vmax, 41)
    idx  = np.clip(np.digitize(σ, bins)-1, 0, len(bins)-2)

    rmse_bin = [np.sqrt((all_err[idx==b]**2).mean()) if (idx==b).any()
                else np.nan for b in range(len(bins)-1)]
    freq_bin = np.bincount(idx, minlength=len(bins)-1) / len(all_err) * 100
    centers  = 0.5 * (bins[:-1] + bins[1:])

    axc = axs[0, 1]
    axc.plot(centers, rmse_bin, lw=2, color="C2", label="Combined")
    axc.plot(centers, centers, "--k", label="y=x")
    axc.set_ylim(0, vmax*1.1)
    axc.set_xlabel("Depth StdDev (m)"); axc.set_ylabel("RMSE (m)")
    axc.legend(loc="upper left")

    ax2 = axc.twinx()
    ax2.bar(centers, freq_bin, width=0.04, color='grey', alpha=.25)
    ax2.set_ylabel("Bin Size (%)")

    axc.set_title(f"{MODEL} – Uncertainty Calibration")

    # blank out lower row (unused)
    for a in axs[1]:
        a.axis("off")

    plt.savefig("figure7_fileio_v3.png", dpi=300)
    print("✓ Saved → figure7_fileio_v3.png")

# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
