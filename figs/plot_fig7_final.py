#!/usr/bin/env python3
# figure7_metric3dv2_reproduce.py
#
# *File-based* reproduction of Metric3Dv2 Fig-7 on ETH-3D DSLR split
# ─────────────────────────────────────────────────────────────────
#  • reads <scene>/metric3dv2.h5  (keys: depth, depth_variance, one group per image)
#  • reads <scene>/gt_depth/*.npy (ground-truth depth in metres)
#  • *no* per-scene scale alignment – instead, a single global scale
#  • sensitivity uses raw σprior for ranking (no temperature applied)
#  • only one temperature α is fitted for the calibration plot
#  • depth-proportional σ = √k ⋅ depth with k=1.6e-4 (paper constant)
#
# Directory layout example:
#   cache_dir/
#       courtyard/
#           metric3dv2.h5
#           gt_depth/DSC_0286.npy ...
#       delivery_area/...
# ----------------------------------------------------------------

import h5py, cv2, math, tqdm, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

# ───────────────────────────── config ────────────────────────────
DATA_ROOT = Path("/mnt/data/0/cedar/datasets/mpsfm_local/benchmarks/eth3d/cache_dir")
SCENES = [
    "courtyard", "delivery_area", "electro", "facade", "kicker", "meadow",
    "office", "pipes", "playground", "relief", "relief_2", "terrace", "terrains",
]
K_CONST = 1.6e-4                          # σ_depth = √k * depth

# ───────────────────── helper functions ──────────────────────────
def resize(x: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """bilinear resize to `hw` (H,W) if shape differs"""
    return x if x.shape == hw else cv2.resize(x.astype(np.float32), (hw[1], hw[0]),
                                              interpolation=cv2.INTER_LINEAR)

def make_sensitivity(err: np.ndarray, sigma: np.ndarray):
    order = np.argsort(sigma)
    cum   = np.sqrt(np.cumsum(err[order] ** 2) / np.arange(1, len(err) + 1))
    rec   = np.arange(1, len(err) + 1) / len(err) * 100.0
    return rec, cum

# ─────────────────────────── main ───────────────────────────────
def main():
    # gather predictions & GT for *all* images first
    pred_depths, pred_sigmas, gt_depths, masks = [], [], [], []

    for scene in tqdm.tqdm(SCENES, desc="Collect"):
        scene_dir = DATA_ROOT / scene
        h5_path   = scene_dir / "metric3dv2.h5"
        gt_dir    = scene_dir / "gt_depth"

        with h5py.File(h5_path, "r") as f:
            for img_key in f.keys():                      # one HDF5 group per image
                d_pred = f[img_key]["depth"][()].astype(np.float32)  # (H,W)
                s_pred = np.sqrt(f[img_key]["depth_variance"][()]).astype(np.float32)

                gt_path = gt_dir / Path(img_key).with_suffix(".npy")
                if not gt_path.exists():
                    continue
                d_gt = np.load(gt_path).squeeze().astype(np.float32)
                mask = np.isfinite(d_gt) & (d_gt > 0)     # keep all valid metres

                # resize predictions to GT resolution if needed
                d_pred = resize(d_pred, d_gt.shape)
                s_pred = resize(s_pred, d_gt.shape)

                pred_depths.append(d_pred)
                pred_sigmas.append(s_pred)
                gt_depths  .append(d_gt)
                masks      .append(mask)

    # ── global metric scale ─────────────────────────────────────────
    pred_all = np.concatenate([p[m] for p, m in zip(pred_depths, masks)])
    gt_all   = np.concatenate([g[m] for g, m in zip(gt_depths,  masks)])

    med_pred = np.median(pred_all)
    med_gt   = np.median(gt_all)
    g_scale  = med_gt / med_pred
    print(f"[global] median-scale factor = {g_scale:.3f}")

    # apply global scale to depth *and* sigma
    pred_depths = [p * g_scale for p in pred_depths]
    pred_sigmas = [s * g_scale for s in pred_sigmas]

    # ── flatten to 1-D arrays ──
    all_err, all_depth, sigma_prior_raw = [], [], []
    for p, s, g, m in zip(pred_depths, pred_sigmas, gt_depths, masks):
        all_err.extend(np.abs(p - g)[m])
        all_depth.extend(p[m])
        sigma_prior_raw.extend(s[m])

    all_err          = np.asarray(all_err,          np.float32)
    all_depth        = np.asarray(all_depth,        np.float32)
    sigma_prior_raw  = np.asarray(sigma_prior_raw,  np.float32)

    # depth-proportional uncertainty
    sigma_depth = np.sqrt(K_CONST) * all_depth
    sigma_comb  = np.maximum(sigma_prior_raw, sigma_depth)

    # temperature scaling of σ_prior for calibration plot only
    alpha_p = (sigma_prior_raw * all_err).sum() / (sigma_prior_raw**2).sum()
    sigma_prior_cal = sigma_prior_raw * alpha_p
    sigma_comb_cal  = np.maximum(sigma_prior_cal, sigma_depth)

    # ─────────────────── plotting ───────────────────
    fig, (ax_s, ax_c) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    MODEL = "Metric3Dv2-file"

    # sensitivity upper bound
    ub_sorted = np.sort(all_err)
    ax_s.plot(np.linspace(0, 100, len(ub_sorted)),
              np.sqrt(np.cumsum(ub_sorted**2) / np.arange(1, len(ub_sorted)+1)),
              'k', label="Upper Bound")

    for σ, col, lab in [(sigma_prior_raw, 'C0', 'Prior'),
                        (sigma_depth,     'C1', 'Depth'),
                        (sigma_comb,      'C2', 'Combined')]:
        rc, rm = make_sensitivity(all_err, σ)
        ax_s.plot(rc, rm, col, label=lab)

    ax_s.set_xlim(0, 100); ax_s.set_ylim(0.8, 0.0)  # invert y
    ax_s.set_xlabel("Recall (%)"); ax_s.set_ylabel("RMSE (m)")
    ax_s.set_title(f"{MODEL} – Sensitivity (k={K_CONST:.1e})")
    ax_s.legend()

    # calibration (Combined, temperature-scaled)
    vmax = 2.0
    bins = np.linspace(0, vmax, 41)
    idx  = np.clip(np.digitize(sigma_comb_cal, bins)-1, 0, len(bins)-2)

    rmse_bin = [np.sqrt((all_err[idx==b]**2).mean()) if (idx==b).any() else np.nan
                for b in range(len(bins)-1)]
    freq_bin = np.bincount(idx, minlength=len(bins)-1) / len(all_err) * 100
    centers  = 0.5 * (bins[:-1] + bins[1:])

    ax_c.plot(centers, rmse_bin, 'C2', lw=2, label="Combined")
    ax_c.plot([0, vmax], [0, vmax], '--k', label="y=x")
    ax_c.set_xlim(0, vmax); ax_c.set_ylim(0, vmax)
    ax_c.set_xlabel("Depth StdDev (m)"); ax_c.set_ylabel("RMSE (m)")
    ax_c.legend(loc="upper left")
    ax_c.twinx().bar(centers, freq_bin, width=0.04, color='grey', alpha=.3)
    ax_c.set_title(f"{MODEL} – Uncertainty Calibration")

    plt.savefig("figure7_metric3dv2_reproduce.png", dpi=300)
    print("✓ Saved → figure7_metric3dv2_reproduce.png")

# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
