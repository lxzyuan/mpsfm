#!/usr/bin/env python3
# reproduce_fig7.py
"""
复现 MP-SfM 论文 Figure-7（ETH3D 深度误差 vs 不确定度曲线）

数据结构示例:
ETH3D/
├─ courtyard_dslr_images/          (解压 *_dslr_images.7z)
│   └─ courtyard/images/dslr_images/DSC_0286.JPG  ...
└─ courtyard_dslr_depth/           (解压 *_dslr_depth.7z)
    └─ courtyard/ground_truth_depth/dslr_images/DSC_0286.JPG  (float32 深度伪 JPG)
"""

import math
from pathlib import Path

import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

# ------------------ 配置区域 ------------------ #
DATA_ROOT = Path("/mnt/data/0/cedar/datasets/ETH3D")        # 修改为你的 ETH3D 路径


MODEL_ZOO = {      # 每个模型: 生成器、k 常数、是否用 Combined
    "Metric3Dv2": dict(
        factory=lambda: __import__(
            "mpsfm.extraction.imagewise.geometry.models.depth.metric3dv2",
            fromlist=["Metric3Dv2"],
        ).Metric3Dv2({"return_types": ["depth", "depth_variance"]}),
        k=1.6e-4,
        use_combined=True,
    ),
    "MASt3R": dict(
        factory = lambda: __import__(
            "mpsfm.extraction.pairwise.models.mast3r",
            fromlist=["Mast3rMatcher"],
        ).Mast3rMatcher({}),
        k = 0.0,
        use_combined = False,
    ),
}

SCENES = ["courtyard", "delivery_area", "facade", "forest", "indoor", "kicker",
          "meadow", "office", "pipes", "playground", "relief", "relief_2",
          "relief_3", "square", "terrains", "terrains_2", "terrains_3",
          "tunnel", "underpass", "wood_pile"]  # 20/30 够快，可按需增减
SCENES = ["courtyard"]
# ------------------------------------------------ #

def list_frames(scene):
    img_dir   = DATA_ROOT/f"{scene}_dslr_jpg"/scene/"images/dslr_images"
    depth_dir = DATA_ROOT/f"{scene}_dslr_depth"/scene/"ground_truth_depth/dslr_images"
    for jpg in sorted(img_dir.glob("*.JPG")):
        yield jpg, depth_dir/jpg.name

def read_depth_float32_jpg(path, hw):
    """ETH3D _dslr_depth: 伪 JPG, 实为 little-endian float32"""
    H, W = hw
    with open(path, "rb") as f:
        buf = f.read()
    depth = np.frombuffer(buf, "<f4").reshape(H, W)
    return depth

def get_pred(net, rgb):
    """Return depth and uncertainty from a geometry network."""
    im = rgb.astype(np.float32) / 255.0
    H, W = im.shape[:2]

    # simple pinhole intrinsics as fx, fy, cx, cy
    fx = fy = max(H, W)
    cx, cy = W / 2.0, H / 2.0
    intr = np.array([fx, fy, cx, cy], dtype=np.float32)

    with torch.no_grad():
        out = net({"image": im, "intrinsics": intr})

    depth = out["depth"]
    if "depth_variance" in out:
        sigma = np.sqrt(out["depth_variance"])
    else:
        sigma = out.get("sigma")
        if sigma is None:
            raise KeyError("Model output missing uncertainty information")

    return depth, sigma

def make_sensitivity(err, sig_arr):
    N = len(err)
    order = np.argsort(sig_arr)
    rmse = np.sqrt(np.cumsum(err[order]**2)/np.arange(1,N+1))
    recall = np.arange(1,N+1)/N*100.0
    return recall, rmse

def main():
    fig, axs = plt.subplots(2,2, figsize=(9,7), constrained_layout=True)

    for row,(name,cfg) in enumerate(MODEL_ZOO.items()):
        net = cfg["factory"]().eval().cuda()
        k = cfg["k"]
        use_comb = cfg["use_combined"]
        all_err = []
        sigs = {"prior": [], "depth": [], "comb": []}

        for scene in tqdm.tqdm(SCENES, desc=f"{name} scenes"):
            for rgb_path, depth_path in list_frames(scene):
                # 读取彩色 & 真值深度
                rgb   = iio.imread(rgb_path)
                H,W   = rgb.shape[:2]
                depth = read_depth_float32_jpg(depth_path, (H,W))
                mask  = np.isfinite(depth) & (depth>0)

                # 预测
                d_pred, sig_prior = get_pred(net, rgb)

                sig_depth = math.sqrt(k) * d_pred
                sig_comb  = np.maximum(sig_prior, sig_depth) if use_comb else sig_prior

                err = np.abs(d_pred-depth)[mask].astype(np.float32)
                all_err.extend(err)
                sigs["prior"].extend(sig_prior[mask])
                sigs["depth"].extend(sig_depth[mask])
                sigs["comb"].extend(sig_comb[mask])

        # --- 敏感度曲线 ---
        colors = dict(prior="C0", depth="C1", comb="C2", upper="k")
        ax = axs[row,0]
        N = len(all_err)
        all_err = np.asarray(all_err)
        upper_rmse = np.sqrt(np.cumsum(np.sort(all_err)**2)/np.arange(1,N+1))
        recall = np.arange(1,N+1)/N*100
        ax.plot(recall, upper_rmse, color=colors["upper"], label="Upper Bound")

        for tag in ("prior","depth","comb"):
            if tag=="depth" and k==0.0:
                continue
            r, rm = make_sensitivity(all_err, np.asarray(sigs[tag]))
            ax.plot(r, rm, color=colors[tag], label=tag.capitalize())

        ax.invert_yaxis()
        ax.set_xlim(0,100)
        ax.set_xlabel("Recall (%)")
        ax.set_ylabel("RMSE (m)")
        ax.set_title(f"{name} – Sensitivity")
        ax.legend()

        # --- 不确定度校准图 ---
        sel = "comb" if use_comb else "prior"
        σ = np.asarray(sigs[sel])
        err = all_err
        α = (σ*err).sum() / (σ**2).sum()         # 全局 scaling
        σ *= α

        bins = np.linspace(0, 2.0, 41)
        inds = np.digitize(σ, bins) - 1
        inds = np.clip(inds, 0, len(bins) - 2)
        rmse_bin = [np.sqrt((err[inds==b]**2).mean()) if (inds==b).any() else np.nan
                    for b in range(len(bins)-1)]
        freq_bin = np.bincount(inds, minlength=len(bins)-1)/len(err)*100
        center = (bins[:-1]+bins[1:])/2

        axc = axs[row,1]
        axc.plot(center, rmse_bin, color=colors[sel], lw=2, label=sel.capitalize())
        axc.plot(center, center, '--k', lw=1, label="Perfect")
        axc.set_xlabel("Depth StdDev (m)")
        axc.set_ylabel("RMSE (m)")
        axc.set_ylim(0,2.2)
        axc.legend(loc="upper left")
        ax2 = axc.twinx()
        ax2.bar(center, freq_bin, width=0.04, color=colors[sel], alpha=.25)
        ax2.set_ylabel("Bin Size (%)")
        axc.set_title(f"{name} – Uncertainty Calibration")

    plt.savefig("figure7_reproduced.png", dpi=300)
    print("Saved → figure7_reproduced.png")

if __name__ == "__main__":
    main()
