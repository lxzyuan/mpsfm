#!/usr/bin/env python3
# figure7_eth3d.py
"""
Reproduce Figure-7 (depth-uncertainty analysis on ETH3D) for
Metric3Dv2  and  MASt3R.

*   Automatically fits the depth-proportional uncertainty factor **k**
    for every model where `k=None`.
*   Works with ETH3D *dslr* split:
        <scene>_dslr_images/**/dslr_images/*.JPG   – RGB
        <scene>_dslr_depth/**/dslr_images/*.png    – 16 bit or 32 bit depth (m / mm)
"""

import math, cv2, tqdm, torch, imageio.v2 as iio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pycolmap

# ─────────────────────────────── paths ─────────────────────────────── #
DATA_ROOT_depth = Path("/mnt/data/0/cedar/datasets/ETH3D")
DATA_ROOT = Path("/home/cedar/gitcode/mpsfm/local/benchmarks/eth3d/data")
SCENES     = ["courtyard"]          # 可换成全 20-30 个场景


# ─────────────────────────── model registry ────────────────────────── #
MODEL_ZOO = {
    "Metric3Dv2": dict(
        factory=lambda: __import__(
            "mpsfm.extraction.imagewise.geometry.models.depth.metric3dv2",
            fromlist=["Metric3Dv2"]
        ).Metric3Dv2({"return_types": ["depth", "depth_variance"]}),
        required_inputs=["image"],        # 单视图
        k=None,                          # ← 自适应求解
        use_combined=True,
    ),
    # "MASt3R": dict(
    #     factory=lambda: __import__(
    #         "mpsfm.extraction.pairwise.models.mast3r",
    #         fromlist=["Mast3rMatcher"]
    #     ).Mast3rMatcher({}),
    #     required_inputs=["image0", "image1"],  # 双视图
    #     k=0.0,                                 # 只用 prior
    #     use_combined=False,
    # ),
}


def load_cameras_txt(path: Path):
    """
    Parse COLMAP cameras.txt → {camera_id: (model, w, h, params[list])}
    """
    cams = {}
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            toks = line.split()
            cid   = int(toks[0])
            model = toks[1]
            w, h  = map(int, toks[2:4])
            params = list(map(float, toks[4:]))
            cams[cid] = (model, w, h, params)
    return cams

def load_images_txt(path: Path):
    """
    Return dict[image_name] = camera_id
    images.txt has *two lines* per image; only the 1st contains CAMERA_ID.
    """
    imgs = {}
    with path.open() as f:
        while True:
            line1 = f.readline()
            if not line1:
                break                                # EOF
            if line1.startswith("#") or not line1.strip():
                continue                             # comment / blank
            line2 = f.readline()                     # skip point line

            toks = line1.split()
            if len(toks) < 10:
                continue
            cam_id   = int(toks[8])                  # 9-th token
            img_name = toks[9]                       # 10-th token
            imgs[img_name] = cam_id
    return imgs


def make_K(params):
    """
    目前 ETH3D DSLR 用 THIN_PRISM_FISHEYE:
      params = [fx, fy, cx, cy, k1, k2, k3, k4, p1, p2]
    取前 4 个即可得到 3×3 内参矩阵 K
    """
    fx, fy, cx, cy = params[:4]
    # K = np.array([[fx, 0,  cx],
    #               [0,  fy, cy],
    #               [0,   0,  1]], dtype=np.float32)
    return np.array([fx, fy, cx, cy], dtype=np.float32)


# ──────────────────────────── ETH3D I/O ────────────────────────────── #
def list_frames(scene: str):
    rgb_dir   = DATA_ROOT / scene / "images"
    depth_dir = DATA_ROOT_depth / f"{scene}_dslr_depth" / scene / "ground_truth_depth/dslr_images"

    for rgb in sorted(rgb_dir.glob("*.png")):
        yield rgb, depth_dir / (rgb.stem + ".JPG")

def read_depth_float32_jpg(path, hw):
    """ETH3D _dslr_depth: 伪 JPG, 实为 little-endian float32"""
    H, W = hw
    with open(path, "rb") as f:
        buf = f.read()
    depth = np.frombuffer(buf, "<f4").reshape(H, W)
    return depth


# ──────────────────────────── inference ────────────────────────────── #
def resize_to_full(x, hw):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = x.astype(np.float32, copy=False)      # ensure float32

    # → Resize only when尺寸不同
    if x.shape != hw:
        x = cv2.resize(x, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    return x

def get_pred(net, rgb_np, intr, pair_np=None, names=("im0", "im1")):
    reqs = net.required_inputs
    if "image" in reqs:                                  # Metric3Dv2
        # rgb_t = torch.from_numpy(rgb_np.transpose(2,0,1)).unsqueeze(0).cuda() / 255.
        # intr_t = torch.from_numpy(intr).unsqueeze(0).cuda()
        im = rgb_np.astype(np.float32) / 255.0 
        with torch.no_grad():
            out = net({"image": im, "intrinsics": intr})
        depth = out["depth"]
        sigma = np.sqrt(out["depth_variance"])
    elif "image0" in reqs:                               # MASt3R
        i0 = torch.from_numpy(rgb_np.transpose(2,0,1)).unsqueeze(0).cuda()/255.
        i1 = torch.from_numpy(pair_np.transpose(2,0,1)).unsqueeze(0).cuda()/255.
        with torch.no_grad():
            out = net({"image0": i0, "image1": i1, "name0": names[0], "name1": names[1]}, mode="depth")
        depth = out["depth0"][0,0].cpu().numpy()
        sigma = np.sqrt(out["variance0"][0,0].cpu().numpy())
    else:
        raise ValueError(f"Unsupported required_inputs: {reqs}")

    return depth, sigma


# ──────────────────────────── utils ────────────────────────────────── #
def make_sensitivity(err, sig):
    order  = np.argsort(sig)
    cum_rmse = np.sqrt(np.cumsum(err[order]**2) / np.arange(1, len(err)+1))
    recall = np.arange(1, len(err)+1) / len(err) * 100.0
    return recall, cum_rmse


# ───────────────────────────── main ────────────────────────────────── #
def main():
    fig, axs = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)

    for r, (name, cfg) in enumerate(MODEL_ZOO.items()):
        net = cfg["factory"]().eval().cuda()
        net.required_inputs = cfg["required_inputs"]          # 强制声明

        all_err, all_depth = [], []
        sig_prior_list = []

        # ── pass 1: collect depth / error / prior σ ──
        for scene in tqdm.tqdm(SCENES, desc=f"{name}"):
            frames = list(list_frames(scene))
            rec_dir = DATA_ROOT / scene / "rec"
            rec = pycolmap.Reconstruction(rec_dir)
            for i, (rgb_p, dep_p) in enumerate(frames):
                rgb   = iio.imread(rgb_p)
                H,W   = rgb.shape[:2]
                depth_gt = read_depth_float32_jpg(dep_p, (H,W))
                mask = np.isfinite(depth_gt) & (depth_gt > 0)

                cams  = load_cameras_txt(camera_dir/"cameras.txt")
                imgs  = load_images_txt(camera_dir/"images.txt")
                name = "dslr_images/" + rgb_p.name
                cid  = imgs[name]
                intr = make_K(cams[cid][3])
                # H, W = rgb.shape[:2]
                # fx = fy = max(H, W)
                # cx, cy = W / 2.0, H / 2.0
                # # intr = np.array([fx, fy, cx, cy], np.float32)
                # intr = np.array([fx, fy, cx, cy], dtype=np.float32)

                # second image for MASt3R
                rgb_pair = None;  names = (rgb_p.name, rgb_p.name)
                if "image0" in net.required_inputs:
                    j = i+1 if i < len(frames)-1 else i-1
                    rgb_pair = iio.imread(frames[j][0])
                    names = (rgb_p.name, frames[j][0].name)

                d_pred, sig_prior = get_pred(net, rgb, intr, rgb_pair, names)
                scale = np.median(depth_gt[mask]) / np.median(d_pred[mask])
                print("###", scale)
                d_pred   *= scale
                sig_prior *= scale

                d_pred    = resize_to_full(d_pred, depth_gt.shape)
                sig_prior = resize_to_full(sig_prior, depth_gt.shape)

                err = np.abs(d_pred - depth_gt)[mask].astype(np.float32)
                all_err.extend(err)
                all_depth.extend(d_pred[mask])
                sig_prior_list.extend(sig_prior[mask])

        all_err   = np.asarray(all_err)
        all_depth = np.asarray(all_depth)
        sig_prior_arr = np.asarray(sig_prior_list)

        # ── fit k* if needed ──
        if cfg["k"] is None:
            num = np.sum(all_err**2 * all_depth**2)
            den = np.sum(all_depth**4) + 1e-12
            k_opt = num / den
        else:
            k_opt = cfg["k"]

        # σ_depth & σ_comb
        sig_depth_arr = math.sqrt(k_opt) * all_depth
        if cfg["use_combined"]:
            sig_comb_arr = np.maximum(sig_prior_arr, sig_depth_arr)
        else:
            sig_comb_arr = sig_prior_arr

        # ── sensitivity curves ──
        ax = axs[r, 0]
        colors = dict(prior="C0", depth="C1", comb="C2", upper="k")

        # Upper-bound (oracle) curve
        ord_err = np.sort(all_err)
        upper_rmse = np.sqrt(np.cumsum(ord_err**2) / np.arange(1, len(ord_err)+1))
        recall_all = np.arange(1, len(ord_err)+1) / len(ord_err) * 100
        ax.plot(recall_all, upper_rmse, color="k", label="Upper Bound")

        # prior
        rc, rm = make_sensitivity(all_err, sig_prior_arr)
        ax.plot(rc, rm, color="C0", label="Prior")
        # depth-prop
        if k_opt > 0:
            rc, rm = make_sensitivity(all_err, sig_depth_arr)
            ax.plot(rc, rm, color="C1", label="Depth")
        # comb
        if cfg["use_combined"]:
            rc, rm = make_sensitivity(all_err, sig_comb_arr)
            ax.plot(rc, rm, color="C2", label="Comb")

        ax.invert_yaxis();  ax.set_xlim(0, 100)
        ax.set_xlabel("Recall (%)");  ax.set_ylabel("RMSE (m)")
        ax.set_title(f"{name} – Sensitivity  (k={k_opt:.2e})")
        ax.legend()

        # ── calibration plot ──
        sel_arr = sig_comb_arr if cfg["use_combined"] else sig_prior_arr
        α = (sel_arr * all_err).sum() / (sel_arr**2).sum()   # global scale
        σ = sel_arr * α

        # bins = np.linspace(0, 2.0, 41)
        vmax = np.nanpercentile(σ, 99.5)
        bins = np.linspace(0, vmax, 41)

        idx  = np.digitize(σ, bins) - 1
        idx  = np.clip(idx, 0, len(bins)-2)

        rmse_bin  = [np.sqrt((all_err[idx==b]**2).mean()) if (idx==b).any() else np.nan
                     for b in range(len(bins)-1)]
        freq_bin  = np.bincount(idx, minlength=len(bins)-1) / len(all_err) * 100
        centers   = 0.5*(bins[:-1] + bins[1:])

        axc = axs[r, 1]
        lab = "Comb" if cfg["use_combined"] else "Prior"
        axc.plot(centers, rmse_bin, lw=2, color="C2" if cfg["use_combined"] else "C0", label=lab)
        axc.plot(centers, centers, "--k", label="Perfect")
        axc.set_xlabel("Depth StdDev (m)");  axc.set_ylabel("RMSE (m)");  axc.set_ylim(0, 2.2)
        axc.legend(loc="upper left")
        ax2 = axc.twinx()
        ax2.bar(centers, freq_bin, width=0.04, color="grey", alpha=.25)
        ax2.set_ylabel("Bin Size (%)")
        axc.set_title(f"{name} – Uncertainty Calibration")

    plt.savefig("figure7_reproduced.png", dpi=300)
    print("✓ Saved → figure7_reproduced.png")

if __name__ == "__main__":
    main()
