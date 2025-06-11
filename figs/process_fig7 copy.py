#!/usr/bin/env python3
# process_fig7.py  –  ETH3D depth-uncertainty reproduction (patched)

import math, cv2, tqdm, torch, imageio.v2 as iio
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import pycolmap

# ───────────────────────────── paths ───────────────────────────── #
DATA_ROOT       = Path("/mnt/data/0/cedar/datasets/mpsfm_local/benchmarks/eth3d/data")
DATA_ROOT_DEPTH = Path("/mnt/data/0/cedar/datasets/ETH3D")
# SCENES          = ["courtyard", "delivery_area", "electro", "facade", "kicker", "meadow",
#                    "office", "pipes", "playground", "relief", "relief_2", "terrace", "terrains"]              # or full list
SCENES = ["courtyard"]

# ─────────────────────────── model zoo ──────────────────────────── #
MODEL_ZOO = {
    "Metric3Dv2": dict(
        factory=lambda: __import__(
            "mpsfm.extraction.imagewise.geometry.models.depth.metric3dv2",
            fromlist=["Metric3Dv2"]
        ).Metric3Dv2({"return_types": ["depth", "depth_variance"]}),
        required_inputs=["image"],
        k=1.6e-4,     # None
        use_combined=True,
    ),
    # "MASt3R": {...}
}

# ───────────────────── helpers copied from prep-script ───────────── #


# ---------- NEW: robust scene-level scale ----------
def scene_scale(rec, img_by_name, frames, net):
    """return robust scale factor for one scene"""
    ratios = []
    for rgb_p, dep_p in frames:
        rgb = iio.imread(rgb_p)
        # im  = rec.images_by_name[rgb_p.name]
        im = img_by_name[rgb_p.name]
        cam = rec.cameras[im.camera_id]
        rot = get_rot90(im.cam_from_world)
        intr = make_K(cam.params)

        depth_full = read_depth_float32_jpg(dep_p)
        depth_gt   = cv2.resize(rotate_image(depth_full, rot),
                                rgb.shape[1::-1],
                                interpolation=cv2.INTER_NEAREST)
        mask = np.isfinite(depth_gt) & (depth_gt > 0)
        mask &= (depth_gt < 15)
        if mask.sum() < 0.01 * depth_gt.size:         #  <1% 像素 → 跳过
            continue

        with torch.no_grad():
            d_pred, _ = get_pred(net, rgb, intr)

        d_pred = cv2.resize(d_pred.astype(np.float32),
                            depth_gt.shape[1::-1],
                            interpolation=cv2.INTER_LINEAR)
        ratio = np.median(depth_gt[mask] / (d_pred[mask] + 1e-6))
        if 0.1 < ratio < 10:                          # 排除极端异常
            ratios.append(ratio)
    return float(np.median(ratios)) if ratios else 1.0

def get_rot90(cfw):
    g_w = np.array([0, 0, -1])
    g_c = cfw.rotation.matrix() @ g_w
    angle = np.rad2deg(np.arctan2(g_c[1], g_c[0]))
    return int((np.round(angle / 90) - 1) % 4)

def rotate_image(arr, k):
    return np.rot90(arr, k % 4, axes=(0, 1))

def make_K(params):
    return np.array(params[:4], dtype=np.float32)

# ───────────────────────── ETH3D I/O ────────────────────────────── #
def list_frames(scene: str):
    rgb_dir   = DATA_ROOT / scene / "images"
    depth_dir = DATA_ROOT_DEPTH / f"{scene}_dslr_depth" / scene / \
                "ground_truth_depth/dslr_images_processed"
    for rgb in sorted(rgb_dir.glob("*.png")):
        yield rgb, depth_dir / (rgb.stem + ".npy")

def _deduce_hw(numel: int) -> tuple[int, int]:
    """给 float32 元素数 → 返回 (H, W) 近似 3:2，且 W≥H。"""
    # 理论比值 1.5 左右；用 sqrt(n/1.5) 做首估，然后回退到可整除的高度
    h = int(round(math.sqrt(numel / 1.5)))
    h = max(1, min(h, int(math.sqrt(numel))))
    while numel % h:
        h -= 1
    w = numel // h
    if w < h:                    # 保证 W≥H
        h, w = w, h
    return h, w

def read_depth_float32_jpg(path: Path):
    """ETH3D dslr_depth：原始 little-endian float32 流，自动推断宽高"""
    with open(path, "rb") as f:
        depth = np.frombuffer(f.read(), "<f4")
    H, W = _deduce_hw(depth.size)
    return depth.reshape(H, W)

def resize_to_full(x, hw):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if x.shape != hw:
        x = cv2.resize(x.astype(np.float32, copy=False),
                       (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    return x

# ─────────────────────────── inference ──────────────────────────── #
def get_pred(net, rgb_np, intr, pair_np=None, names=("im0", "im1")):
    reqs = net.required_inputs
    if "image" in reqs:                                   # Metric3Dv2
        im = rgb_np.astype(np.float32) / 255.0
        with torch.no_grad():
            out = net({"image": im, "intrinsics": intr})
        return out["depth"], np.sqrt(out["depth_variance"])
    elif "image0" in reqs:                                # MASt3R
        i0 = torch.from_numpy(rgb_np.transpose(2,0,1)).unsqueeze(0).cuda()/255.
        i1 = torch.from_numpy(pair_np.transpose(2,0,1)).unsqueeze(0).cuda()/255.
        with torch.no_grad():
            out = net({"image0": i0, "image1": i1,
                        "name0": names[0], "name1": names[1]}, mode="depth")
        return out["depth0"][0,0].cpu().numpy(), \
               np.sqrt(out["variance0"][0,0].cpu().numpy())
    else:
        raise ValueError(f"Unsupported required_inputs: {reqs}")

def make_sensitivity(err, sig):
    order = np.argsort(sig)
    cum = np.sqrt(np.cumsum(err[order]**2) /
                  np.arange(1, len(err)+1))
    rec = np.arange(1, len(err)+1) / len(err) * 100.0
    return rec, cum

# ───────────────────────────── main ─────────────────────────────── #
def main():
    fig, axs = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)

    for r, (mname, cfg) in enumerate(MODEL_ZOO.items()):
        net = cfg["factory"]().eval().cuda()
        net.required_inputs = cfg["required_inputs"]

        all_err, all_depth, sig_prior_list = [], [], []

        for scene in tqdm.tqdm(SCENES, desc=mname):
            rec = pycolmap.Reconstruction(DATA_ROOT / scene / "rec")
            img_by_name = {im.name: im for im in rec.images.values()}

            frames = list(list_frames(scene))
            s_scene = scene_scale(rec, img_by_name, frames, net)
            print(f"[{scene}] robust scale = {s_scene:.3f}")

            for i, (rgb_p, dep_p) in enumerate(frames):
                rgb = iio.imread(rgb_p)
                H, W = rgb.shape[:2]

                im  = img_by_name[rgb_p.name]
                cam = rec.cameras[im.camera_id]
                rot = get_rot90(im.cam_from_world)
                intr = make_K(cam.params)

                depth_full = read_depth_float32_jpg(dep_p)
                depth_full = rotate_image(depth_full, rot) 
                depth_gt   = cv2.resize(
                    depth_full, (W, H), interpolation=cv2.INTER_NEAREST
                )
                mask = np.isfinite(depth_gt) & (depth_gt > 0)

                rgb_pair, names = None, (rgb_p.name, rgb_p.name)
                if "image0" in net.required_inputs:
                    j = i+1 if i < len(frames)-1 else i-1
                    rgb_pair = iio.imread(frames[j][0])
                    names = (rgb_p.name, frames[j][0].name)

                d_pred, sig_prior = get_pred(net, rgb, intr, rgb_pair, names)
                # scale = np.median(depth_gt[mask]) / np.median(d_pred[mask])
                # scale = np.median(depth_gt[mask] / (d_pred[mask]+1e-6))

                # print("###", scale)
                # if scale > 5:                    # 阈值自调
                #     print(rgb_p)                 # 哪一张图
                #     print("valid px:", mask.sum())
                #     print("gt med:", np.median(depth_gt[mask]),
                #         "pred med:", np.median(d_pred[mask]))
                #     # 可视化
                #     plt.subplot(1,2,1); plt.imshow(depth_gt, vmin=0, vmax=10); plt.title("GT")
                #     plt.subplot(1,2,2); plt.imshow(d_pred,  vmin=0, vmax=10); plt.title("Pred")
                #     plt.show()

                d_pred   *= s_scene
                sig_prior *= s_scene

                d_pred    = resize_to_full(d_pred, depth_gt.shape)
                sig_prior = resize_to_full(sig_prior, depth_gt.shape)

                err = np.abs(d_pred - depth_gt)[mask].astype(np.float32)
                all_err.extend(err)
                all_depth.extend(d_pred[mask])
                sig_prior_list.extend(sig_prior[mask])

        all_err   = np.asarray(all_err)
        all_depth = np.asarray(all_depth)
        sig_prior_arr = np.asarray(sig_prior_list)

        k_opt = cfg["k"]
        if k_opt is None:
            k_opt = (all_err**2 * all_depth**2).sum() / \
                    (all_depth**4).sum().clip(min=1e-12)

        sig_depth_arr = math.sqrt(k_opt) * all_depth
        sig_comb_arr  = np.maximum(sig_prior_arr, sig_depth_arr) \
                        if cfg["use_combined"] else sig_prior_arr

        # ── sensitivity plot ──
        ax = axs[r, 0]
        ord_err = np.sort(all_err)
        ax.plot(np.arange(1,len(ord_err)+1)/len(ord_err)*100,
                np.sqrt(np.cumsum(ord_err**2) /
                        np.arange(1,len(ord_err)+1)), 'k', label="Upper Bound")
        rc, rm = make_sensitivity(all_err, sig_prior_arr)
        ax.plot(rc, rm, 'C0', label="Prior")
        if k_opt > 0:
            rc, rm = make_sensitivity(all_err, sig_depth_arr)
            ax.plot(rc, rm, 'C1', label="Depth")
        if cfg["use_combined"]:
            rc, rm = make_sensitivity(all_err, sig_comb_arr)
            ax.plot(rc, rm, 'C2', label="Comb")
        ax.invert_yaxis(); ax.set_xlim(0,100)
        ax.set_xlabel("Recall (%)"); ax.set_ylabel("RMSE (m)")
        ax.set_title(f"{mname} – Sensitivity (k={k_opt:.2e})")
        ax.legend()

        # ── calibration plot ──
        σ_sel = sig_comb_arr if cfg["use_combined"] else sig_prior_arr
        α = (σ_sel * all_err).sum() / (σ_sel**2).sum()
        σ = σ_sel * α

        vmax = np.nanpercentile(σ, 99.5)
        bins = np.linspace(0, vmax, 41)
        idx  = np.clip(np.digitize(σ, bins)-1, 0, len(bins)-2)

        rmse_bin = [np.sqrt((all_err[idx==b]**2).mean()) if (idx==b).any()
                    else np.nan for b in range(len(bins)-1)]
        freq_bin = np.bincount(idx, minlength=len(bins)-1) / len(all_err) * 100
        centers  = 0.5 * (bins[:-1] + bins[1:])

        axc = axs[r, 1]
        lab = "Comb" if cfg["use_combined"] else "Prior"
        axc.plot(centers, rmse_bin, lw=2,
                 color="C2" if cfg["use_combined"] else "C0", label=lab)
        axc.plot(centers, centers, '--k'); axc.set_ylim(0, 2.2)
        axc.set_xlabel("Depth StdDev (m)"); axc.set_ylabel("RMSE (m)")
        axc.legend(loc="upper left")
        ax2 = axc.twinx()
        ax2.bar(centers, freq_bin, width=0.04, color='grey', alpha=.25)
        ax2.set_ylabel("Bin Size (%)")
        axc.set_title(f"{mname} – Uncertainty Calibration")

    plt.savefig("figure7_updatescalemeter.png", dpi=300)
    print("✓ Saved → figure7_reproduced.png")

# ───────────────────────── entry ─────────────────────────── #
if __name__ == "__main__":
    main()
