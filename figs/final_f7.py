#!/usr/bin/env python3
# fig7_metric3dv2_eth3d.py
# ------------------------------------------------------------
# * Reproduces MP-SfM Figure-7 for Metric3Dv2 on ETH3D-train *
# ------------------------------------------------------------
# 关键特性
# 1. 逐张图做 “scale-only” 线性对齐（无偏置）
# 2. σ_prior = √var × SIG_SCALE，σ_depth = α·depth
# 3. σ_comb = max(σ_prior, σ_depth) ，再自动 γ 缩放使 β≈0.34
# 4. 敏感度曲线按 σ 排序；校准采用等频(quantile)分箱
# 5. 生成论文同版式双子图（左 Sensitivity，右 Calibration）
# ------------------------------------------------------------

import numpy as np, h5py, cv2, matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# ---------- 路径 & 场景 ----------
CACHE_DIR = Path("/mnt/data/0/cedar/datasets/mpsfm_local/benchmarks/eth3d/cache_dir")
SCENES = ["courtyard","delivery_area","electro","facade","kicker","meadow",
          "office","pipes","playground","relief","relief_2","terrace","terrains"]

# ---------- 不确定性参数 ----------
ALPHA_DEPTH = 0.20     # σ_depth = α·depth
SIG_SCALE   = 0.10     # σ_prior = √var × scale
SIG_MIN     = 0.005    # 5 mm 下限（裁剪避免 0）
TARGET_BETA = 0.34     # β 目标 (err–σ 线性标定)
NBINS       = 20       # 校准等频分箱
SAVE_FIG    = "fig7_metric3dv2_eth3d.png"

# ---------- util ----------
def resize(arr, hw, interp=cv2.INTER_LINEAR):
    return arr if arr.shape == hw else cv2.resize(
        arr.astype(np.float32), (hw[1], hw[0]), interpolation=interp)

def rmse_cumulative(err, key):
    idx  = np.argsort(key)
    rms  = np.sqrt(np.cumsum(err[idx]**2) / np.arange(1, len(err)+1))
    rec  = np.linspace(0, 100, len(err))
    return rec, rms

# ---------- accumulate 全像素 ----------
err_all, depth_raw_all, var_all = [], [], []

for sc in tqdm(SCENES, desc="load"):
    h5f = CACHE_DIR/sc/"metric3dv2.h5"
    gt_dir = CACHE_DIR/sc/"gt_depth"
    with h5py.File(h5f, "r") as f:
        for k in f.keys():
            gt_path = gt_dir/Path(k).with_suffix(".npy")
            if not gt_path.exists():
                continue

            gt   = np.load(gt_path).astype(np.float32).squeeze()
            pred = resize(f[k]["depth"][()].astype(np.float32), gt.shape)
            var  = resize(f[k]["depth_variance"][()].astype(np.float32), gt.shape)

            mask = np.isfinite(gt) & (gt > 0) & np.isfinite(pred)
            if not mask.any(): continue

            # --- scale-only 对齐 ---
            s = np.median(gt[mask] / pred[mask])
            err_all   .append(np.abs(pred[mask]*s - gt[mask]))
            depth_raw_all.append(pred[mask])   # 用 "未对齐" 深度计算 σ_depth
            var_all   .append(var[mask])

err_all        = np.concatenate(err_all).astype(np.float32)
depth_raw_all  = np.concatenate(depth_raw_all).astype(np.float32)
var_all        = np.concatenate(var_all).astype(np.float32)

# ---------- 不确定性 ----------
sigma_prior = np.clip(np.sqrt(var_all) * SIG_SCALE, SIG_MIN, None)
sigma_depth = np.clip(ALPHA_DEPTH * depth_raw_all, SIG_MIN, None)
sigma_comb_raw = np.maximum(sigma_prior, sigma_depth)

# ---------- γ 缩放使 β ≈ TARGET_BETA ----------
beta_raw = (sigma_comb_raw * err_all).sum() / (sigma_comb_raw**2).sum()
gamma = beta_raw / TARGET_BETA
sigma_prior *= gamma
sigma_depth *= gamma
sigma_comb_raw *= gamma
beta = (sigma_comb_raw * err_all).sum() / (sigma_comb_raw**2).sum()
print(f"β raw={beta_raw:.3f}  γ={gamma:.3f}  β new={beta:.3f}")

# ---------- 敏感度曲线 ----------
rec_u, rmse_upper = rmse_cumulative(err_all, err_all)            # Oracle
rec_p, rmse_prior = rmse_cumulative(err_all, sigma_prior)
rec_d, rmse_depth = rmse_cumulative(err_all, sigma_depth)
rec_c, rmse_comb  = rmse_cumulative(err_all, sigma_comb_raw)

# ---------- 校准: 等频分箱 ----------
edges = np.quantile(sigma_comb_raw, np.linspace(0,1,NBINS+1))
edges[0], edges[-1] = -np.inf, np.inf
idx = np.clip(np.digitize(sigma_comb_raw, edges)-1, 0, NBINS-1)

centers, rmse_bin, counts = [], [], []
for b in range(NBINS):
    m = idx == b
    centers.append(sigma_comb_raw[m].mean())
    rmse_bin.append(np.sqrt((err_all[m]**2).mean()))
    counts.append(m.sum())
centers  = np.array(centers)
rmse_bin = np.array(rmse_bin)
bin_frac = np.array(counts) / len(err_all) * 100


# ---------- 绘图 ----------
fig,(axL,axR) = plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)

# 左：敏感度
axL.plot(rec_u, rmse_upper,'k',lw=2, label="Upper Bound")
axL.plot(rec_p, rmse_prior,'C0',lw=2,label="Prior")
axL.plot(rec_d, rmse_depth,'C1',lw=2,label="Depth")
axL.plot(rec_c, rmse_comb ,'C2',lw=2,label="Combined")
axL.set_xlim(0,100); axL.set_ylim(rmse_upper.max()*1.05,0)
axL.set_xlabel("Recall (%)"); axL.set_ylabel("RMSE (m)")
axL.set_title("Metric3Dv2 Depth Sensitivity Analysis")
axL.grid(alpha=.3); axL.legend()

# 右：校准
axR.plot(centers, rmse_bin,'C2',lw=2,label="Combined")
axR.plot([0,0.4],[0,0.4],'k--',label="y=x")
axR.set_xlim(0,0.4); axR.set_ylim(0,0.4)
axR.set_xlabel("Depth StdDev (m)"); axR.set_ylabel("RMSE (m)")
axR.set_title("Metric3Dv2 Depth Uncertainty Calibration")
axR.grid(alpha=.3); axR.legend(loc="upper left")

# 直方图
widths = np.diff(edges)
widths[0] = widths[1]         # 替换 inf
widths[-1]= widths[-2]
axH = axR.twinx()
axH.bar(centers, bin_frac, width=widths, align="center",
        color='C2', alpha=.25, edgecolor='none')
axH.set_ylabel("Bin Size (%)")
axH.set_ylim(0, bin_frac.max()*1.2)

plt.savefig(SAVE_FIG, dpi=300)
print("✓ saved", SAVE_FIG)
