import numpy as np, imageio.v2 as iio
from pathlib import Path


root = Path("/mnt/d/work/mpsfm/ETH3D")
rgb_path   = root/"courtyard_dslr_jpg/courtyard/images/dslr_images/DSC_0286.JPG"  # 彩色
depth_raw  = root/"courtyard_dslr_depth/courtyard/ground_truth_depth/dslr_images/DSC_0286.JPG"  # float32 深度

# 1) 先用普通方式获取分辨率
H, W = iio.imread(rgb_path).shape[:2]     # 4032 × 6048

# 2) 直接把“JPG”当二进制读进来
with open(depth_raw, "rb") as f:
    depth = np.frombuffer(f.read(), dtype="<f4").reshape(H, W)

# 3) 深度单位 = 米
mask_valid = np.isfinite(depth) & (depth>0)

print(depth[mask_valid].min(), depth[mask_valid].max())     # the result: 1.7111907 6.7706184
print(depth[2000,3000])     # the result: 4.856572

