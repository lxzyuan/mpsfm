import pycolmap
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

def read_intrinsics(rec_dir: str | Path,
                    as_matrix: bool = False
                   ) -> Dict[str, Tuple[np.ndarray, Tuple[int, int]]]:
    """
    Load per-image intrinsics from a COLMAP/pycolmap reconstruction.

    Parameters
    ----------
    rec_dir : str or Path
        Directory that contains the COLMAP binary reconstruction
        (e.g. the “rec” folder written by the previous script).
    as_matrix : bool, default True
        If True  → return the 3×3 calibration matrix **K**.  
        If False → return the raw parameter vector stored by COLMAP
        (fx, fy, cx, cy, … depending on the camera model).

    Returns
    -------
    intrinsics : dict
        {image_name : (K_or_params, (width, height))}
        `K_or_params` is either the 3×3 numpy array **K** or
        the 1-D parameter vector; the tuple gives the image size.
    """
    rec  = pycolmap.Reconstruction(rec_dir)
    intr = {}

    for img in rec.images.values():
        cam = rec.cameras[img.camera_id]

        if as_matrix:
            # pycolmap provides the 3×3 intrinsic matrix directly
            K = cam.calibration_matrix()
            intr[img.name] = (K, (cam.width, cam.height))
        else:
            intr[img.name] = (np.array(cam.params, copy=True),
                              (cam.width, cam.height))

    return intr


# intr = read_intrinsics("/home/cedar/gitcode/mpsfm/local/benchmarks/eth3d/data/courtyard/rec")      # 默认返回 K
# K, (w, h) = intr["DSC_0286.png"]
# print("Calibration matrix:\n", K)


rec_dir = Path("/home/cedar/gitcode/mpsfm/local/benchmarks/eth3d/data/courtyard/rec")         # 处理脚本写出的新重建目录
rec = pycolmap.Reconstruction(rec_dir)      # 载入

for img in rec.images.values():
    cfw = img.cam_from_world               # Rigid3d 对象
    R = cfw.rotation.matrix()              # 3×3 旋转矩阵
    t = cfw.translation                    # 长度 3 的平移向量
    T = cfw.matrix()                       # 3×4 [R|t]（同上）
    # 如有需要再拼成 4×4 齐次矩阵
    # print(img.name, T)
    print("#######", img.name, T.shape)


for img_id, image in rec.images.items():
    print("image", img_id, image.name)
