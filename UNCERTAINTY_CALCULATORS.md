# Standalone Uncertainty Calculators for Depth and Normals

This document provides Python classes and helper functions for calculating depth variance and normal vector covariance, based on the logic found in the MP-SfM project. These are designed to be more standalone for easier integration into other projects.

## I. Depth Uncertainty Calculator

The `DepthUncertaintyCalculator` class computes a pixel-wise scalar variance map for a given depth map. It can incorporate model-provided variances, variance derived from flip-consistency (comparing a depth map with one from a flipped image), and terms proportional to depth values.

### Code: `DepthUncertaintyCalculator`

```python
import numpy as np
from typing import Optional, Dict, List, Tuple

class DepthUncertaintyCalculator:
    """
    Calculates pixel-wise scalar variance for a depth map based on various inputs 
    and configuration parameters.
    """
    def __init__(self, config_dict: Dict):
        """
        Initializes the calculator with configuration parameters.

        Args:
            config_dict (dict): Configuration parameters such as:
                'use_model_prior_uncertainty': bool (default: True) - Use variance from model.
                'use_flip_consistency': bool (default: False) - Use variance from depth_map vs flipped_depth_map.
                'proportional_depth_scalar': float or None (default: None) - Factor for (depth * factor)^2 variance.
                'use_fixed_uncertainty': bool (default: False) - Use a fixed variance value.
                'fixed_uncertainty_value': float (default: 0.03) - Value if use_fixed_uncertainty.
                'model_variance_multiplier': float (default: 1.0) - Scales model-provided variances.
                'flip_consistency_variance_multiplier': float (default: 1.0) - Scales variance from d1-d2 difference.
                'inherent_noise_std': float (default: 0.02) - Min std deviation for depth.
                'max_allowed_std': float or None (default: None) - Max std deviation for depth.
                'final_variance_multiplier': float (default: 1.0) - Overall scaler for the final variance.
        """
        self.config = config_dict

    def _get_config_param(self, key: str, default_value):
        return self.config.get(key, default_value)

    def _fuse_depth_maps(self, 
                         base_depth: np.ndarray, 
                         flipped_depth: Optional[np.ndarray],
                         base_variance_for_fusion: Optional[np.ndarray],
                         flipped_variance_for_fusion: Optional[np.ndarray]
                        ) -> np.ndarray:
        """ Fuses base_depth and flipped_depth if flip_consistency is enabled. """
        if self._get_config_param('use_flip_consistency', False) and flipped_depth is not None:
            if base_variance_for_fusion is not None and flipped_variance_for_fusion is not None:
                epsilon = 1e-9
                inv_var1 = 1.0 / (base_variance_for_fusion + epsilon)
                inv_var2 = 1.0 / (flipped_variance_for_fusion + epsilon)
                sum_inv_var = inv_var1 + inv_var2
                weight1 = inv_var1 / (sum_inv_var + epsilon) 
                weight2 = inv_var2 / (sum_inv_var + epsilon)
                return base_depth * weight1 + flipped_depth * weight2
            else:
                return (base_depth + flipped_depth) / 2.0
        return base_depth

    def calculate_variance(self,
                           base_depth_map: np.ndarray,
                           base_model_variance_map: Optional[np.ndarray] = None,
                           flipped_depth_map: Optional[np.ndarray] = None,
                           flipped_model_variance_map: Optional[np.ndarray] = None,
                           combined_validity_mask: Optional[np.ndarray] = None
                           ) -> Dict[str, np.ndarray]:
        """
        Calculates the fused depth map and its pixel-wise variance.
        Args:
            base_depth_map (np.ndarray): Primary HxW depth map.
            base_model_variance_map (Optional[np.ndarray]): HxW variance from model for base_depth_map.
            flipped_depth_map (Optional[np.ndarray]): HxW depth map from flipped image.
            flipped_model_variance_map (Optional[np.ndarray]): HxW variance from model for flipped_depth_map.
            combined_validity_mask (Optional[np.ndarray]): HxW boolean mask. True for valid pixels.
        Returns:
            Dict[str, np.ndarray]: Contains:
                "fused_depth_map": The (potentially fused) HxW depth map.
                "variance_map": The final HxW scalar variance map.
                "final_validity_mask": The HxW boolean validity mask used.
        """
        cfg = self.config
        model_var_mult = self._get_config_param('model_variance_multiplier', 1.0)
        scaled_base_model_var = None
        if base_model_variance_map is not None and self._get_config_param('use_model_prior_uncertainty', True):
            scaled_base_model_var = base_model_variance_map * model_var_mult
            
        scaled_flipped_model_var = None
        if flipped_model_variance_map is not None and self._get_config_param('use_model_prior_uncertainty', True):
            scaled_flipped_model_var = flipped_model_variance_map * model_var_mult

        fused_depth = self._fuse_depth_maps(base_depth_map, flipped_depth_map, 
                                            scaled_base_model_var, scaled_flipped_model_var)
        variance_candidates = []
        if self._get_config_param('use_model_prior_uncertainty', True):
            if self._get_config_param('use_flip_consistency', False) and scaled_base_model_var is not None and scaled_flipped_model_var is not None:
                epsilon = 1e-9
                inv_var_sum = (1.0 / (scaled_base_model_var + epsilon)) + (1.0 / (scaled_flipped_model_var + epsilon))
                variance_candidates.append(1.0 / (inv_var_sum + epsilon))
            elif scaled_base_model_var is not None:
                variance_candidates.append(scaled_base_model_var)
        
        if self._get_config_param('use_flip_consistency', False) and flipped_depth_map is not None:
            var_from_diff = (base_depth_map - flipped_depth_map)**2
            var_from_diff *= self._get_config_param('flip_consistency_variance_multiplier', 1.0)
            if not (self._get_config_param('use_model_prior_uncertainty', True) and \
                    scaled_base_model_var is not None and scaled_flipped_model_var is not None):
                 variance_candidates.append(var_from_diff)


        proportional_scalar = self._get_config_param('proportional_depth_scalar', None)
        var_proportional = None
        if proportional_scalar is not None and proportional_scalar > 0:
            var_proportional = (fused_depth * proportional_scalar)**2
            
        if not variance_candidates:
            if var_proportional is not None:
                 current_variance = var_proportional
            elif self._get_config_param('use_fixed_uncertainty', False):
                current_variance = np.ones_like(fused_depth) * self._get_config_param('fixed_uncertainty_value', 0.03)
            else:
                current_variance = np.ones_like(fused_depth) * (self._get_config_param('inherent_noise_std', 0.02)**2)
        else:
            current_variance = variance_candidates[0]
            if len(variance_candidates) > 1: 
                 current_variance = np.maximum(current_variance, variance_candidates[1]) 

            if var_proportional is not None: 
                current_variance = np.maximum(current_variance, var_proportional)

        min_var = self._get_config_param('inherent_noise_std', 0.02)**2
        max_allowed_std_val = self._get_config_param('max_allowed_std', None)
        max_var_clip = max_allowed_std_val**2 if max_allowed_std_val is not None else np.inf
        actual_min_var = min_var if max_var_clip == np.inf or min_var <= max_var_clip else max_var_clip
        
        constrained_variance = np.clip(current_variance, actual_min_var, max_var_clip)
        final_variance = constrained_variance * (self._get_config_param('final_variance_multiplier', 1.0)**2)
        
        final_valid_mask = combined_validity_mask
        if final_valid_mask is None:
            final_valid_mask = np.ones_like(fused_depth, dtype=bool)
        
        zero_depth_mask = (fused_depth <= 1e-3) 
        final_valid_mask[zero_depth_mask] = False
        final_variance[~final_valid_mask] = 1e6 
        fused_depth[~final_valid_mask] = 0 
            
        return {
            "fused_depth_map": fused_depth,
            "variance_map": final_variance,
            "final_validity_mask": final_valid_mask
        }
```

### Usage Notes for `DepthUncertaintyCalculator`

*   **Initialization**: Instantiate with a `config_dict` defining parameters like `use_model_prior_uncertainty`, `use_flip_consistency`, `proportional_depth_scalar`, noise levels, and multipliers.
*   **`calculate_variance` Method**:
    *   **Inputs**: `base_depth_map`, optional `base_model_variance_map`, `flipped_depth_map`, `flipped_model_variance_map`, and a `combined_validity_mask` (pre-calculated from model output, continuity, sky etc.).
    *   **Outputs**: A dictionary with `"fused_depth_map"`, `"variance_map"`, and `"final_validity_mask"`.
*   The internal logic fuses depth maps, then calculates and combines variance components according to the configuration before applying final constraints and masking.

## II. Normal Uncertainty Calculator and Helper Functions

The `NormalUncertaintyCalculator` class computes a pixel-wise 3x3 Cartesian covariance matrix for surface normals. This often involves transformations to/from spherical coordinates and Jacobian calculations.

### Helper Functions for Normal Uncertainty

```python
import numpy as np
from typing import Optional, Tuple

EPSILON = 1e-9 # For numerical stability

def cart_to_spherical(cartesian_normals: np.ndarray) -> np.ndarray:
    """
    Converts a batch of 3D Cartesian normal vectors to spherical coordinates (theta, phi).
    Theta: polar angle (from Z-axis, range [0, pi]).
    Phi: azimuthal angle (in XY plane from X-axis, range [-pi, pi]).
    Args: cartesian_normals: NumPy array of shape (..., 3). Input vectors will be normalized.
    Returns: NumPy array of shape (..., 2) containing (theta, phi) in radians.
    """
    if cartesian_normals.shape[-1] != 3:
        raise ValueError("Last dimension of cartesian_normals must be 3.")
    norm_val = np.linalg.norm(cartesian_normals, axis=-1, keepdims=True)
    normalized_normals = cartesian_normals / (norm_val + EPSILON)
    theta = np.arccos(np.clip(normalized_normals[..., 2], -1.0, 1.0))
    phi = np.arctan2(normalized_normals[..., 1], normalized_normals[..., 0])
    return np.stack([theta, phi], axis=-1)

def spherical_to_cartesian_jacobian(spherical_normals: np.ndarray) -> np.ndarray:
    """
    Computes the Jacobian of the spherical (theta, phi) to Cartesian (x,y,z) transformation (unit sphere).
    Args: spherical_normals: NumPy array of shape (..., 2) with (theta, phi) in radians.
    Returns: Jacobian matrix of shape (..., 3, 2).
    """
    if spherical_normals.shape[-1] != 2:
        raise ValueError("Last dimension of spherical_normals must be 2 (theta, phi).")
    theta = spherical_normals[..., 0]
    phi = spherical_normals[..., 1]
    J_matrix = np.zeros((*theta.shape, 3, 2), dtype=theta.dtype)
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    J_matrix[..., 0, 0] = cos_theta * cos_phi; J_matrix[..., 0, 1] = -sin_theta * sin_phi
    J_matrix[..., 1, 0] = cos_theta * sin_phi; J_matrix[..., 1, 1] = sin_theta * cos_phi
    J_matrix[..., 2, 0] = -sin_theta
    return J_matrix

def diff_angle_rad(angle1_rad: np.ndarray, angle2_rad: np.ndarray) -> np.ndarray:
    """Computes signed minimum difference (angle1 - angle2) in [-pi, pi]."""
    diff = angle1_rad - angle2_rad
    return np.arctan2(np.sin(diff), np.cos(diff))

def cart_to_spherical_mean_detailed(N1_cart: np.ndarray, N2_cart: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Computes mean of two Cartesian normals in spherical space, adjusting angles. """
    N1_sph = cart_to_spherical(N1_cart)
    N2_sph = cart_to_spherical(N2_cart)
    N2_sph_adjusted = N2_sph.copy()
    phi_diff = N2_sph_adjusted[..., 1] - N1_sph[..., 1]
    N2_sph_adjusted[..., 1][phi_diff > np.pi] -= 2 * np.pi
    N2_sph_adjusted[..., 1][phi_diff < -np.pi] += 2 * np.pi
    mean_sph = (N1_sph + N2_sph_adjusted) / 2.0
    mean_sph[..., 1] = np.arctan2(np.sin(mean_sph[..., 1]), np.cos(mean_sph[..., 1]))
    return mean_sph, N1_sph, N2_sph_adjusted

def covar_from_spherical_diffs(mean_normal_spherical: np.ndarray, 
                               N1_spherical: np.ndarray, 
                               N2_spherical_adjusted: np.ndarray) -> np.ndarray:
    """ Computes 2x2 spherical covariance from differences to the mean. """
    diff1 = diff_angle_rad(N1_spherical, mean_normal_spherical)
    diff2 = diff_angle_rad(N2_spherical_adjusted, mean_normal_spherical)
    var_theta = (diff1[..., 0]**2 + diff2[..., 0]**2) / 2.0
    var_phi   = (diff1[..., 1]**2 + diff2[..., 1]**2) / 2.0
    cov_theta_phi = (diff1[..., 0] * diff1[..., 1] + diff2[..., 0] * diff2[..., 1]) / 2.0
    shape_prefix = mean_normal_spherical.shape[:-1]
    cov_sphere = np.zeros((*shape_prefix, 2, 2), dtype=mean_normal_spherical.dtype)
    cov_sphere[..., 0, 0] = var_theta; cov_sphere[..., 1, 1] = var_phi
    cov_sphere[..., 0, 1] = cov_theta_phi; cov_sphere[..., 1, 0] = cov_theta_phi
    return cov_sphere

def scalar_variance_to_spherical_covariance(scalar_variance_map: np.ndarray, 
                                            inherent_polar_noise_std: float) -> np.ndarray:
    """ Converts per-pixel scalar variance to a diagonal 2x2 spherical covariance matrix. """
    shape_prefix = scalar_variance_map.shape
    cov_sphere_diag = np.zeros((*shape_prefix, 2, 2), dtype=scalar_variance_map.dtype)
    noise_var = inherent_polar_noise_std**2
    var_with_noise = np.maximum(scalar_variance_map, noise_var)
    cov_sphere_diag[..., 0, 0] = var_with_noise
    cov_sphere_diag[..., 1, 1] = var_with_noise
    return cov_sphere_diag

def calculate_two_view_spherical_covariance_and_mean(
    N1_cart: np.ndarray, N2_cart: np.ndarray, 
    v1_scalar_map: Optional[np.ndarray], v2_scalar_map: Optional[np.ndarray], 
    inherent_polar_noise_std: float,
    model_variance_multiplier: float,
    flip_consistency_variance_multiplier: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: 
    mean_N_sph, N1_sph, N2_sph_adj = cart_to_spherical_mean_detailed(N1_cart, N2_cart)
    cov_sphere_from_diff = covar_from_spherical_diffs(mean_N_sph, N1_sph, N2_sph_adj)
    cov_sphere_from_diff *= flip_consistency_variance_multiplier
    if v1_scalar_map is not None:
        effective_v_sph = v1_scalar_map * model_variance_multiplier
        cov_sphere_from_diff[..., 0, 0] = np.maximum(cov_sphere_from_diff[..., 0, 0], effective_v_sph)
        cov_sphere_from_diff[..., 1, 1] = np.maximum(cov_sphere_from_diff[..., 1, 1], effective_v_sph)
    if v2_scalar_map is not None: 
        effective_v_sph = v2_scalar_map * model_variance_multiplier
        cov_sphere_from_diff[..., 0, 0] = np.maximum(cov_sphere_from_diff[..., 0, 0], effective_v_sph)
        cov_sphere_from_diff[..., 1, 1] = np.maximum(cov_sphere_from_diff[..., 1, 1], effective_v_sph)
    noise_var = inherent_polar_noise_std**2
    cov_sphere_from_diff[..., 0, 0] = np.maximum(cov_sphere_from_diff[..., 0, 0], noise_var)
    cov_sphere_from_diff[..., 1, 1] = np.maximum(cov_sphere_from_diff[..., 1, 1], noise_var)
    mean_N_cart = (N1_cart + N2_cart) / 2.0
    mean_N_cart /= (np.linalg.norm(mean_N_cart, axis=-1, keepdims=True) + EPSILON)
    return cov_sphere_from_diff, mean_N_sph, mean_N_cart
```

### Code: `NormalUncertaintyCalculator`

```python
import numpy as np
from typing import Optional, Dict, Tuple
# Assuming helper functions from above are defined in the same file or imported

class NormalUncertaintyCalculator:
    """ Calculates pixel-wise 3x3 Cartesian covariance matrix for surface normals. """
    def __init__(self, config_dict: Dict):
        """
        Args:
            config_dict (dict): Parameters like 'use_flip_consistency', 
                                'inherent_polar_noise_std', 'model_variance_multiplier', etc.
        """
        self.config = config_dict

    def _get_config_param(self, key: str, default_value):
        return self.config.get(key, default_value)

    def calculate_covariance(self,
                             base_normal_map: np.ndarray,
                             base_model_scalar_variance_map: Optional[np.ndarray] = None,
                             flipped_normal_map: Optional[np.ndarray] = None,
                             flipped_model_scalar_variance_map: Optional[np.ndarray] = None,
                             combined_validity_mask: Optional[np.ndarray] = None
                             ) -> Dict[str, np.ndarray]:
        """
        Calculates fused normal map and its 3x3 Cartesian covariance matrix.
        Args: (see class docstring for NormalUncertaintyCalculator in previous steps for details)
        Returns: Dict with "fused_normal_map" (HxWx3) and "covariance_map" (HxWx3x3).
        """
        cfg = self.config
        spherical_cov_map_2x2: np.ndarray
        fused_normals_cartesian: np.ndarray
        mean_normals_for_jacobian_spherical: np.ndarray

        if self._get_config_param('use_flip_consistency', False):
            if flipped_normal_map is None:
                raise ValueError("Flipped normal map required for flip consistency.")
            spherical_cov_map_2x2, mean_normals_for_jacobian_spherical, fused_normals_cartesian = \
                calculate_two_view_spherical_covariance_and_mean(
                    base_normal_map, flipped_normal_map,
                    base_model_scalar_variance_map, flipped_model_scalar_variance_map,
                    self._get_config_param('inherent_polar_noise_std', 0.017), # ~1 deg
                    self._get_config_param('model_variance_multiplier', 1.0),
                    self._get_config_param('flip_consistency_variance_multiplier', 1.0)
                )
        else:
            if base_model_scalar_variance_map is None:
                raise ValueError("Base model scalar variance required if not using flip consistency.")
            fused_normals_cartesian = base_normal_map
            mean_normals_for_jacobian_spherical = cart_to_spherical(fused_normals_cartesian)
            scaled_scalar_variance = base_model_scalar_variance_map * \
                                     self._get_config_param('model_variance_multiplier', 1.0)
            spherical_cov_map_2x2 = scalar_variance_to_spherical_covariance(
                scaled_scalar_variance,
                self._get_config_param('inherent_polar_noise_std', 0.017)
            )

        jacobian_map_hw3x2 = spherical_to_cartesian_jacobian(mean_normals_for_jacobian_spherical)
        # Cov_cartesian = J @ Cov_sphere @ J.T
        temp_mult_hw3x2 = np.einsum('...ji,...jk->...ik', jacobian_map_hw3x2, spherical_cov_map_2x2, optimize='optimal')
        cartesian_covariance_map_3x3 = np.einsum('...ji,...kj->...ik', temp_mult_hw3x2, jacobian_map_hw3x2, optimize='optimal')
        
        final_cartesian_covariance_map = cartesian_covariance_map_3x3 * \
                                         (self._get_config_param('final_covariance_multiplier', 1.0)**2)

        if combined_validity_mask is not None:
            identity_large = np.eye(3) * 1e6 
            final_cartesian_covariance_map[~combined_validity_mask] = identity_large
            # Ensure broadcasting for assignment if mask is HxW and normals HxWx3
            if fused_normals_cartesian.ndim > combined_validity_mask.ndim: 
                 fused_normals_cartesian[~combined_validity_mask, :] = np.array([0.0,0.0,1.0]) 
            elif fused_normals_cartesian.ndim == combined_validity_mask.ndim: # Should not happen if shapes are HxWx3 and HxW
                 fused_normals_cartesian[~combined_validity_mask] = np.array([0.0,0.0,1.0])


        return {
            "fused_normal_map": fused_normals_cartesian,
            "covariance_map": final_cartesian_covariance_map
        }
```

### Usage Notes for `NormalUncertaintyCalculator`

*   **Helpers**: Relies on the helper functions defined above for coordinate transformations and spherical covariance calculations.
*   **Initialization**: Takes a `config_dict` with parameters like `use_flip_consistency`, noise levels, and multipliers.
*   **`calculate_covariance` Method**:
    *   **Inputs**: `base_normal_map` and its `base_model_scalar_variance_map`, optional `flipped_normal_map` and its `flipped_model_scalar_variance_map`, and a `combined_validity_mask`. Normal maps should be pre-normalized. Scalar variances are expected (e.g., derived from a model's kappa value).
    *   **Outputs**: A dictionary with `"fused_normal_map"` and `"covariance_map"` (HxWx3x3 Cartesian).
*   The core logic involves:
    1.  Calculating a 2x2 spherical covariance matrix:
        *   If `use_flip_consistency` is true, `calculate_two_view_spherical_covariance_and_mean` is used.
        *   Otherwise, `scalar_variance_to_spherical_covariance` is used with the (scaled) base model scalar variance.
    2.  Transforming this 2x2 spherical covariance to a 3x3 Cartesian covariance using the appropriate Jacobian: `Cov_cart = J @ Cov_sphere @ J.T`.
    3.  Applying a final scaling multiplier.
    4.  Setting covariance for invalid pixels to a large identity matrix.
```

## III. Usage Example

The following script (`usage_example.py`, intended to be placed in the root of the MP-SfM repository) demonstrates how to use the `DepthUncertaintyCalculator` and `NormalUncertaintyCalculator`. It shows how to load data from HDF5 files (as typically produced by MP-SfM's extraction or processing steps) and pass this data to the calculators.

**Important Notes on Data for this Example:**
*   **Depth Data**: The example loads "processed" depth data (`pdepth`) from `sfm_outputs/depths.h5`. This file does *not* contain the raw model variance or flipped depth maps that `DepthUncertaintyCalculator` would ideally use to demonstrate its full capabilities (like fusing raw model variances or deriving variance purely from flip differences). Thus, the depth uncertainty calculation in the example primarily relies on the `proportional_depth_scalar` or `fixed_uncertainty` configuration options. To use the full features of `DepthUncertaintyCalculator`, you would need to load data from the HDF5 files generated during the *initial extraction step* (e.g., `cache_dir/metric3dv2.h5`), which contain fields like `depth`, `depth_variance`, `depth2`, `depth_variance2`.
*   **Normal Data**: The example loads normal data (including raw model scalar variances and data from flipped images like `normals2`) from an HDF5 file generated during the *initial extraction step* (e.g., `cache_dir/metric3dv2.h5`). This allows a more comprehensive demonstration of `NormalUncertaintyCalculator`, including its flip consistency logic.

```python
# usage_example.py

import h5py
import numpy as np
from pathlib import Path

# Assuming uncertainty_calculators.py is in the same directory or PYTHONPATH
try:
    from uncertainty_calculators import DepthUncertaintyCalculator, NormalUncertaintyCalculator
except ImportError:
    print("Error: Could not import calculators. Ensure uncertainty_calculators.py is accessible.")
    print("If running this example from a different directory, you might need to adjust sys.path.")
    exit(1)

# --- Data Loading Helper Functions ---

def load_processed_depth_data(sfm_outputs_depths_h5_path: Path, image_name_for_lookup: str) -> dict:
    """
    Loads 'pdepth' (prior depth) and 'continuity_mask' for an image from 
    the depths.h5 file typically found in sfm_outputs.
    It tries to find a key related to image_name_for_lookup.
    """
    data = {}
    if not sfm_outputs_depths_h5_path.exists():
        print(f"Warning: Processed depths file not found: {sfm_outputs_depths_h5_path}")
        return data
        
    with h5py.File(sfm_outputs_depths_h5_path, 'r') as f:
        found_key = None
        # Try direct match first (e.g. if keys are image names or simple IDs passed as image_name_for_lookup)
        if image_name_for_lookup in f.get('pdepth', {}):
            found_key = image_name_for_lookup
        else: # Fallback: iterate through keys in 'pdepth' to find a match
            for key in f.get('pdepth', {}).keys():
                if image_name_for_lookup in key: # Check if the provided name is part of the HDF5 key
                    found_key = key
                    print(f"Found data for image name containing '{image_name_for_lookup}' under HDF5 key '{key}' for pdepth.")
                    break
        
        if found_key:
            if f['pdepth'][found_key] is not None:
                data['pdepth'] = f['pdepth'][found_key][()]
            
            # Try to find corresponding continuity mask
            continuity_found_key = None
            if 'continuity' in f:
                if found_key in f['continuity']: # If key was an ID that matches
                     continuity_found_key = found_key
                elif image_name_for_lookup in f['continuity']: # Direct name match for continuity
                     continuity_found_key = image_name_for_lookup
                else: # Fallback for continuity
                    for key_c in f.get('continuity', {}).keys():
                        if image_name_for_lookup in key_c:
                            continuity_found_key = key_c
                            print(f"Found continuity data for image name containing '{image_name_for_lookup}' under HDF5 key '{key_c}'.")
                            break
            
            if continuity_found_key and f['continuity'][continuity_found_key] is not None:
                 data['continuity_mask'] = f['continuity'][continuity_found_key][()].astype(bool)
            else:
                print(f"Warning: Continuity mask not found for key related to '{image_name_for_lookup}' in {sfm_outputs_depths_h5_path}")
        else:
            print(f"Warning: No 'pdepth' data found for key related to '{image_name_for_lookup}' in {sfm_outputs_depths_h5_path}")
            available_keys = list(f.get('pdepth', {}).keys())
            if available_keys:
                 print(f"Available keys in 'pdepth': {available_keys[:20]}{'...' if len(available_keys) > 20 else ''}") # Print first 20
            else:
                 print("No 'pdepth' group or keys found.")
    return data

def load_raw_extraction_data(extraction_h5_path: Path, image_name: str) -> dict:
    """
    Loads raw extraction data (like normals, model variances) for an image 
    from an HDF5 file (e.g., metric3dv2.h5 from cache_dir).
    Assumes image_name is a direct key in this HDF5 file.
    """
    data = {}
    if not extraction_h5_path.exists():
        print(f"Warning: Raw extraction file not found: {extraction_h5_path}")
        return data

    with h5py.File(extraction_h5_path, 'r') as f:
        if image_name in f:
            group = f[image_name]
            for key in group.keys():
                data[key] = group[key][()]
        else:
            print(f"Warning: Data for '{image_name}' not found in {extraction_h5_path}")
            print(f"Available image groups: {list(f.keys())[:20]}{'...' if len(list(f.keys())) > 20 else ''}")
    return data

# --- Main Example ---
if __name__ == "__main__":
    mpsfm_root = Path(".").resolve() # Assumes script is run from MP-SfM root
    
    # --- User Configuration for Data ---
    # Option 1: Use data from `local/example` (requires running `reconstruct.py` on it first)
    example_name = "local/example"
    # Image filename (used as key in cache_dir HDF5 files)
    image_filename = "indoor_DSC02865.JPG" 
    # Key for sfm_outputs/depths.h5: This is often the reconstruction image ID (string).
    # You might need to inspect your depths.h5 or know this from an MP-SfM run.
    # For the first image in `local/example` sequence, it's typically '1'.
    image_key_in_processed_depths_h5 = "1" 
    raw_normals_model_write_name = "metric3dv2" # From extraction config, e.g., model.write_name
    
    # Option 2: Define your own paths (modify as needed)
    # example_name = "path/to/your/dataset_folder_containing_cache_and_sfm_outputs"
    # image_filename = "your_image.jpg"
    # image_key_in_processed_depths_h5 = "your_image_id_or_name_in_depths_h5"
    # raw_normals_model_write_name = "name_of_your_normal_extraction_h5_file_without_ext"

    print(f"--- Using Data From Example: {example_name} ---")
    print(f"Target Image Filename (for raw data lookup): {image_filename}")
    print(f"Target Image Key (for processed depth lookup): {image_key_in_processed_depths_h5}")
    
    sfm_outputs_dir = mpsfm_root / example_name / "sfm_outputs"
    processed_depths_h5 = sfm_outputs_dir / "depths.h5"
    
    cache_dir = mpsfm_root / example_name / "cache_dir" 
    raw_normals_source_h5 = cache_dir / f"{raw_normals_model_write_name}.h5"
    
    print(f"Processed depths HDF5 path: {processed_depths_h5}")
    print(f"Raw normals HDF5 path: {raw_normals_source_h5}\n")

    # --- Load Data ---
    depth_data_from_sfm_out = load_processed_depth_data(processed_depths_h5, image_key_in_processed_depths_h5)
    normal_data_raw = load_raw_extraction_data(raw_normals_source_h5, image_filename)

    if not depth_data_from_sfm_out.get('pdepth', None) is not None:
        print(f"Stopping: Could not load 'pdepth' for '{image_key_in_processed_depths_h5}' from {processed_depths_h5}.")
    if not normal_data_raw.get('normals', None) is not None:
        print(f"Stopping: Could not load 'normals' for '{image_filename}' from {raw_normals_source_h5}.")

    # Proceed only if essential data is loaded
    run_depth_example = depth_data_from_sfm_out.get('pdepth', None) is not None
    run_normal_example = normal_data_raw.get('normals', None) is not None


    # --- Example Configurations (defined in script) ---
    depth_calc_config = {
        'use_model_prior_uncertainty': False, 
        'use_flip_consistency': False,    
        'proportional_depth_scalar': 0.03, # Main uncertainty source for this example
        'use_fixed_uncertainty': False,
        'inherent_noise_std': 0.01,
        'max_allowed_std': 0.5, 
        'final_variance_multiplier': 1.0
    }
    print(f"Depth Calculator Config: {depth_calc_config}")

    normal_calc_config = {
        'use_flip_consistency': True, 
        'inherent_polar_noise_std': np.deg2rad(2.0), # 2 degrees
        'model_variance_multiplier': 1.0, 
        'flip_consistency_variance_multiplier': 1.0,
        'final_covariance_multiplier': 1.0
    }
    print(f"Normal Calculator Config: {normal_calc_config}\n")

    # --- Instantiate and Use DepthUncertaintyCalculator ---
    if run_depth_example:
        print("--- Running DepthUncertaintyCalculator ---")
        depth_calculator = DepthUncertaintyCalculator(depth_calc_config)
        
        base_depth = depth_data_from_sfm_out['pdepth']
        
        valid_mask_depth = base_depth > 1e-3 
        if 'continuity_mask' in depth_data_from_sfm_out:
            if depth_data_from_sfm_out['continuity_mask'].shape == valid_mask_depth.shape:
                valid_mask_depth &= depth_data_from_sfm_out['continuity_mask']
            else:
                print(f"Warning: Shape mismatch for continuity mask ({depth_data_from_sfm_out['continuity_mask'].shape}) and depth ({valid_mask_depth.shape}). Not using continuity mask.")

        print(f"  Input base_depth_map (pdepth) shape: {base_depth.shape}")
        print("  Note: Using 'pdepth' from sfm_outputs/depths.h5. This file does not contain raw model variances or flipped depth maps.")
        print("        Thus, 'use_model_prior_uncertainty' and 'use_flip_consistency' are effectively False or limited in this demo.")
        print("        Uncertainty will be primarily driven by 'proportional_depth_scalar' or 'fixed_uncertainty' if configured.")

        depth_results = depth_calculator.calculate_variance(
            base_depth_map=base_depth.copy(), # Pass a copy to avoid in-place modification issues
            base_model_variance_map=None, 
            flipped_depth_map=None,       
            flipped_model_variance_map=None,
            combined_validity_mask=valid_mask_depth.copy()
        )
        print(f"  Output 'fused_depth_map' shape: {depth_results['fused_depth_map'].shape}")
        print(f"  Output 'variance_map' shape: {depth_results['variance_map'].shape}")
        if np.any(depth_results['final_validity_mask']):
            sample_y, sample_x = np.argwhere(depth_results['final_validity_mask'])[0]
            print(f"  Depth Std Dev (sample at {sample_y},{sample_x}): {np.sqrt(depth_results['variance_map'][sample_y, sample_x]):.4f}")
        else:
            print("  No valid pixels in depth output to sample std dev.")
        print(f"  Number of valid pixels in depth output: {np.sum(depth_results['final_validity_mask'])}\n")


    # --- Instantiate and Use NormalUncertaintyCalculator ---
    if run_normal_example:
        print("--- Running NormalUncertaintyCalculator ---")
        normal_calculator = NormalUncertaintyCalculator(normal_calc_config)

        base_normals = normal_data_raw['normals']
        base_model_scalar_var = normal_data_raw.get('normals_variance') 
        
        flipped_normals = normal_data_raw.get('normals2')
        flipped_model_scalar_var = normal_data_raw.get('normals2_variance')

        print(f"  Input base_normal_map shape: {base_normals.shape}")
        if base_model_scalar_var is not None:
            print(f"  Input base_model_scalar_variance_map shape: {base_model_scalar_var.shape}")
        if flipped_normals is not None and normal_calc_config['use_flip_consistency']:
            print(f"  Input flipped_normal_map shape: {flipped_normals.shape}")
        if flipped_model_scalar_var is not None and normal_calc_config['use_flip_consistency']:
            print(f"  Input flipped_model_scalar_variance_map shape: {flipped_model_scalar_var.shape}")


        if normal_calc_config['use_flip_consistency'] and flipped_normals is None:
            print("  Warning: 'use_flip_consistency' is True, but 'normals2' (flipped_normal_map) not found in raw data. Disabling for this run.")
            normal_calculator.config['use_flip_consistency'] = False # Modify local instance's config
        
        if not normal_calculator.config['use_flip_consistency'] and base_model_scalar_var is None:
             print(f"  Error: Normal calculation needs 'base_model_scalar_variance_map' if not using flip consistency. Not found in data.")
        else:
            normal_valid_mask = None
            if run_depth_example and depth_results['final_validity_mask'].shape == base_normals.shape[:2]:
                normal_valid_mask = depth_results['final_validity_mask'].copy()
                print(f"  Using validity mask from depth output for normals.")
            elif run_depth_example:
                 print(f"  Warning: Depth validity mask shape {depth_results['final_validity_mask'].shape} " +
                       f"differs from normal map shape {base_normals.shape[:2]}. Not using mask for normals.")


            normal_results = normal_calculator.calculate_covariance(
                base_normal_map=base_normals.copy(),
                base_model_scalar_variance_map=base_model_scalar_var.copy() if base_model_scalar_var is not None else None,
                flipped_normal_map=flipped_normals.copy() if flipped_normals is not None and normal_calculator.config['use_flip_consistency'] else None,
                flipped_model_scalar_variance_map=flipped_model_scalar_var.copy() if flipped_model_scalar_var is not None and normal_calculator.config['use_flip_consistency'] else None,
                combined_validity_mask=normal_valid_mask
            )
            print(f"  Output 'fused_normal_map' shape: {normal_results['fused_normal_map'].shape}")
            print(f"  Output 'covariance_map' shape: {normal_results['covariance_map'].shape}")
            
            valid_idx_for_normals = normal_valid_mask if normal_valid_mask is not None else np.ones(base_normals.shape[:2], dtype=bool)
            if np.any(valid_idx_for_normals):
                sample_y, sample_x = np.argwhere(valid_idx_for_normals)[0]
                print(f"  Normal Covariance (sample at {sample_y},{sample_x}):\n{normal_results['covariance_map'][sample_y, sample_x]}")
            else:
                print("  No valid pixels to sample normal covariance.")

    print("\n--- Example Finished ---")
```
