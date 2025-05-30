import numpy as np
from typing import Optional, Dict, List, Tuple

# Helper Functions for Normal Uncertainty
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

def cart_to_spherical_mean_detailed(N1_cart: np.ndarray, N2_cart: np.ndarray)         -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            if not (self._get_config_param('use_model_prior_uncertainty', True) and                     scaled_base_model_var is not None and scaled_flipped_model_var is not None):
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
            spherical_cov_map_2x2, mean_normals_for_jacobian_spherical, fused_normals_cartesian =                 calculate_two_view_spherical_covariance_and_mean(
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
            scaled_scalar_variance = base_model_scalar_variance_map *                                      self._get_config_param('model_variance_multiplier', 1.0)
            spherical_cov_map_2x2 = scalar_variance_to_spherical_covariance(
                scaled_scalar_variance,
                self._get_config_param('inherent_polar_noise_std', 0.017)
            )

        jacobian_map_hw3x2 = spherical_to_cartesian_jacobian(mean_normals_for_jacobian_spherical)
        # Cov_cartesian = J @ Cov_sphere @ J.T
        # J is (..., 3, 2), Cov_sphere is (..., 2, 2)
        # temp should be (..., 3, 2)
        temp_mult_hw3x2 = jacobian_map_hw3x2 @ spherical_cov_map_2x2

        # temp is (..., 3, 2), J.T is (..., 2, 3)
        # final result should be (..., 3, 3)
        cartesian_covariance_map_3x3 = temp_mult_hw3x2 @ np.swapaxes(jacobian_map_hw3x2, -1, -2)

        final_cartesian_covariance_map = cartesian_covariance_map_3x3 *                                          (self._get_config_param('final_covariance_multiplier', 1.0)**2)

        if combined_validity_mask is not None:
            identity_large = np.eye(3) * 1e6
            final_cartesian_covariance_map[~combined_validity_mask] = identity_large
            # Ensure broadcasting for assignment if mask is HxW and normals HxWx3
            if fused_normals_cartesian.ndim > combined_validity_mask.ndim:
                 fused_normals_cartesian[~combined_validity_mask, :] = np.array([0.0,0.0,1.0])
            elif fused_normals_cartesian.ndim == combined_validity_mask.ndim:
                 fused_normals_cartesian[~combined_validity_mask] = np.array([0.0,0.0,1.0])


        return {
            "fused_normal_map": fused_normals_cartesian,
            "covariance_map": final_cartesian_covariance_map
        }

# --- How to Use These Calculators ---
"""
The classes `DepthUncertaintyCalculator` and `NormalUncertaintyCalculator`
are designed to be used independently of the main MP-SfM pipeline, provided
you have the necessary input data (depth maps, normal maps, and potentially
raw model-derived variances or confidences) as NumPy arrays.

General Workflow:

1.  Prepare Input Data:
    *   Depth Data:
        *   `base_depth_map`: Your primary depth map (HxW NumPy array).
        *   `base_model_variance_map` (Optional): Pixel-wise scalar variance associated
          with `base_depth_map`. This might come from your depth estimation model's
          confidence, converted to variance.
        *   `flipped_depth_map` (Optional): A second depth map, e.g., from a
          horizontally flipped version of the input image, registered to the
          `base_depth_map`'s viewpoint.
        *   `flipped_model_variance_map` (Optional): Variance for `flipped_depth_map`.
    *   Normal Data:
        *   `base_normal_map`: Your primary normal map (HxWx3 NumPy array, normalized).
        *   `base_model_scalar_variance_map` (Optional): Pixel-wise scalar variance
          associated with `base_normal_map` (e.g., derived from a model's kappa
          value or concentration parameter).
        *   `flipped_normal_map` (Optional): A second normal map from a flipped image,
          geometrically aligned with `base_normal_map`.
        *   `flipped_model_scalar_variance_map` (Optional): Scalar variance for
          `flipped_normal_map`.
    *   `combined_validity_mask` (Optional): An HxW boolean NumPy array where True
      indicates a pixel where the prior is considered valid. This can be derived
      from depth > 0, continuity checks, sky masks, etc.

2.  Define Configuration Dictionaries:
    *   Create a `config_dict` for `DepthUncertaintyCalculator` and another for
      `NormalUncertaintyCalculator`. These dictionaries hold parameters that
      control how uncertainty is calculated (see the __init__ docstrings of each
      class for available parameters and their typical effects).
      Example:
      ```python
      depth_config = {
          'use_model_prior_uncertainty': True,
          'use_flip_consistency': True,
          'proportional_depth_scalar': 0.03, # 3% of depth value
          'inherent_noise_std': 0.01, # 1cm
          # ... other params
      }
      normal_config = {
          'use_flip_consistency': True,
          'inherent_polar_noise_std': np.deg2rad(1.5), # 1.5 degrees
          # ... other params
      }
      ```

3.  Instantiate and Use Calculators:
    *   `depth_calc = DepthUncertaintyCalculator(depth_config)`
    *   `depth_results = depth_calc.calculate_variance(...)`
    *
    *   `normal_calc = NormalUncertaintyCalculator(normal_config)`
    *   `normal_results = normal_calc.calculate_covariance(...)`

4.  Interpret Results:
    *   `depth_results` will be a dictionary:
        {
            "fused_depth_map": np.ndarray, # The processed depth map
            "variance_map": np.ndarray,    # Pixel-wise scalar variance
            "final_validity_mask": np.ndarray # Boolean validity mask
        }
    *   `normal_results` will be a dictionary:
        {
            "fused_normal_map": np.ndarray, # The processed normal map
            "covariance_map": np.ndarray   # Pixel-wise 3x3 Cartesian covariance
        }

For a runnable demonstration, please see `usage_example.py` which shows how
to load data from HDF5 files (as produced by MP-SfM's extraction or processing)
and pass it to these calculators. The example also highlights how the choice of
input data (e.g., raw extraction outputs vs. processed data from sfm_outputs)
affects which features of the calculators can be fully utilized.
"""
