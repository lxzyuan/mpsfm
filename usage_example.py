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

# --- Data Loading Helper Function ---

def load_raw_priors_data(extraction_h5_path: Path, image_name: str) -> dict:
    """
    Loads raw extraction data (depth, normals, model variances, flipped versions, validity)
    for an image from an HDF5 file (e.g., metric3dv2.h5 from custom_extraction_output/).
    Assumes image_name is a direct key in this HDF5 file.
    """
    data = {}
    if not extraction_h5_path.exists():
        print(f"Error: Raw priors file not found: {extraction_h5_path}")
        return data

    with h5py.File(extraction_h5_path, 'r') as f:
        if image_name in f:
            group = f[image_name]
            print(f"Available keys for image '{image_name}' in '{extraction_h5_path}': {list(group.keys())}")
            for key in group.keys():
                data[key] = group[key][()]
        else:
            print(f"Error: Data for '{image_name}' not found in {extraction_h5_path}")
            print(f"Available image groups in HDF5: {list(f.keys())[:20]}{'...' if len(list(f.keys())) > 20 else ''}")
    return data

# --- Main Example ---
if __name__ == "__main__":
    mpsfm_root = Path(".").resolve() # Assumes script is run from MP-SfM root

    # --- User Configuration for Data ---
    # This example now assumes you have first run `run_custom_extraction.py`
    # with `extract_depth_metric3d.yaml` (which has model.write_name: metric3dv2).
    # The output of that script is the input for this one.

    # Directory where `run_custom_extraction.py` saves its output
    custom_extraction_dir = mpsfm_root / "custom_extraction_output"
    # The HDF5 file containing raw priors (depth, normals, variances, flipped versions)
    # Name comes from `model.write_name` in `extract_depth_metric3d.yaml`
    raw_priors_h5_file = custom_extraction_dir / "metric3dv2.h5"

    # Image filename (used as key in the HDF5 file)
    # This should be one of the images processed by `run_custom_extraction.py`
    # (e.g., from `local/example/images/`)
    image_filename_key = "indoor_DSC02865.JPG"

    print(f"--- Using Data Spec ---")
    print(f"Raw Priors HDF5 (source for depth & normals): {raw_priors_h5_file}")
    print(f"Target Image Filename Key: {image_filename_key}")
    print(f"-------------------------\n")

    # --- Load All Raw Data ---
    raw_data = load_raw_priors_data(raw_priors_h5_file, image_filename_key)

    if not raw_data:
        print(f"Stopping: Could not load any data for '{image_filename_key}' from {raw_priors_h5_file}.")
        exit(1)

    # Check for essential data components
    has_base_depth = 'depth' in raw_data
    has_base_normals = 'normals' in raw_data

    run_depth_example = has_base_depth
    run_normal_example = has_base_normals

    # --- Example Configurations (defined in script) ---
    # Now we can enable more features of the calculators
    depth_calc_config = {
        'use_model_prior_uncertainty': True,
        'use_flip_consistency': True,
        'proportional_depth_scalar': 0.03,
        'model_variance_multiplier': 1.0, # Raw model variance is used directly
        'flip_consistency_variance_multiplier': 1.0, # Variance from (d1-d2)^2 also scaled by this
        'inherent_noise_std': 0.01,
        'max_allowed_std': 0.75,
        'final_variance_multiplier': 1.0
    }
    print(f"Depth Calculator Config: {depth_calc_config}")

    normal_calc_config = {
        'use_flip_consistency': True,
        'inherent_polar_noise_std': np.deg2rad(1.5), # 1.5 degrees
        'model_variance_multiplier': 1.0,
        'flip_consistency_variance_multiplier': 1.0,
        'final_covariance_multiplier': 1.0
    }
    print(f"Normal Calculator Config: {normal_calc_config}\n")

    # --- Create Combined Validity Mask ---
    # Uses 'valid' and 'valid2' (if flip consistency) from the raw extraction data
    combined_valid_mask = None
    if run_depth_example: # Base validity on depth data primarily
        if 'valid' in raw_data:
            combined_valid_mask = raw_data['valid'].astype(bool)
            if depth_calc_config['use_flip_consistency'] and 'valid2' in raw_data:
                combined_valid_mask &= raw_data['valid2'].astype(bool)
        if combined_valid_mask is not None:
             combined_valid_mask &= (raw_data['depth'] > 1e-3) # Depth must be positive
             if depth_calc_config['use_flip_consistency'] and raw_data.get('depth2') is not None:
                  combined_valid_mask &= (raw_data['depth2'] > 1e-3)
        print(f"Combined validity mask created with shape: {combined_valid_mask.shape if combined_valid_mask is not None else 'N/A'}")


    # --- Instantiate and Use DepthUncertaintyCalculator ---
    if run_depth_example:
        print("--- Running DepthUncertaintyCalculator ---")
        depth_calculator = DepthUncertaintyCalculator(depth_calc_config)

        # Inputs from the raw_data dictionary
        base_depth = raw_data['depth']
        base_model_var = raw_data.get('depth_variance')
        flipped_depth = raw_data.get('depth2')
        flipped_model_var = raw_data.get('depth_variance2')

        print(f"  Input base_depth_map shape: {base_depth.shape}")
        if base_model_var is not None:
             print(f"  Input base_model_variance_map shape: {base_model_var.shape}")
        if flipped_depth is not None and depth_calc_config['use_flip_consistency']:
             print(f"  Input flipped_depth_map shape: {flipped_depth.shape}")
        if flipped_model_var is not None and depth_calc_config['use_flip_consistency']:
             print(f"  Input flipped_model_variance_map shape: {flipped_model_var.shape}")

        if depth_calc_config['use_model_prior_uncertainty'] and base_model_var is None:
            print("  Warning: Depth config 'use_model_prior_uncertainty' is True, but 'depth_variance' not found in raw data. Disabling.")
            depth_calculator.config['use_model_prior_uncertainty'] = False
        if depth_calc_config['use_flip_consistency'] and flipped_depth is None:
            print("  Warning: Depth config 'use_flip_consistency' is True, but 'depth2' not found in raw data. Disabling.")
            depth_calculator.config['use_flip_consistency'] = False

        depth_results = depth_calculator.calculate_variance(
            base_depth_map=base_depth.copy(),
            base_model_variance_map=base_model_var.copy() if base_model_var is not None else None,
            flipped_depth_map=flipped_depth.copy() if flipped_depth is not None else None,
            flipped_model_variance_map=flipped_model_var.copy() if flipped_model_var is not None else None,
            combined_validity_mask=combined_valid_mask.copy() if combined_valid_mask is not None else None
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

        base_normals = raw_data['normals']
        # 'normals_variance' from Metric3Dv2 output is already kappa_to_alpha(conf)**2
        base_model_scalar_var = raw_data.get('normals_variance')

        flipped_normals = raw_data.get('normals2')
        flipped_model_scalar_var = raw_data.get('normals2_variance')

        print(f"  Input base_normal_map shape: {base_normals.shape}")
        if base_model_scalar_var is not None:
            print(f"  Input base_model_scalar_variance_map shape: {base_model_scalar_var.shape}")
        if flipped_normals is not None and normal_calc_config['use_flip_consistency']:
            print(f"  Input flipped_normal_map shape: {flipped_normals.shape}")
        if flipped_model_scalar_var is not None and normal_calc_config['use_flip_consistency']:
            print(f"  Input flipped_model_scalar_variance_map shape: {flipped_model_scalar_var.shape}")

        if normal_calc_config['use_flip_consistency'] and flipped_normals is None:
            print("  Warning: Normal config 'use_flip_consistency' is True, but 'normals2' not found. Disabling.")
            normal_calculator.config['use_flip_consistency'] = False

        if not normal_calculator.config['use_flip_consistency'] and base_model_scalar_var is None:
             print(f"  Error: Normal calculation needs 'base_model_scalar_variance_map' if not using flip consistency. Not found.")
        else:
            # Use the validity mask derived from depth, if available and shapes match
            normal_valid_mask_for_calc = None
            if combined_valid_mask is not None:
                if combined_valid_mask.shape == base_normals.shape[:2]:
                    normal_valid_mask_for_calc = combined_valid_mask.copy()
                    print(f"  Using combined validity mask (from depth) for normals.")
                else:
                    print(f"  Warning: Combined validity mask shape {combined_valid_mask.shape} differs from normal map shape {base_normals.shape[:2]}. Not using mask for normals.")

            normal_results = normal_calculator.calculate_covariance(
                base_normal_map=base_normals.copy(),
                base_model_scalar_variance_map=base_model_scalar_var.copy() if base_model_scalar_var is not None else None,
                flipped_normal_map=flipped_normals.copy() if flipped_normals is not None and normal_calculator.config['use_flip_consistency'] else None,
                flipped_model_scalar_variance_map=flipped_model_scalar_var.copy() if flipped_model_scalar_var is not None and normal_calculator.config['use_flip_consistency'] else None,
                combined_validity_mask=normal_valid_mask_for_calc
            )
            print(f"  Output 'fused_normal_map' shape: {normal_results['fused_normal_map'].shape}")
            print(f"  Output 'covariance_map' shape: {normal_results['covariance_map'].shape}")

            # Determine a valid point to sample covariance from
            final_normal_valid_mask = np.ones(base_normals.shape[:2], dtype=bool)
            if normal_valid_mask_for_calc is not None:
                final_normal_valid_mask = normal_valid_mask_for_calc

            # Additionally, ensure the fused normal itself is valid (not default from masking)
            check_fused_normals = normal_results['fused_normal_map']
            non_default_normal_mask = (np.abs(check_fused_normals[...,0]) > 1e-5) | \
                                      (np.abs(check_fused_normals[...,1]) > 1e-5) | \
                                      (np.abs(check_fused_normals[...,2] - 1.0) > 1e-5) # Check if not [0,0,1]
            final_normal_valid_mask &= non_default_normal_mask


            if np.any(final_normal_valid_mask):
                sample_y, sample_x = np.argwhere(final_normal_valid_mask)[0]
                print(f"  Normal Covariance (sample at {sample_y},{sample_x}):\n{normal_results['covariance_map'][sample_y, sample_x]}")
            else:
                print("  No valid pixels (after considering default normals) to sample normal covariance.")

    print("\n--- Example Finished ---")

