# Standalone Uncertainty Calculators for Depth and Normals

This document provides Python classes and helper functions for calculating depth variance and normal vector covariance, based on the logic found in the MP-SfM project. These are designed to be more standalone for easier integration into other projects.

## I. Depth Uncertainty Calculator

The `DepthUncertaintyCalculator` class computes a pixel-wise scalar variance map for a given depth map. It can incorporate model-provided variances, variance derived from flip-consistency (comparing a depth map with one from a flipped image), and terms proportional to depth values.

### Code: `DepthUncertaintyCalculator`

[The Python code for `DepthUncertaintyCalculator` can be found in the `uncertainty_calculators.py` file.]

### Usage Notes for `DepthUncertaintyCalculator`

*   **Initialization**: Instantiate with a `config_dict` defining parameters like `use_model_prior_uncertainty`, `use_flip_consistency`, `proportional_depth_scalar`, noise levels, and multipliers.
*   **`calculate_variance` Method**:
    *   **Inputs**: `base_depth_map`, optional `base_model_variance_map`, `flipped_depth_map`, `flipped_model_variance_map`, and a `combined_validity_mask` (pre-calculated from model output, continuity, sky etc.).
    *   **Outputs**: A dictionary with `"fused_depth_map"`, `"variance_map"`, and `"final_validity_mask"`.
*   The internal logic fuses depth maps, then calculates and combines variance components according to the configuration before applying final constraints and masking.

## II. Normal Uncertainty Calculator and Helper Functions

The `NormalUncertaintyCalculator` class computes a pixel-wise 3x3 Cartesian covariance matrix for surface normals. This often involves transformations to/from spherical coordinates and Jacobian calculations.

### Helper Functions for Normal Uncertainty

[The Python code for these helper functions can be found in the `uncertainty_calculators.py` file.]

### Code: `NormalUncertaintyCalculator`

[The Python code for `NormalUncertaintyCalculator` can be found in the `uncertainty_calculators.py` file. Please refer to `uncertainty_calculators.py` for the complete and up-to-date implementation of all calculators and helpers. The `usage_example.py` script demonstrates how to use these classes.]

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
