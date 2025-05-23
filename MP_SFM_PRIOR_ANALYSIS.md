# MP-SfM Prior Analysis: Extraction, Data Structures, and Encapsulation Ideas

This document summarizes the key components and logic within the MP-SfM codebase responsible for extracting monocular depth and normal priors, managing their uncertainties, and the data structures involved. It also proposes conceptual strategies for encapsulating these functionalities for easier reuse.

## I. Core Modules and Process Overview

The extraction and handling of monocular priors in MP-SfM involves several key stages:

1.  **Extraction Orchestration**:
    *   **Location**: `mpsfm/extraction/imagewise/geometry/base.py`
    *   **Process**: A `main` function in this file orchestrates the extraction. It iterates through images, loads a specified geometry model (e.g., Metric3Dv2, DSINE) using `mpsfm.extraction.load_model`, prepares input data for the model, calls the model for inference, and then writes the output dictionary to an HDF5 file.
    *   **Model Loading**: `mpsfm.extraction.load_model` (defined in `mpsfm/extraction/__init__.py`) dynamically finds and instantiates the correct model Python class based on a configuration name. These model classes inherit from `mpsfm.extraction.base_model.BaseModel`.
    *   **Base Model Functionality**: `BaseModel` (`mpsfm/extraction/base_model.py`) provides common functionalities like configuration merging (OmegaConf), automatic model weight downloading (via gdown or wget if weights are not found locally), and a standard interface (`_init`, `_forward`) that specific model wrappers implement.

2.  **Specific Monocular Model Wrappers**:
    *   **Location**: e.g., `Metric3Dv2` in `mpsfm/extraction/imagewise/geometry/models/depth/metric3dv2.py`, `DSINE` in `mpsfm/extraction/imagewise/geometry/models/normals/dsine.py`.
    *   **Function**: These classes wrap the actual third-party PyTorch models (like Metric3D, DSINE).
    *   `_init(self, conf)`: Loads the pre-trained weights for the specific third-party model and any of its native configurations.
    *   `_forward(self, data_dict)`:
        *   Preprocesses the input image and intrinsics from `data_dict` to the format expected by the third-party model.
        *   Performs inference using the third-party model.
        *   Postprocesses the raw outputs: This includes resizing/interpolating to match original image dimensions, converting coordinate systems if needed (e.g., `omni_to_bni` for normals), and deriving initial uncertainty values (e.g., depth variance from model confidence).
        *   Handles "flip consistency": If configured (e.g., by `return_types` in the config requesting "depth2" or "normals2"), it re-runs inference on a horizontally flipped version of the image and processes these flipped results, transforming them back to the original image's coordinate frame.
        *   Returns a `prediction_dict` containing various prior maps (e.g., "depth", "depth_variance", "normals", "normals_confidence", "valid", and potentially "depth2", "normals2", "depth_variance2", etc.).

3.  **Data Storage (HDF5 Format)**:
    *   The `write` function in `mpsfm/extraction/imagewise/geometry/base.py` takes the `prediction_dict` from the model wrapper.
    *   It saves this dictionary into an HDF5 file (e.g., `metric3dv2.h5`, `dsine-kappa.h5`).
    *   Within the HDF5 file, data for each image is stored in a group named after the image (e.g., group "image001.jpg").
    *   Inside each image group, datasets are created for each key in the `prediction_dict` (e.g., dataset "depth", dataset "depth_variance").

4.  **In-Pipeline Data Structures (During SfM Reconstruction)**:
    *   **`MpsfmReconstruction`**: Located in `mpsfm/sfm/scene/reconstruction/base.py`. This class wraps a `pycolmap.Reconstruction` object. Its `_images` attribute is a dictionary mapping image IDs to instances of the MP-SfM `Image` class (see below), allowing custom data to be associated with each image in the COLMAP reconstruction.
    *   **`Image`**: Located in `mpsfm/sfm/scene/image/base.py`. This class wraps a `pycolmap.Image` object. It's the primary container for prior information related to an image during the SfM process.
        *   Its `init_depth()` method is responsible for loading data from HDF5 files (using `mpsfm.utils.io.get_mono_map`) and instantiating `Depth` and `Normals` objects.
        *   It stores these instances as `self.depth` and `self.normals`.
    *   **`Depth`**: Located in `mpsfm/sfm/scene/image/depth.py`. Manages depth information for an image.
    *   **`Normals`**: Located in `mpsfm/sfm/scene/image/normals.py`. Manages normal information for an image.

5.  **Uncertainty Calculation and Representation**:
    *   **`Depth` Class (`mpsfm.sfm.scene.image.depth.Depth`)**:
        *   Stores the final processed depth map as `self.data_prior` (NumPy array).
        *   Stores pixel-wise scalar variance as `self.uncertainty` (NumPy array).
        *   The uncertainty calculation is highly configurable via its `conf` object (passed during instantiation):
            *   It can fuse the model's own prior variance (from `depth_dict['depth_variance']`).
            *   It can incorporate variance derived from "flip consistency" (i.e., `(depth - depth2)^2`).
            *   It can add a variance term proportional to the depth value itself (`(depth * conf.depth_uncertainty_scalar)^2`).
            *   It can use a fixed variance value (`conf.fixed_uncertainty_val`).
            *   The combination often involves taking the maximum (most conservative) variance if multiple sources are enabled.
            *   Final variance is subject to clipping by `conf.inherent_noise` (min) and `conf.max_std` (max), and scaling by `conf.std_multiplier`.
    *   **`Normals` Class (`mpsfm.sfm.scene/image/normals.Normals`)**:
        *   Stores the final processed normal map as `self.data` (HxWx3 NumPy array).
        *   Stores pixel-wise 3x3 Cartesian covariance matrices as `self.uncertainty` (HxWx3x3 NumPy array).
        *   The uncertainty calculation is also configurable:
            *   **With Flip Consistency**: The `two_view_covariance` function is used. This takes the original normal map (`N1`), the flipped normal map (`N2`), and their respective scalar variances (`v1`, `v2` - derived from model's confidence/kappa). It computes a 2x2 covariance in spherical coordinates (theta, phi), incorporating `v1`, `v2`, noise parameters, and multipliers. This 2x2 spherical covariance is then transformed into a 3x3 Cartesian covariance using the Jacobian of the spherical-to-Cartesian transformation.
            *   **Without Flip Consistency (Single Normal Map)**: The scalar variance map (`v1`) provided by the model is used to form a diagonal 2x2 spherical covariance matrix at each pixel (i.e., `[[v1, 0], [0, v1]]`). Inherent polar noise is added to these diagonal elements. This spherical covariance is then transformed to a 3x3 Cartesian covariance using the Jacobian derived from the mean normal map.
        *   The final 3x3 covariance is also scaled by `conf.std_multiplier`.
        *   Downscaled versions of normals and their covariances are also computed and stored.

## II. Conceptual Code Snippets for Encapsulation

*(These are high-level conceptual snippets. Some details are simplified for brevity.)*

**Snippet A: Orchestrating Initial Extraction (Conceptual)**
\`\`\`python
# Based on mpsfm/extraction/imagewise/geometry/base.py
# Depends on: mpsfm.extraction.load_model, HDF5 utils, DataLoader setup

def run_geometry_extraction_conceptual(model_config, image_paths_list, scene_parser_obj, output_h5_path):
    # 1. Setup PyTorch DataLoader using scene_parser_obj for image_paths_list.
    #    The loader provides batches with 'image', 'intrinsics', 'meta'.
    # loader = scene_parser_obj.dataset(...).get_dataloader()

    # 2. Load the specific model wrapper (e.g., Metric3Dv2, DSINE)
    # model_instance = load_model(model_config) # from mpsfm.extraction import load_model
    
    # 3. For each data_batch from DataLoader:
    #      # Prepare input: scale image, scale intrinsics if needed
    #      prepared_input_dict = {"image": scaled_image_np, "intrinsics": scaled_intrinsics_np, "meta": data_batch["meta"]}
    #      
    #      # Perform inference via the model wrapper
    #      prediction_output_dict = model_instance(prepared_input_dict) # Calls model_instance._forward()
    #      
    #      # Add image name to predictions for storage
    #      image_name = data_batch["meta"]["image_name"][0]
    #      prediction_output_dict["name"] = image_name 
    #      
    #      # Save to HDF5
    #      # with h5py.File(output_h5_path, "a") as fd:
    #      #   grp = fd.create_group(image_name)
    #      #   for k, v in prediction_output_dict.items(): grp.create_dataset(k, data=v)
    pass
\`\`\`
*This snippet outlines how models are loaded and invoked per image, and results saved to HDF5.*

**Snippet B: Model Wrapper's Forward Pass (Conceptual `Metric3Dv2._forward`)**
\`\`\`python
# Based on mpsfm/extraction/imagewise/geometry/models/depth/metric3dv2.Metric3Dv2

class ConceptualMetric3DWrapper: # Simplified
    def __init__(self, model_specific_conf):
        self.conf = model_specific_conf
        # In reality: self.third_party_metric3d_model = load_pytorch_model_from_disk()
        # In reality: self.metric3d_native_config = load_config_for_third_party_model()
        pass

    def _process_single_view_output(self, raw_depth_tensor, raw_normal_tensor, raw_confidence_tensor, 
                                   padding_info, original_hw_shape, metric3d_cfg_details):
        # 1. Slice padding from raw tensors and interpolate to original_hw_shape.
        #    depth_interpolated, normal_interpolated, confidence_interpolated = ...
        # 2. Scale depth: 
        #    depth_scaled = depth_interpolated * metric3d_cfg_details.normalize_scale / metric3d_cfg_details.label_scale_factor
        # 3. Calculate depth_variance from confidence: 
        #    error = depth_scaled * (1 - confidence_interpolated); variance = error**2
        # 4. Convert normal_confidence to normal_variance (e.g., using kappa_to_alpha function).
        # 5. Transform normals to desired output coordinate system (e.g., omni_to_bni).
        # return {"depth": depth_scaled_np, "depth_variance": variance_np, 
        #           "normals": processed_normals_np, "normals_confidence": normal_confidence_np, 
        #           "normals_variance": normal_variance_np, "valid": valid_mask_np} # valid_mask from depth range
        pass

    def forward(self, data_from_orchestrator): # data_from_orchestrator has 'image', 'intrinsics'
        # 1. Preprocess data_from_orchestrator['image'] (e.g., using Metric3D's transform_test_data_scalecano)
        #    This yields: input_tensor_for_nn, padding_info, label_scale_factor, normalize_scale_factor.
        #
        # 2. Perform inference:
        #    raw_nn_output = self.third_party_metric3d_model.inference(input_tensor_for_nn)
        #    # raw_nn_output contains 'prediction' (depth), 'prediction_normal', 'confidence'.
        # 
        # 3. Post-process the output:
        #    results_dict = self._process_single_view_output(raw_nn_output['prediction'], raw_nn_output['prediction_normal'], 
        #                                                  raw_nn_output['confidence'], padding_info, 
        #                                                  original_hw_shape=data_from_orchestrator['image'].shape[:2], 
        #                                                  metric3d_cfg_details=self.metric3d_native_config.data_basic)
        #
        # 4. Handle flip consistency if self.conf.return_types requests "depth2", "normals2", etc.:
        #    - Create flipped_input_tensor_for_nn.
        #    - flipped_raw_nn_output = self.third_party_metric3d_model.inference(flipped_input_tensor_for_nn)
        #    - flipped_results_dict = self._process_single_view_output(...)
        #    - Transform flipped_results_dict geometrically (e.g., flip normals' x-component).
        #    - Add to main results_dict: results_dict["depth2"] = flipped_results_dict["depth"], etc.
        #
        # 5. Filter results_dict to include only keys specified in self.conf.return_types.
        # return results_dict
        pass
\`\`\`
*This snippet shows how a model wrapper runs the underlying neural network, post-processes results to get depth, normals, and their initial variances/confidences, and handles flip-consistency.*

**Snippet C: Initializing `Depth` and `Normals` Objects in MP-SfM `Image` (Conceptual)**
\`\`\`python
# Based on mpsfm.sfm.scene.image.base.Image.init_depth()

class ConceptualMPSFMImageWrapper:
    def __init__(self, image_level_config):
        self.conf = image_level_config # Contains .depth and .normals sub-configs
        self.depth = None
        self.normals = None
        # self._pycolmap_image_obj = ... (actual pycolmap.Image)
        # self._pycolmap_camera_obj = ... (actual pycolmap.Camera)

    def initialize_priors(self, image_name_str, keypoints_np_for_image,
                          raw_depth_data_dict, raw_normals_data_dict, # Loaded from HDF5
                          sky_mask_np_array=None):
        
        # Instantiate Depth object
        # The Depth class's __init__ (see Snippet D) will use its conf and raw_depth_data_dict
        # to calculate final depth and variance maps.
        self.depth = ConceptualDepthPrior(
            conf=self.conf.depth, 
            depth_dict_raw=raw_depth_data_dict, 
            camera=self._pycolmap_camera_obj, # For resolution
            kps=keypoints_np_for_image,    # For sampling at keypoints
            mask=sky_mask_np_array
        )
        
        # Instantiate Normals object
        continuity_mask = self.depth.get_continuity_mask() # Assuming Depth class provides this
        
        # The Normals class's __init__ (see Snippet E) will use its conf and raw_normals_data_dict
        # to calculate final normal map and 3x3 covariance maps.
        self.normals = ConceptualNormalPrior(
            conf=self.conf.normals,
            normals_dict_raw=raw_normals_data_dict,
            camera=self._pycolmap_camera_obj,
            mask=sky_mask_np_array,
            continuity_mask=continuity_mask
        )
\`\`\`
*This shows how raw data loaded from HDF5 is passed to `Depth` and `Normals` class constructors along with their specific configurations.*

**Snippet D: `Depth` Class Uncertainty Logic (Conceptual `Depth.__init__`)**
\`\`\`python
# Based on mpsfm.sfm.scene.image.depth.Depth

class ConceptualDepthPrior:
    def __init__(self, conf, depth_dict_raw, camera, kps, mask):
        self.conf = conf # Config for uncertainty calculation
        
        # 1. Fuse multiple depth estimates (if flip_consistency enabled in conf):
        #    e.g., self.data_prior = (depth_dict_raw['depth'] + depth_dict_raw['depth2']) / 2
        #    Store the chosen/fused depth map in self.data_prior (HxW NumPy array).

        # 2. Calculate initial pixel-wise variance based on conf:
        #    - If conf.prior_uncertainty: use depth_dict_raw['depth_variance'] (and 'depth_variance2').
        #    - If conf.flip_consistency: use (depth_dict_raw['depth'] - depth_dict_raw['depth2'])**2.
        #    - If conf.depth_uncertainty_scalar: calculate (self.data_prior * conf.depth_uncertainty_scalar)**2.
        #    - If conf.fixed_uncertainty: use conf.fixed_uncertainty_val.
        #    Store this initial variance.

        # 3. Combine/select variance: 
        #    If multiple strategies active (e.g., prior_uncertainty and depth_uncertainty_scalar),
        #    often take the np.maximum() to be conservative.
        #    Store in self.uncertainty (HxW NumPy array of variances).

        # 4. Apply multipliers and clipping:
        #    self.uncertainty *= (conf.std_multiplier**2)
        #    self.uncertainty = np.clip(self.uncertainty, conf.inherent_noise**2, conf.max_std**2 if conf.max_std else very_large_number)

        # 5. Apply validity masks (from input `mask`, model's `depth_dict_raw['valid']`, continuity checks):
        #    self.uncertainty[~valid_mask_combined] = very_large_value
        #    self.valid_mask = valid_mask_combined
        
        # (Actual implementation involves careful handling of which variances to fuse if multiple are present)
        pass
    
    def get_continuity_mask(self):
        # Calculate and return a mask where depth is locally continuous.
        pass
\`\`\`
*Highlights the configurable logic in the `Depth` class for calculating final pixel-wise scalar variance from various sources.*

**Snippet E: `Normals` Class Uncertainty Logic (Conceptual `Normals.__init__`)**
\`\`\`python
# Based on mpsfm.sfm.scene.image.normals.Normals

class ConceptualNormalPrior:
    def __init__(self, conf, normals_dict_raw, camera, mask, continuity_mask):
        self.conf = conf # Config for uncertainty calculation

        N1_raw = normals_dict_raw['normals'] # HxWx3
        v1_scalar_raw = normals_dict_raw['normals_variance'] # HxW, scalar variance for each normal

        # Resize N1_raw, v1_scalar_raw to camera resolution. Normalize N1_resized.
        # N1 = processed N1_raw
        # v1_scalar = processed v1_scalar_raw

        if conf.get('flip_consistency', False):
            N2_raw = normals_dict_raw['normals2']
            v2_scalar_raw = normals_dict_raw['normals2_variance']
            # Resize N2_raw, v2_scalar_raw. Normalize N2_resized.
            # N2 = processed N2_raw
            # v2_scalar = processed v2_scalar_raw
            
            # self.data (HxWx3) = mean of N1, N2 (normalized)
            
            # self.uncertainty (HxWx3x3) is computed by `two_view_covariance(N1, N2, v1_scalar, v2_scalar, conf_params)`
            # `two_view_covariance` converts N1, N2 to spherical coords, calculates a 2x2 spherical covariance
            # matrix (incorporating v1_scalar, v2_scalar, noise, multipliers), then transforms this
            # 2x2 spherical covariance to a 3x3 Cartesian covariance using Jacobians.
        else: # Single normal map
            # self.data (HxWx3) = N1
            
            # Convert v1_scalar (per-pixel scalar variance) to a HxWx3x3 Cartesian covariance matrix:
            # 1. Create HxWx2x2 diagonal spherical covariance matrix `C_sphere` where diagonals are v1_scalar.
            # 2. Add `conf.inherent_polar_noise**2` to diagonals of `C_sphere`.
            # 3. Convert self.data (mean normal) to spherical coordinates: `N_spherical` (HxWx2).
            # 4. Calculate Jacobian `J` (HxWx3x2) of spherical-to-Cartesian transform at `N_spherical`.
            # 5. self.uncertainty (HxWx3x3) = J @ C_sphere @ J.transpose.
            # 6. Scale self.uncertainty by `conf.prior_std_multiplier**2`.
        
        # Scale final self.uncertainty by `conf.std_multiplier**2`.
        # Apply input `mask` and `continuity_mask` to self.uncertainty (e.g., by setting cov for masked pixels to identity * large_val).
        # Compute self.data_downscaled, self.uncertainty_downscaled.
        pass
\`\`\`
*Highlights the logic in the `Normals` class for calculating a 3x3 Cartesian covariance matrix per pixel, either from flip-consistent normal pairs or a single normal map with its scalar variance, involving spherical coordinate transformations and Jacobians.*

## III. Proposed Encapsulation Strategies for Reusability

To make these functionalities easier to integrate into another project like `align3r`:

1.  **`MonocularPriorExtractor` Class**:
    *   **Responsibility**: Abstract away the specifics of different third-party models (Metric3D, DSINE, etc.). Handle model loading (using existing `load_model` logic which includes `BaseModel` for downloads) and performing raw inference.
    *   **Interface**: `MonocularPriorExtractor(model_config_dict) -> extractor_instance`; `extractor_instance.extract(image_numpy, intrinsics_numpy=None) -> prediction_dict`.
    *   The `prediction_dict` would contain the direct outputs from the model wrappers (e.g., "depth", "depth_variance", "normals", "normals_confidence", "valid", and "depth2", "normals2" if the model's internal flip-consistency is used).

2.  **`PriorHDF5Store` Class**:
    *   **Responsibility**: Centralize all HDF5 reading and writing operations for prior data.
    *   **Interface**: `save_priors(hdf5_filepath, image_name, prediction_dict, model_identifier_str)`; `load_priors(hdf5_filepath, image_name, model_identifier_str) -> prediction_dict`.

3.  **`DepthPrior` and `NormalPrior` Classes (with Factory Functions)**:
    *   **Responsibility**: These classes would take the raw `prediction_dict` (loaded from HDF5 or directly from `MonocularPriorExtractor`) and the relevant processing configuration. They would encapsulate all the complex logic for fusing multiple estimates (e.g., original and flipped), calculating final uncertainty (scalar variance for depth, 3x3 covariance for normals), applying masks, and providing clean accessors to the processed data.
    *   **Factory Interface (Conceptual)**:
        *   `create_depth_prior(raw_depth_dict, camera_colmap_obj, depth_processing_config_dict, keypoints_optional_np, mask_optional_np) -> DepthPriorInstance`
        *   `create_normal_prior(raw_normals_dict, camera_colmap_obj, normal_processing_config_dict, keypoints_optional_np, mask_optional_np, continuity_mask_optional_np) -> NormalPriorInstance`
    *   **Instance Attributes**:
        *   `DepthPriorInstance.mean_depth_map` (NumPy HxW)
        *   `DepthPriorInstance.variance_map` (NumPy HxW)
        *   `DepthPriorInstance.valid_mask` (NumPy HxW)
        *   `NormalPriorInstance.mean_normal_map` (NumPy HxWx3)
        *   `NormalPriorInstance.covariance_map` (NumPy HxWx3x3)
        *   Methods to sample prior data and uncertainty at specific keypoint locations.

This layered encapsulation would allow `align3r` (or other projects) to:
1.  Use `MonocularPriorExtractor` to get raw predictions for an image.
2.  (Optionally) Use `PriorHDF5Store` to cache/load these raw predictions.
3.  Use the factory functions with appropriate configurations to create `DepthPrior` and `NormalPrior` objects, which then provide the processed maps and their complexly derived uncertainties ready for consumption.

This structure promotes modularity and makes the sophisticated uncertainty calculations in MP-SfM more accessible.
