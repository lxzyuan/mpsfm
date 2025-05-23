# Code Structure of MP-SfM

This document outlines the structure of the MP-SfM (Monocular Priors for Structure-from-Motion) codebase.

## High-Level Overview

MP-SfM is a Structure-from-Motion pipeline that integrates monocular depth and normal predictions into classical multi-view reconstruction. This hybrid approach improves robustness in difficult scenarios while maintaining strong performance in standard conditions. The codebase is primarily written in Python and leverages the `pycolmap` library for some underlying SfM operations.

## Main Entry Points

The primary ways to run the MP-SfM pipeline are:

*   **`reconstruct.py`**: This script is used for running the MP-SfM pipeline on custom datasets. It takes arguments for data directories, intrinsics, configurations, etc., and produces a 3D reconstruction.
*   **`scripts/benchmark.py`**: This script is used to run benchmark evaluations on standard datasets like ETH3D and SMERF. It automates the process of running reconstructions for multiple scenes and configurations.
*   **`demo.ipynb`**: A Jupyter Notebook that provides a minimal usage example and demonstrates how to run the pipeline and visualize outputs.

## Top-Level Directory Structure

The repository is organized into the following main directories:

*   **`mpsfm/`**: Contains the core source code of the MP-SfM library. This is where the primary logic for data processing, feature extraction, SfM, evaluation, and utilities resides.
*   **`configs/`**: Holds YAML configuration files that control various aspects of the pipeline, including model choices, hyperparameters for different modules (extraction, matching, SfM), and dataset-specific settings.
*   **`scripts/`**: Contains utility scripts for tasks such as running benchmarks (`benchmark.py`) and aggregating experiment results (e.g., `aggregate_experiments/eth3d.py`).
*   **`local/`**: Intended for storing local data, including example datasets (`local/example`) and test sets (`local/testsets/eth3d`, `local/testsets/smerf`). This directory is typically in `.gitignore`.
*   **`assets/`**: Contains static assets like images, GIFs, and diagrams used in the `README.md` or other documentation.
*   **`third_party/`**: Includes external libraries or code integrated as Git submodules (e.g., DSINE, Depth-Anything-V2, RoMa, Metric3D). These are often pre-existing models or tools leveraged by MP-SfM.
*   **`LICENSE`**: The license file for the project.
*   **`README.md`**: The main documentation file providing an overview of the project, setup instructions, and usage examples.
*   **`requirements.txt`**: Lists the Python dependencies required for the project.
*   **`pyproject.toml`**: Project metadata file, often used by modern Python packaging tools.
*   **`format.sh`**: A shell script likely used for code formatting (e.g., running Black, Flake8).

## Core Library Structure: `mpsfm/`

The `mpsfm/` directory is the heart of the library and is further organized as follows:

*   **`mpsfm/data_proc/`**: Modules for processing and loading various datasets.
    *   `basedataset.py`: Base classes for dataset handling.
    *   `simple.py`: Parser and dataset class for custom user-provided data (used by `reconstruct.py`).
    *   Specific dataset parsers (e.g., `eth3d.py`, `smerf.py`).
    *   `hloc/`: Utilities related to the Hierarchical Localization (hloc) data format and feature/match loading.
    *   `prepare/`: Scripts to download and preprocess benchmark datasets.
*   **`mpsfm/extraction/`**: Modules responsible for extracting various types of information from images.
    *   `base.py`, `base_model.py`: Base classes for extraction tasks.
    *   `imagewise/`: Modules for extracting information from individual images.
        *   `features/`: Feature detectors (e.g., SuperPoint).
        *   `geometry/`: Monocular depth and normal estimators (e.g., Metric3Dv2, DSINE).
        *   `mask/`: Sky segmentation or other masking tools.
    *   `pairs/` & `pairwise/`: Modules for processing pairs of images, primarily for feature matching.
        *   `match_sparse.py`: Sparse feature matching (e.g., LightGlue).
        *   `match_dense_2view.py`: Dense feature matching (e.g., RoMA, MASt3R).
*   **`mpsfm/sfm/`**: Core Structure-from-Motion pipeline components.
    *   `reconstruction_manager.py`: Manages the overall reconstruction process, orchestrating the mapper.
    *   `mapper/`: Contains the main mapping logic.
        *   `base.py` (`MpsfmMapper`): The central class that implements the incremental SfM pipeline, including initialization, image registration, triangulation, bundle adjustment, and integration of monocular priors.
        *   `bundle_adjustment.py`: Optimization routines.
        *   `depthconsistency.py`: Checks and enforces consistency with monocular depth priors.
        *   `image_selection.py`: Logic for choosing the next image to register or the initial pair.
        *   `registration.py`: Handles image registration (PnP).
        *   `triangulator.py`: 3D point triangulation.
    *   `estimators/`: Modules for pose estimation (absolute and relative).
    *   `scene/`: Data structures for representing the 3D scene, extending `pycolmap` objects.
        *   `reconstruction.py` (`MpsfmReconstruction`): Custom reconstruction class that extends `pycolmap.Reconstruction` to store and manage monocular depth/normal data and their uncertainties.
        *   `image/` (e.g., `base.py`, `depth.py`, `normals.py`): Custom image, depth, and normal classes that manage the per-image prior information, its uncertainties, and integration into the SfM pipeline.
        *   `correspondences.py`: Manages 2D and 3D correspondences.
*   **`mpsfm/eval/`**: Modules for evaluating reconstruction quality.
    *   `sfm/`: SfM-specific evaluation metrics and dataset aggregators.
*   **`mpsfm/test/`**: Classes that set up and run reconstructions for different scenarios or datasets.
    *   `simple.py` (`SimpleTest`): Used by `reconstruct.py` for custom data.
    *   Dataset-specific test classes (e.g., `eth3d.py`, `smerf.py`) used by `scripts/benchmark.py`.
*   **`mpsfm/utils/`**: General utility functions.
    *   `geometry.py`: Geometric computations.
    *   `io.py`: Input/output operations.
    *   `parsers.py`: Argument and config parsers.
    *   `tools.py`: Miscellaneous helper tools.
    *   `viz.py`, `viz_3d.py`: Visualization utilities.
*   **`mpsfm/vars/`**: Global and local variable definitions.
    *   `gvars.py`: Global variables/paths (e.g., config directories).
    *   `lvars.py`: Local paths, often for benchmark data and experiment outputs (configurable by the user).
*   **`mpsfm/baseclass.py`**: A base class with configuration handling, likely inherited by many other classes in the system.
*   **`mpsfm/__init__.py`**: Makes the `mpsfm` directory a Python package.

## High-Level Reconstruction Workflow (e.g., via `reconstruct.py`)

The following outlines a typical execution flow when running a reconstruction, for example, using the `reconstruct.py` script:

1.  **Initialization & Configuration (`reconstruct.py`)**:
    *   The user executes `reconstruct.py` with command-line arguments specifying the data location, configuration name (e.g., `sp-lg_m3dv2`), and other parameters.
    *   The script loads the specified YAML configuration file from the `configs/` directory. These configs define parameters for all stages of the pipeline.
    *   An instance of `mpsfm.test.simple.SimpleTest` is created, initialized with this configuration.

2.  **Data Parsing (`mpsfm.test.simple.SimpleTest` & `mpsfm.data_proc.simple.SimpleParser`)**:
    *   The `SimpleTest` object calls its internal `SimpleParser`.
    *   `SimpleParser` reads the images from the specified directory and parses the `intrinsics.yaml` file to determine camera parameters.
    *   It populates a `pycolmap.Reconstruction` object with `pycolmap.Camera` and `pycolmap.Image` objects. This object initially contains camera information and image paths but no 3D points or registered poses.

3.  **Reconstruction Management (`mpsfm.sfm.reconstruction_manager.ReconstructionManager`)**:
    *   `SimpleTest` instantiates `ReconstructionManager`, passing the main configuration and the `SimpleParser` (which holds the initial `pycolmap.Reconstruction`).
    *   `ReconstructionManager` is a fairly thin wrapper that primarily instantiates and calls the main pipeline engine: `mpsfm.sfm.mapper.base.MpsfmMapper`.

4.  **Core SfM Pipeline (`mpsfm.sfm.mapper.base.MpsfmMapper`)**:
    This is where the bulk of the SfM work happens. The `MpsfmMapper` is initialized with the configuration, paths, and the `SimpleParser`. Its `__call__` method executes the pipeline:

    *   **a. Extraction (`mpsfm.extraction.Extraction`)**:
        *   Relevant information is extracted from images, driven by the configuration. This can include:
            *   **Features**: e.g., SuperPoint keypoints and descriptors.
            *   **Monocular Priors**: Depth and/or normal maps from models like Metric3Dv2, DSINE.
            *   **Masks**: e.g., Sky masks.
            *   **Image Pairs**: Determining which image pairs to match, often based on image retrieval (e.g., NetVLAD) or other strategies.
            *   **Matches**: Sparse (e.g., LightGlue) or dense (e.g., RoMA, MASt3R) feature matching between selected pairs.
        *   Extracted data is often cached to disk in `cache_dir` to speed up subsequent runs.

    *   **b. Data Preparation & Initialization**:
        *   An `MpsfmReconstruction` object is created, which is a custom wrapper around `pycolmap.Reconstruction`. It's designed to store and manage not just COLMAP's standard data but also the monocular priors and their uncertainties.
        *   `Correspondences` are established using the extracted features and matches.
        *   Monocular depth/normal maps are loaded into the `MpsfmReconstruction`.

    *   **c. Initial Pair Reconstruction**:
        *   The `ImageSelection` module identifies a robust initial pair of images.
        *   The `MpsfmRegistration` module attempts to register this pair (estimate relative pose) and `MpsfmTriangulator` triangulates initial 3D points.
        *   This initial reconstruction undergoes refinement (`post_init_refinement`):
            *   Bundle Adjustment (`Optimizer`): Camera poses and 3D points are optimized.
            *   Prior Integration: Monocular depth priors are used to help determine the scale and potentially refine point positions.
            *   Filtering: Unreliable points or views might be filtered out.
        *   `DepthConsistencyChecker` may be used to validate the initial model against depth priors.
        *   A global bundle adjustment is typically run.

    *   **d. Incremental Reconstruction (Loop)**:
        *   The pipeline iteratively adds more views:
            *   **Image Selection (`ImageSelection`)**: Choose the next best view to register.
            *   **Registration (`MpsfmRegistration`)**: Estimate the pose of the new view using 2D-3D correspondences (PnP).
            *   **Triangulation (`MpsfmTriangulator`)**: Triangulate new 3D points observed by the new view and existing registered views.
            *   **Refinement (`post_registration_refinement`, `iterative_local_refinement`)**:
                *   Local Bundle Adjustment: Refine the pose of the new image and nearby views/points.
                *   Prior Integration: Monocular priors are used to refine points and check for consistency (`DepthConsistencyChecker`).
                *   Filtering: Remove erroneous points/observations.
            *   **Global Refinement**: Periodically, a global bundle adjustment (`iterative_global_refinement`) is run over the entire model to maintain consistency and accuracy. This also involves prior integration and filtering.
        *   This loop continues until no more images can be reliably registered or all images are processed.

    *   **e. Finalization**:
        *   A final global bundle adjustment is performed.
        *   The `MpsfmMapper` returns the completed `MpsfmReconstruction` object.

5.  **Output (`reconstruct.py`)**:
    *   The `ReconstructionManager` returns the final `MpsfmReconstruction` to `SimpleTest`.
    *   `reconstruct.py` takes this reconstruction object and calls its `write()` method to save the results to the specified output directory (`sfm_outputs_dir`). This typically includes sparse point clouds, camera poses, and potentially dense reconstructions if generated.

This workflow highlights how MP-SfM extends traditional SfM by systematically incorporating monocular priors at various stages, from initialization and point triangulation to bundle adjustment and consistency checks, aiming for more robust and accurate 3D reconstructions.
