# Guide to Uncertainty Propagation in MP-SfM

## 1. Introduction

This document explains how the initial uncertainties of monocular depth priors (scalar variance) and normal priors (3x3 covariance matrix), as calculated by utilities like `uncertainty_calculators.py`, are propagated and utilized within the main MP-SfM Structure-from-Motion pipeline.

The primary components responsible for using these uncertainties are:
*   `mpsfm.sfm.mapper.bundle_adjustment.Optimizer` (for Bundle Adjustment)
*   `mpsfm.sfm.mapper.depthconsistency.DepthConsistencyChecker` (for ensuring geometric consistency with depth priors)
*   `mpsfm.sfm.scene.image.integration.Integration` (mixin for the `mpsfm.sfm.scene.image.base.Image` class, for refining depth maps using normal priors and propagating normal uncertainty into the refined depth's uncertainty)
*   `mpsfm.sfm.mapper.triangulator.MpsfmTriangulator` (for point triangulation, with indirect influence from prior uncertainty).

## 2. Depth Prior Uncertainty (Scalar Variance) Propagation

The per-pixel scalar variance associated with each monocular depth map is a crucial piece of information that MP-SfM uses to modulate the influence of these priors. This uncertainty is typically stored in an `Image.depth.uncertainty` attribute (as a full map) and sampled at keypoint locations into `Image.depth.uncertainty_update` for efficient use in optimization. Its propagation and application occur in several key stages of the SfM pipeline:

### 2.1. Bundle Adjustment (BA) - `Optimizer`

Bundle Adjustment is the primary stage where depth uncertainties have a significant impact. The `mpsfm.sfm.mapper.bundle_adjustment.Optimizer` class incorporates depth priors to regularize the complex non-linear optimization problem of refining 3D point positions, camera poses, and potentially intrinsics.

*   **Purpose in BA**: Depth priors help stabilize the solution, prevent scale drift (especially in monocular SfM), and guide the reconstruction towards geometrically plausible outcomes, particularly in regions with sparse features or ambiguous geometry. The uncertainty associated with these priors dictates how strongly they influence the solution.

*   **Mechanism of Use**:

    *   **Weighted Depth Residuals**:
        *   **Concept**: The core of BA is minimizing a cost function composed of various residuals. For depth priors, a residual term is typically formulated to penalize the difference between the depth of a 3D point (when projected into an image frame) and the monocular depth value observed for that corresponding pixel/keypoint in the prior map.
        *   **Uncertainty Application**: The scalar depth variance (σ²) is used to weight this residual, often as `1/σ²` (precision). An observation with lower variance (higher certainty) gets a higher weight, meaning it will exert a stronger "pull" on the 3D point to conform to the prior depth. Conversely, a high-variance (low certainty) prior will have a diminished influence, allowing the geometric reprojection error terms to dominate for that observation.
        *   **Code Pointer**: The `__build_problem` method within `mpsfm.sfm.mapper.bundle_adjustment.Optimizer`.
            *   `uncertainty_update = image.depth.uncertainty_update`: This dictionary/array holds the pre-sampled variances at keypoint locations.
            *   `variances = np.array([uncertainty_update[pt2D_id] for pt2D_id in p2Ds])`: Retrieves the variance for each 2D observation involved in a depth residual.
            *   `inv_uncert = 1 / variances.clip(1e-6, None)`: Calculates the precision.
            *   `magnitudes = depths**2 * inv_uncert` is passed to `pycolmap.create_depth_bundle_adjuster`. Given that the depth residuals are often formulated in log-depth space (`logloss=True`), this `magnitudes` term acts as the information matrix (or inverse variance) for the log-depth residual. Specifically, if residual `r = log(d_observed) - log(d_model)`, and `var(log(d)) ≈ var(d)/d²`, then `magnitudes` is `d² * (1/var(d)) = 1 / var(log(d))`.

    *   **Robust Loss Function Scaling**:
        *   **Concept**: To mitigate the impact of outlier observations (e.g., an incorrect depth prior value), robust loss functions (like Cauchy or Huber) are employed instead of a simple L2 (squared error) loss. These functions have a scale parameter that determines the point at which an error is considered large enough to be down-weighted.
        *   **Uncertainty Application**: The standard deviation of the depth prior (σ = √variance) is used to adaptively set this scale parameter for the robust loss function associated with each depth residual. A more uncertain prior (larger σ) will have a larger scale for its loss function, meaning it tolerates a greater discrepancy before its influence is capped or reduced.
        *   **Code Pointer**: `mpsfm.sfm.mapper.bundle_adjustment.Optimizer.__build_problem`.
            *   `params = m * variances**0.5 / depths`: This `params` variable, which includes the standard deviation (`variances**0.5`), is passed to `pycolmap.create_depth_bundle_adjuster` and is used to set the scale of the robust loss function (e.g., for Cauchy loss, the parameter is `delta`). `m` itself is a product of a base robustness parameter (`conf.rob_std`) and a dynamic `truncation_multiplier`.

    *   **Dynamic Outlier Truncation Scaling (`truncation_multiplier`)**:
        *   **Concept**: The overall sensitivity to outliers across all depth priors can be dynamically adjusted based on how well the current 3D model globally agrees with the monocular depth observations.
        *   **Uncertainty Application**: Depth variances are used to "whiten" or normalize the residuals between projected 3D point depths and monocular depths. The statistical spread (e.g., Median Absolute Deviation, MAD) of these whitened residuals is then used to set a `truncation_multiplier`. This multiplier then scales the `m` factor mentioned above.
        *   **Code Pointer**: `mpsfm.sfm.mapper.bundle_adjustment.Optimizer.update_truncation_multiplier` method.
            *   `dstds = variances**0.5` (standard deviations from depth prior uncertainties).
            *   `log_stds = dstds / depths` (std dev of log-depth).
            *   `witened_log_distances = (log(depths) - log(depth3ds)) / log_stds`.
            *   The MAD of `witened_log_distances` determines `sigma`, which becomes `self.truncation_multiplier`.

### 2.2. Depth Consistency Checks - `DepthConsistencyChecker`

After image registration and initial point triangulation, MP-SfM performs depth consistency checks to ensure that the 3D model aligns with the monocular depth priors across multiple views.

*   **Purpose**: This step helps to identify and potentially filter out parts of the reconstruction that are inconsistent with the observed depth information, which is particularly useful for handling drift or errors in challenging geometric configurations.

*   **Mechanism of Use**:
    *   **Concept**: For a pair of images observing common 3D points (or where one can reproject its depth into the other), the depth value of a point from one image's perspective (either its direct monocular prior or a reprojected 3D point's depth) is compared against the monocular depth prior in the second image.
    *   **Uncertainty Application**: The scalar depth variances from *both* images' depth priors are used to establish a statistically meaningful tolerance for this comparison. A larger discrepancy between the two depth estimates is permissible if one or both priors are highly uncertain.
    *   **Code Pointer**: `mpsfm.sfm.mapper.depthconsistency.DepthConsistencyChecker.check_depth_consistency` method.
        *   `var1 = self.mpsfm_rec.images[imid1].depth.uncertainty.copy()`: Retrieves the full variance map for the source image.
        *   This variance is used in `self.mpsfm_rec.lifted_pointcovs_cam(...)` which conceptually "lifts" a 2D pixel with its depth and depth uncertainty to a 3D point with a 3D covariance. This 3D covariance is then projected into the second camera's view, and the variance of the z-component in the second camera (`std1bar**2`) gives the uncertainty of the reprojected depth.
        *   The comparison is typically of the form `t1 = (depth_reprojected_from_img1_into_img2 - depth_prior_img2) / combined_standard_deviation`.
        *   The `combined_standard_deviation` is derived from `std1bar` (uncertainty of the reprojected depth) and `std2` (uncertainty of image 2's own monocular depth prior at that point): `sqrt((std1bar * c)**2 + (std2_at_point * c)**2)`. The `c` factor scales these standard deviations.
        *   If `abs(t1)` exceeds a threshold (e.g., `score_thresh`), the point is considered inconsistent. Higher original depth uncertainties lead to a larger `combined_standard_deviation`, thus a more lenient threshold for consistency.

### 2.3. Triangulation - `MpsfmTriangulator` (Indirect Influence)

While the scalar depth variance might not be directly used in the mathematical formulation of standard triangulation algorithms (which primarily use 2D correspondences and camera poses), it has an indirect influence in MP-SfM, particularly in scenarios involving "lifting" points using monocular depth.

*   **Purpose**: To create initial 3D points from matched 2D features.
*   **Mechanism ("Lifting" Low-Parallax or Risky Points)**:
    *   **Concept**: When a 3D point is triangulated with poor geometry (e.g., very small triangulation angle between rays, indicating high uncertainty in depth), its position can be very unreliable. MP-SfM can use a monocular depth prior to directly place this point in 3D by "lifting" one of its 2D observations along its ray by the prior depth.
    *   **Uncertainty Application (Indirect)**:
        *   The decision to use a monocular depth value for lifting depends on the prior being considered "valid". This validity is determined when the `Image.depth` object is initialized, and the process takes into account the initial raw uncertainty from the model – very noisy or unreliable regions of the depth map might be marked as invalid from the start.
        *   `valid = self.mpsfm_rec.images[limid].depth.valid_at_kps(xy)`: This check ensures that the depth prior at the keypoint location is considered usable.
        *   If the depth value used for lifting (`d = self.mpsfm_rec.images[limid].depth.data_at_kps(xy)`) is a result of previous bundle adjustment (i.e., `image.depth.data` which might have been refined), then its value has already been influenced by its uncertainty during those BA steps.
    *   **Code Pointer**: `mpsfm.sfm.mapper.triangulator.MpsfmTriangulator._triangulate_image` and `retriangulate` methods, specifically the sections that handle `lift_low_parallax` or re-triangulation of risky points.

In summary, the scalar depth variance is a vital input that allows the MP-SfM pipeline to intelligently weight the influence of monocular depth priors, making the system more robust to noise and outliers, and enabling more reliable reconstruction in challenging scenarios compared to purely geometric methods.

## 3. Normal Prior Uncertainty (3x3 Covariance) Propagation

The per-pixel 3x3 Cartesian covariance matrix associated with each monocular normal map (stored in `Image.normals.uncertainty`) represents a more complex directional uncertainty. In MP-SfM, this information is primarily used to guide a per-image depth map optimization process, making the depth map consistent with the normal map. The uncertainty of this *optimized depth map* then carries the influence of normal uncertainty into the main Bundle Adjustment.

### 3.1. Normal-Guided Depth Map Integration - `Image` class (via `Integration` mixin)

This is the core stage where normal uncertainties (the 3x3 covariance matrices) are actively used. The `mpsfm.sfm.scene.image.integration.Integration` class (a mixin for `mpsfm.sfm.scene.image.base.Image`) contains methods to optimize an image's current depth map (`self.depth.data`) by making it conform to the geometry implied by its normal map (`self.normals.data`).

*   **Purpose**: To produce a depth map that is locally smooth and consistent with the detailed surface orientations provided by the normal map. This can improve the quality of the depth map, especially if the initial depth prior was noisy or lacked fine detail that the normal map captured.

*   **Mechanism of Use (`_integrate` method)**:
    *   **Concept**: The method sets up an energy minimization problem to solve for an optimal log-depth map `z`. The energy function includes terms that enforce smoothness based on the normal map. Specifically, it tries to ensure that the depth gradients `(dz/du, dz/dv)` are consistent with the normal components `(-nx/nz, -ny/nz)` (after accounting for camera intrinsics and the normal map's coordinate system).
    *   **Uncertainty Application**: The **diagonal elements (variances)** of the 3x3 Cartesian normal covariance matrix (`self.normals.uncertainty`) are used to weight these smoothness constraints.
        *   **Code Pointers**: `mpsfm.sfm.scene.image.integration.Integration.process_normals_prior` and its use within `_integrate` (and `init_int_vars` called by `_integrate`).
            *   `Vnx = (1 / self.conf.normals_magnitude_multiplier) * cp.asarray(normals_uncertainty[..., 1, 1].flatten())` (similarly for `Vny`, `Vnz`, note the component indexing depends on normal vector orientation conventions).
            *   These variances `Vnx, Vny, Vnz` contribute to `Nu_precision` and `Nv_precision` (precision terms for the smoothness constraints in u and v directions). For instance:
                `Nu_precision = 1 / (Vnx * ((uu * Duz_estim + ones_like_z)**2) + Vny * (vv * Duz_estim)**2 + fx**2 * Vnz * Duz_estim**2)`
            *   These precision terms scale the weights (`wu_plus`, `wu_minus`, `wv_plus`, `wv_minus`) that are applied to the squared error terms in the energy function (e.g., `wu_plus * (self.A1.dot(z) + nx)**2`).
        *   **Effect**: If a normal vector component at a pixel has a high variance (from its 3x3 covariance matrix), the corresponding precision term (`Nu_precision` or `Nv_precision`) will be low. This reduces the weight of the smoothness constraint that relies on that normal component in that particular direction. Thus, noisy or uncertain parts of the normal map have less influence on shaping the integrated depth map.
*   **Propagation to Integrated Depth's Uncertainty**:
    *   **Concept**: After the depth map `z` is optimized using the normal-guided smoothness terms, MP-SfM estimates the uncertainty of this *newly refined depth map*. This new uncertainty now implicitly carries information from both the original depth prior's uncertainty and the normal prior's uncertainty (which influenced the integration process).
    *   **Uncertainty Use**: The Hessian of the depth integration energy function is computed. This Hessian matrix depends on the normal component variances (via `Nu_precision`, `Nv_precision`). The inverse of this Hessian (or a method to solve systems with it, like Cholesky decomposition) is then used to estimate the variance of the optimized log-depth values (`varlogd`) at specific keypoint locations.
    *   **Code Pointers**: `mpsfm.sfm.scene.image.integration.Integration.calculate_hessian` and `calculate_int_covs_at_kps` (which calls `calculate_int_covs_at_points` that uses `IntegrationUncertainty.solve`).
        *   `calculate_hessian`: Builds the Hessian matrix `A_mat` (same as in `_integrate`), which includes `Nu_precision` and `Nv_precision` in its terms.
        *   `IntegrationUncertainty.solve(tgt, x)`: Solves `Hessian * x = tgt` to get `x`, where `x` effectively gives columns of the inverse Hessian. `variances = (x).sum(0)` sums these up to get the variance of log-depth.
        *   `uncert = log_uncert * self.depth.data_prior_at_kps(kps)**2`: This converts the variance of log-depth to the variance of depth.
        *   `self.depth.uncertainty_update[pts2d] = uncert`: The crucial step where the `Depth` object's sampled uncertainty is updated with this new value that has incorporated the influence of normal uncertainties.
    *   **Effect**: This updated `self.depth.uncertainty_update` (now influenced by normal uncertainties) is then used in the main Bundle Adjustment (`Optimizer`) as described in Section 2.1.

### 3.2. Bundle Adjustment (Indirect Influence via Updated Depth Uncertainty)

*   As established, the `Optimizer` in `bundle_adjustment.py` does not seem to have direct residual terms for normal priors that use the 3x3 covariances.
*   Instead, the primary way normal uncertainty (the 3x3 covariance) influences the global Bundle Adjustment is **indirectly**:
    1.  The normal map and its 3x3 covariance (specifically its diagonal elements) guide the per-image depth map optimization process (`Image.integrate`).
    2.  The uncertainty of this *integrated depth map* is re-estimated, and this new depth uncertainty now reflects the confidence from both the original depth prior *and* the normal prior.
    3.  This updated depth scalar variance (`Image.depth.uncertainty_update`) is then used by the `Optimizer` in its depth prior residual terms, as detailed in Section 2.1.

Therefore, while the full 3x3 normal covariance is processed to extract component-wise variances for the depth integration stage, its influence on BA is channeled through its impact on the per-image refined depth maps and their re-evaluated scalar variances.

## 4. Key Classes and Files Summary

*   **Initial Uncertainty Storage**:
    *   `mpsfm.sfm.scene.image.depth.Depth`: Stores `data_prior` (depth map) and `uncertainty` (scalar variance map).
    *   `mpsfm.sfm.scene.image.normals.Normals`: Stores `data` (normal map) and `uncertainty` (3x3 covariance map).
*   **Uncertainty Usage & Propagation**:
    *   `mpsfm.sfm.mapper.bundle_adjustment.Optimizer`: Uses `Depth.uncertainty_update` for weighting depth residuals and scaling robust loss in BA.
    *   `mpsfm.sfm.mapper.depthconsistency.DepthConsistencyChecker`: Uses `Depth.uncertainty` to set tolerances for consistency checks.
    *   `mpsfm.sfm.scene.image.integration.Integration` (mixin for `mpsfm.sfm.scene.image.base.Image`):
        *   Uses diagonal elements of `Normals.uncertainty` to weight normal-based smoothness terms during per-image depth map optimization.
        *   Propagates normal uncertainty into the uncertainty of the *optimized depth map* by calculating a Hessian and using its inverse. This updated depth uncertainty is then used by the `Optimizer`.
    *   `mpsfm.sfm.mapper.base.MpsfmMapper`: Orchestrates these components.
```
