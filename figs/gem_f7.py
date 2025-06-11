import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import matplotlib.ticker as mticker

# --- Configuration ---
# The root directory containing the dataset scenes.
DATA_ROOT = Path("/mnt/data/0/cedar/datasets/mpsfm_local/benchmarks/eth3d/cache_dir")
# The name of the HDF5 file containing predictions.
H5_FILENAME = "metric3dv2.h5"
# The subdirectory for ground truth depth maps.
GT_DIR_NAME = "gt_depth"

# --- List of all scenes to be processed ---
SCENES = [
    "courtyard", "delivery_area", "electro", "facade", "kicker", "meadow",
    "office", "pipes", "playground", "relief", "relief_2", "terrace", "terrains"
]

# --- Analysis Parameters ---
# The paper states Metric3D-v2 is most reliable when combining its prior
# with a depth-proportional uncertainty.
# This factor for depth-proportional uncertainty is not specified in the paper,
# so we choose a common value (e.g., 5% of the depth value).
UNC_PROPORTIONAL_FACTOR = 0.05
# Number of bins for the calibration plot.
NUM_CALIBRATION_BINS = 20
# Maximum uncertainty value to consider for the calibration plot to avoid outliers.
MAX_UNC_CLIP = 2.0

def load_all_data(data_root: Path, scenes: list):
    """
    Loads all valid ground truth depths, predicted depths, and uncertainties
    from a specific list of scenes.

    Args:
        data_root: Path to the root of the dataset cache.
        scenes: A list of scene names to process.

    Returns:
        A tuple of flattened numpy arrays:
        (all_gt_depths, all_pred_depths, all_pred_uncertainties)
    """
    all_gt_depths = []
    all_pred_depths = []
    all_pred_uncertainties = []

    if not scenes:
        raise ValueError("The list of scenes cannot be empty.")

    print(f"🔍 Processing {len(scenes)} specified scenes. Loading data...")
    for scene_name in tqdm(scenes, desc="Processing scenes"):
        scene_dir = data_root / scene_name
        h5_path = scene_dir / H5_FILENAME
        gt_dir = scene_dir / GT_DIR_NAME

        if not h5_path.exists():
            print(f"Warning: H5 file not found for scene '{scene_name}', skipping.")
            continue
        if not gt_dir.exists():
            print(f"Warning: Ground truth directory not found for scene '{scene_name}', skipping.")
            continue

        with h5py.File(h5_path, 'r') as f:
            gt_files = list(gt_dir.glob("*.npy"))
            for gt_path in gt_files:
                img_name = gt_path.stem
                if img_name not in f:
                    continue

                # Load data
                gt_depth = np.load(gt_path)
                pred_depth = f[img_name]['depth'][:]
                pred_uncertainty = f[img_name]['uncertainty'][:]

                # Create a mask for valid ground truth pixels
                valid_mask = gt_depth > 1e-6
                
                # Ensure dimensions match
                if gt_depth.shape != pred_depth.shape:
                    # Simple center crop if shapes mismatch, can be adjusted
                    h_gt, w_gt = gt_depth.shape
                    h_pred, w_pred = pred_depth.shape
                    h_min, w_min = min(h_gt, h_pred), min(w_gt, w_pred)
                    
                    gt_depth = gt_depth[:h_min, :w_min]
                    pred_depth = pred_depth[:h_min, :w_min]
                    pred_uncertainty = pred_uncertainty[:h_min, :w_min]
                    valid_mask = valid_mask[:h_min, :w_min]

                # Flatten and append valid data
                all_gt_depths.append(gt_depth[valid_mask])
                all_pred_depths.append(pred_depth[valid_mask])
                all_pred_uncertainties.append(pred_uncertainty[valid_mask])

    if not all_gt_depths:
        raise ValueError("No valid data could be loaded. Check paths and file contents.")

    return (
        np.concatenate(all_gt_depths),
        np.concatenate(all_pred_depths),
        np.concatenate(all_pred_uncertainties),
    )

def plot_sensitivity_analysis(ax, gt, pred, uncertainty_prior):
    """Plots the sensitivity analysis (RMSE vs. Recall)."""
    print("📊 Plotting Sensitivity Analysis...")
    # Define the different uncertainty metrics to be tested
    uncertainty_metrics = {
        "Prior": uncertainty_prior,
        "Depth-Proportional": UNC_PROPORTIONAL_FACTOR * pred,
        "Combined": np.maximum(uncertainty_prior, UNC_PROPORTIONAL_FACTOR * pred),
        "Upper Bound (Oracle)": np.abs(pred - gt),
    }

    for name, unc in uncertainty_metrics.items():
        # Sort pixel indices by uncertainty
        sorted_indices = np.argsort(unc)
        
        # Take sorted data
        sorted_gt = gt[sorted_indices]
        sorted_pred = pred[sorted_indices]

        recalls = np.linspace(0.01, 1.0, 100)
        rmses = []
        for r in recalls:
            # Select top r% of most certain pixels
            num_pixels = int(r * len(sorted_gt))
            if num_pixels == 0: continue
            
            gt_subset = sorted_gt[:num_pixels]
            pred_subset = sorted_pred[:num_pixels]
            
            rmse = np.sqrt(np.mean((gt_subset - pred_subset)**2))
            rmses.append(rmse)
        
        ax.plot(recalls * 100, rmses, label=name)

    ax.set_title("Metric3D-v2 Depth Sensitivity Analysis")
    ax.set_xlabel("Recall (%)")
    ax.set_ylabel("RMSE (m)")
    ax.grid(True, linestyle='--')
    ax.legend()
    print("...Sensitivity Analysis plot complete.")


def plot_calibration_analysis(ax, gt, pred, uncertainty_prior):
    """Plots the calibration analysis (Binned RMSE vs. Uncertainty)."""
    print("📊 Plotting Calibration Analysis...")
    # The paper uses the combined uncertainty for Metric3D-v2's calibration
    # NOTE: The paper mentions an optimized scaling factor.
    # As we cannot derive it from the text, we use a scaling factor of 1.0.
    scaling_factor = 1.0
    
    combined_unc = np.maximum(uncertainty_prior, UNC_PROPORTIONAL_FACTOR * pred)
    scaled_unc = combined_unc * scaling_factor
    
    # Clip uncertainties to a max value for stable binning
    valid_mask = scaled_unc < MAX_UNC_CLIP
    scaled_unc = scaled_unc[valid_mask]
    gt = gt[valid_mask]
    pred = pred[valid_mask]
    
    # Bin data by uncertainty
    bins = np.linspace(0, scaled_unc.max(), NUM_CALIBRATION_BINS + 1)
    bin_indices = np.digitize(scaled_unc, bins)
    
    bin_rmses = []
    bin_counts = []
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    
    for i in range(1, len(bins)):
        mask = bin_indices == i
        count = np.sum(mask)
        bin_counts.append(count)
        
        if count > 0:
            bin_rmse = np.sqrt(np.mean((gt[mask] - pred[mask])**2))
            bin_rmses.append(bin_rmse)
        else:
            bin_rmses.append(0)

    # Primary axis for RMSE
    ax.plot(bin_centers, bin_rmses, 'o-', label="Binned RMSE", color='b')
    ax.plot([0, MAX_UNC_CLIP], [0, MAX_UNC_CLIP], 'r--', label="Perfect Calibration (y=x)")
    ax.set_xlabel("Depth Estimate Standard Deviation (m)")
    ax.set_ylabel("RMSE (m)", color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax.set_title("Metric3D-v2 Depth Uncertainty Calibration")
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--')
    ax.set_xlim(0, MAX_UNC_CLIP)
    ax.set_ylim(bottom=0)

    # Secondary axis for histogram
    ax2 = ax.twinx()
    bin_percentages = 100 * np.array(bin_counts) / len(scaled_unc)
    ax2.bar(bin_centers, bin_percentages, width=(bins[1]-bins[0])*0.8, alpha=0.3, color='g', label="Pixel Distribution")
    ax2.set_ylabel("Bin Size (%)", color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='upper right')
    
    print("...Calibration Analysis plot complete.")


def main():
    """Main function to run the full analysis."""
    try:
        gt, pred, unc = load_all_data(DATA_ROOT, SCENES)
        
        # Create the figure with two subplots, similar to Figure 7
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Reproduction of Figure 7 for Metric3D-v2", fontsize=16)
        
        # --- Left Plot: Sensitivity ---
        plot_sensitivity_analysis(ax1, gt, pred, unc)
        
        # --- Right Plot: Calibration ---
        plot_calibration_analysis(ax2, gt, pred, unc)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        print("Please ensure DATA_ROOT and SCENES are set correctly and the file structure is as expected.")

if __name__ == "__main__":
    main()