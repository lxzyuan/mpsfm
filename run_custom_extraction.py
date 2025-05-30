import yaml
from pathlib import Path
from omegaconf import OmegaConf
import argparse

# Adjust these paths as necessary if your script is not in the project root
# or if the mpsfm library is not directly in the Python path.
try:
    from mpsfm.extraction.imagewise.geometry.base import main as extract_geometry_main
    from mpsfm.data_proc.simple import SimpleParser # Assuming you use SimpleParser
    from mpsfm.vars import gvars # For ROOT path if model paths are relative
except ImportError as e:
    print("Could not import MP-SfM modules. Make sure MP-SfM is installed or in your PYTHONPATH.")
    print(f"Import error: {e}")
    print("If running from outside the mpsfm root, you might need to adjust sys.path or install the package.")
    # Example:
    # import sys
    # sys.path.append(str(Path(__file__).parent.parent)) # If script is in a 'scripts' subdir of mpsfm root
    # from mpsfm.extraction.imagewise.geometry.base import main as extract_geometry_main
    # ...
    exit(1)


def run_single_geometry_extraction(config_path, data_directory, image_dir_name="images", intrinsics_filename="intrinsics.yaml", output_dir="custom_extraction_output", image_list_file=None, overwrite=True):
    """
    Runs a single geometry extraction based on a config file.
    """
    cfg_path = Path(config_path)
    data_dir = Path(data_directory)
    images_actual_dir = data_dir / image_dir_name
    intrinsics_actual_path = data_dir / intrinsics_filename
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not cfg_path.exists():
        print(f"Error: Configuration file not found at {cfg_path}")
        return
    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        return
    if not images_actual_dir.exists():
        print(f"Error: Image directory not found at {images_actual_dir}")
        return
    if not intrinsics_actual_path.exists():
        print(f"Error: Intrinsics file not found at {intrinsics_actual_path}")
        return

    # Load the YAML configuration for the extraction
    conf_dict = yaml.safe_load(cfg_path.read_text())
    extraction_conf = OmegaConf.create(conf_dict)

    # Ensure model weights directory is correctly pointed to
    # BaseModel expects conf.model.models_dir
    # If 'models_dir' in conf.model is relative, make it relative to MP-SfM project ROOT
    model_weights_dir_str = extraction_conf.model.get("models_dir", "local/weights")
    model_weights_dir = Path(model_weights_dir_str)
    if not model_weights_dir.is_absolute():
         extraction_conf.model.models_dir = str(gvars.ROOT / model_weights_dir_str)
         print(f"Resolved model weights directory to: {extraction_conf.model.models_dir}")


    if image_list_file:
        image_list_p = Path(image_list_file)
        if not image_list_p.exists():
            print(f"Error: Image list file not found at {image_list_p}")
            return
        image_names_to_process = [line.strip() for line in image_list_p.read_text().splitlines() if line.strip()]
        print(f"Processing {len(image_names_to_process)} images from list: {image_list_file}")
    else:
        image_names_to_process = [f.name for f in images_actual_dir.iterdir() if f.is_file() and f.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        print(f"Found {len(image_names_to_process)} images in directory: {images_actual_dir}")

    if not image_names_to_process:
        print("No images to process.")
        return

    scene_parser = SimpleParser(
        data_dir=data_dir,
        imnames=image_names_to_process,
        intrinsics_pth=intrinsics_actual_path,
        rgb_dir=images_actual_dir
    )

    h5_output_file_path, _ = extract_geometry_main(
        conf=extraction_conf,
        export_dir=output_path,
        overwrite=overwrite,
        image_list=image_names_to_process,
        scene_parser=scene_parser,
        verbose=extraction_conf.get("verbose", 0)
    )
    print(f"Extraction complete. Output saved to: {h5_output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run standalone geometry extraction for MP-SfM.")
    parser.add_argument("--config", type=str, default="extract_depth_metric3d.yaml",
                        help="Path to the YAML configuration file for extraction.")
    parser.add_argument("--data_dir", type=str, default="local/example/",
                        help="Path to the main data directory.")
    parser.add_argument("--image_dir", type=str, default="images",
                        help="Subdirectory name for images within data_dir.")
    parser.add_argument("--intrinsics_file", type=str, default="intrinsics.yaml",
                        help="Filename for intrinsics within data_dir.")
    parser.add_argument("--output_dir", type=str, default="custom_extraction_output",
                        help="Directory to save the HDF5 output file.")
    parser.add_argument("--image_list", type=str, default=None,
                        help="Optional path to a text file listing image basenames to process (one per line). Processes all images in image_dir if not provided.")
    parser.add_argument("--no_overwrite", action="store_false", dest="overwrite",
                        help="Disable overwriting if HDF5 output for an image already exists.")

    args = parser.parse_args()

    try:
        run_single_geometry_extraction(
            config_path=args.config,
            data_directory=args.data_dir,
            image_dir_name=args.image_dir,
            intrinsics_filename=args.intrinsics_file,
            output_dir=args.output_dir,
            image_list_file=args.image_list,
            overwrite=args.overwrite
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
