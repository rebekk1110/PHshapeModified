import os
import sys
import yaml
import json
import rasterio
from .raster_utils import preprocess_raster
from . import mdl1_bolPH_gu
from . import mdl2_simp_bol
from . import mdl_eval
from .visualization import visualize_results,count_buildings

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_raster(file_path):
    with rasterio.open(file_path) as src:
        image = src.read(1)  # Read the first band
        return image, src.transform

def main(cfg):
    os.makedirs(cfg["data"]["output"]["out_simp_folder"], exist_ok=True)
    os.makedirs(cfg["data"]["output"]["out_eval_folder"], exist_ok=True)
    print(f"Created output directories: {cfg['data']['output']['out_simp_folder']}, {cfg['data']['output']['out_eval_folder']}")

    raster_folder = cfg["data"]["input"]["raster_folder"]
    raster_files = cfg["data"]["input"]["raster_files"]

    for raster_file in raster_files:
        raster_path = os.path.join(raster_folder, raster_file)
        raster_image, transform = load_raster(raster_path)
        
        preprocessed_raster = preprocess_raster(raster_image)
        building_outlines = mdl1_bolPH_gu.get_building_outlines_from_raster(preprocessed_raster, transform)

        print(f"Number of building outlines detected: {len(building_outlines)}")
        # Use the base name of the raster file (without extension) for the output JSON
        base_name = os.path.splitext(raster_file)[0]
        
        simplified_outlines = mdl2_simp_bol.main_simp_ol(
            building_outlines,
            out_folder=cfg["data"]["output"]["out_simp_folder"],
            bld_list=[base_name],  # Pass the JSON filename here
            bfr_tole=cfg["params"]["bfr_tole"],
            bfr_otdiff=cfg["params"]["bfr_otdiff"],
            simp_method=cfg["params"]["simp"]["type"]
        )
        
        ph_shape_path = os.path.join(cfg["data"]["output"]["out_simp_folder"], f"{base_name}.json")
        output_path = os.path.join(cfg["data"]["output"]["out_simp_folder"], f"{base_name}_visualization.png")
        
        print(f"Attempting to visualize results:")
        print(f"  Raster path: {raster_path}")
        print(f"  PH shape path: {ph_shape_path}")
        print(f"  Output path: {output_path}")
        
        visualize_results(raster_path, ph_shape_path, output_path)

        num_buildings=count_buildings(ph_shape_path)

    print("Processing completed.")

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config_raster.yaml')
    with open(config_path, "r") as cfg_file:
        cfg = yaml.safe_load(cfg_file)
    main(cfg)