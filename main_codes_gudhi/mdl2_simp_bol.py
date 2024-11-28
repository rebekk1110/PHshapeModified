import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import rasterio
from shapely.geometry import Polygon, MultiPolygon


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from modules.simp_basic_ol import simp_poly_Fd, stop_by_IoU, simp_poly_Extmtd
from utils.mdl_io import load_raster, load_json
from utils.mdl_geo import obj2Geo, get_PolygonCoords_withInter, arr2Geo, poly2Geojson
from utils.mdl_visual import drawmultipolygon, show_ifd_shape
from mdl1_bolPH_gu import main_basicOL
from mdl_eval import main_eval
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Base project path
BASE_PATH = "/Users/Rebekka/GiHub/PHshapeModified"

def get_raster_path(tile_name):
    path = os.path.join(BASE_PATH, "output", "tiles", "tif_tiles", f"{tile_name}.tif")
    logging.info(f"Raster path: {path}")
    return path

def get_mdl1_out_folder():
    path = os.path.join(BASE_PATH, "output", "mdl1_output")
    logging.info(f"MDL1 output folder: {path}")
    return path

def get_mdl2_out_folder():
    path = os.path.join(BASE_PATH, "output", "mdl2_output")
    logging.info(f"MDL2 output folder: {path}")
    return path

def get_gt_shp_path(tile_name):
    path = os.path.join(BASE_PATH, "output", "tiles", "shp_tiles",f"{tile_name}.shp")
    logging.info(f"Ground truth shapefile path: {path}")
    return path
def check_mdl1_output(mdl1_out_folder, tile_name):
    # Check if any files for this tile exist in the mdl1 output folder
    files = [f for f in os.listdir(mdl1_out_folder) if f.startswith(tile_name) and f.endswith('.json')]
    return len(files) > 0

def check_mdl2_output(mdl2_out_folder, tile_name):
    # Check if the simplified buildings file exists
    return os.path.exists(os.path.join(mdl2_out_folder, f"{tile_name}_buildings_comparison.png"))

def main_simp_ol(buildings, out_folder, tile_name, bfr_tole=0.5, bfr_otdiff=0.0, simp_method="haus", is_save_fig=False, is_Debug=False):
    if not buildings:
        logging.warning("No buildings to simplify.")
        return []

    os.makedirs(out_folder, exist_ok=True)
    simplified_buildings = []

    for i, building in enumerate(tqdm(buildings, desc="Simplifying buildings")):
        try:
            if simp_method == "haus":
                thres_haus = (1 - np.cos(30 / 180 * np.pi)) * (bfr_tole - bfr_otdiff)
                b_oli_simp_ext, b_oli_simp_ints, b_oli_simp = simp_poly_Fd(building, thres_mode="haus", thres_haus=thres_haus, isDebug=is_Debug)
            elif simp_method == "iou":
                thres_simparea = 0.95  # You may want to make this configurable
                b_oli_simp_ext, b_oli_simp_ints, b_oli_simp = simp_poly_Fd(building, thres_mode="iou", thres_simparea=thres_simparea, isDebug=is_Debug)
            else:
                b_oli_simp_ext, b_oli_simp_ints, b_oli_simp = simp_poly_Extmtd(building, bfr_otdiff=bfr_otdiff, bfr_tole=bfr_tole)

            simplified_buildings.append(b_oli_simp)

            # Save simplified building
            b_oli_simp_json = poly2Geojson(b_oli_simp, round_precision=6)
            savename = os.path.join(out_folder, f"{tile_name}_simplified_building_{i+1}.json")
            with open(savename, "w+") as sf:
                json.dump(b_oli_simp_json, sf, indent=4)

            if is_save_fig:
                out_path_fig = os.path.join(out_folder, 'figures', f"{tile_name}_simplified_building_{i+1}")
                os.makedirs(out_path_fig, exist_ok=True)
                drawmultipolygon(b_oli_simp, title=f"{tile_name} Simplified Building {i+1}",
                                 savepath=os.path.join(out_path_fig, f"{tile_name}_simplified_building_{i+1}"))
        except Exception as e:
            logging.error(f"Error processing building {i+1}: {str(e)}")

    return simplified_buildings

def plot_simplified_buildings(raster_data, original_buildings, simplified_buildings, output_path, transform, tile_name):
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot original buildings
        ax1.imshow(raster_data, cmap='gray', alpha=0.5, extent=[0, raster_data.shape[1], raster_data.shape[0], 0])
        for building in original_buildings:
            coords = np.array(building.exterior.coords)
            rows, cols = rasterio.transform.rowcol(transform, coords[:, 0], coords[:, 1])
            ax1.plot(cols, rows, color='red', linewidth=2)
        ax1.set_title(f'{tile_name} Original Buildings (Total: {len(original_buildings)})')
        ax1.set_xlim(0, raster_data.shape[1])
        ax1.set_ylim(raster_data.shape[0], 0)

        # Plot simplified buildings
        ax2.imshow(raster_data, cmap='gray', alpha=0.5, extent=[0, raster_data.shape[1], raster_data.shape[0], 0])
        for building in simplified_buildings:
            coords = np.array(building.exterior.coords)
            rows, cols = rasterio.transform.rowcol(transform, coords[:, 0], coords[:, 1])
            ax2.plot(cols, rows, color='blue', linewidth=2)
        ax2.set_title(f'{tile_name} Simplified Buildings (Total: {len(simplified_buildings)})')
        ax2.set_xlim(0, raster_data.shape[1])
        ax2.set_ylim(raster_data.shape[0], 0)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Comparison plot saved to {output_path}")
    except Exception as e:
        logging.error(f"Error plotting buildings: {str(e)}")

if __name__ == "__main__":
    # Get tile name from user input
    tile_name = ("tile_26_9")
    logging.info(f"Processing tile: {tile_name}")

    try:
        # Set up paths and parameters
        raster_path = get_raster_path(tile_name)
        mdl1_out_folder = get_mdl1_out_folder()
        mdl2_out_folder = get_mdl2_out_folder()
        gt_shp_path = get_gt_shp_path(tile_name)

        # Check for existing mdl1 output
        if check_mdl1_output(mdl1_out_folder, tile_name):
            logging.info("MDL1 output found. Skipping MDL1 processing.")
            # Load existing buildings from mdl1 output
            buildings = []
            for file in os.listdir(mdl1_out_folder):
                if file.startswith(tile_name) and file.endswith('.json'):
                    with open(os.path.join(mdl1_out_folder, file), 'r') as f:
                        building_data = json.load(f)
                        buildings.append(Polygon(building_data['coordinates'][0]))
        else:
            # Run mdl1 to get basic outlines
            logging.info("Running MDL1 to get basic outlines...")
            buildings = main_basicOL(raster_path, mdl1_out_folder, is_Debug=True)
        
        logging.info(f"Number of buildings detected: {len(buildings)}")

        if not buildings:
            logging.warning("No buildings detected. Exiting.")
            sys.exit(0)

        # Check for existing mdl2 output
        if check_mdl2_output(mdl2_out_folder, tile_name):
            logging.info("MDL2 output found. Skipping MDL2 processing.")
            # Load existing simplified buildings
            simplified_buildings = []
            for file in os.listdir(mdl2_out_folder):
                if file.startswith(tile_name) and file.endswith('_simplified_building.json'):
                    with open(os.path.join(mdl2_out_folder, file), 'r') as f:
                        building_data = json.load(f)
                        simplified_buildings.append(Polygon(building_data['coordinates'][0]))
        else:
            # Run mdl2 to simplify the outlines
            logging.info("Running MDL2 to simplify the outlines...")
            simplified_buildings = main_simp_ol(buildings, mdl2_out_folder, tile_name, is_Debug=True)

            # Plot comparison
            logging.info("Plotting comparison...")
            with rasterio.open(raster_path) as src:
                raster_data = src.read(1)
                transform = src.transform
            
            plot_simplified_buildings(raster_data, buildings, simplified_buildings, 
                                      os.path.join(mdl2_out_folder, f"{tile_name}_buildings_comparison.png"), transform, tile_name)

        # Run evaluation
        logging.info("Running evaluation...")
        eval_results = main_eval(res_folder=mdl2_out_folder,
                                 res_type=".json",
                                 shp_gt_path=gt_shp_path,
                                 out_folder=os.path.join(mdl2_out_folder, "eval_results"),
                                 tile_name=tile_name,
                                 use_v2_hausdorff=True)

        logging.info("Evaluation complete. Results saved in the output folder.")
        logging.info(f"Mean IoU: {eval_results['IOU'].mean():.4f}")
        logging.info(f"Mean Hausdorff Distance: {eval_results['HD'].mean():.4f}")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise