
##new main_all_gu

import os
import sys
import logging
import geopandas as gpd
from shapely.geometry import Polygon

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdl1_bolPH_gu import main_basicOL
from mdl2_simp_bol import main_simp_ol
from mdl_eval import main_eval
from utils.mdl_io import get_raster_path, get_output_folder, get_gt_shp_path, load_raster
from utils.mdl_visual import plot_buildings, plot_simplified_buildings, plot_gt_vs_simplified
from AlphaAlgorithm import alpha_shape_detection  # Import the new algorithm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_existing_output(out_folder, tile_name, file_pattern):
    return any(f.startswith(tile_name) and f.endswith(file_pattern) for f in os.listdir(out_folder))

def main_all_gu(tile_name, force_rerun=False, algorithm='mdl1'):
    raster_path = get_raster_path(tile_name)
    mdl1_out_folder = get_output_folder("mdl1")
    mdl2_out_folder = get_output_folder("mdl2")
    alpha_out_folder = get_output_folder("alpha")
    eval_out_folder = get_output_folder("eval")
    gt_shp_path = get_gt_shp_path(tile_name)

    if algorithm == 'mdl1':
        # MDL1: Basic Outline Detection
        if force_rerun or not check_existing_output(mdl1_out_folder, tile_name, '.json'):
            logging.info("Running MDL1 to get basic outlines...")
            buildings = main_basicOL(raster_path, mdl1_out_folder, tile_name, is_Debug=True)
        else:
            logging.info("MDL1 output found. Loading existing buildings...")
            buildings = [Polygon(gpd.read_file(os.path.join(mdl1_out_folder, f)).geometry[0]) 
                         for f in os.listdir(mdl1_out_folder) 
                         if f.startswith(tile_name) and f.endswith('.json')]

        # MDL2: Simplification
        if force_rerun or not check_existing_output(mdl2_out_folder, tile_name, '_simplified_building.json'):
            logging.info("Running MDL2 to simplify the outlines...")
            simplified_buildings = main_simp_ol(buildings, mdl2_out_folder, tile_name, raster_path, is_Debug=True)
            raster_data, transform = load_raster(raster_path)
            plot_simplified_buildings(raster_data, buildings, simplified_buildings, 
                                      os.path.join(mdl2_out_folder, f"{tile_name}_buildings_comparison.png"), transform, tile_name)
        else:
            logging.info("MDL2 output found. Loading simplified buildings...")
            simplified_buildings = [Polygon(gpd.read_file(os.path.join(mdl2_out_folder, f)).geometry[0]) 
                                    for f in os.listdir(mdl2_out_folder) 
                                    if f.startswith(tile_name) and f.endswith('_simplified_building.json')]

        eval_folder = mdl2_out_folder

    elif algorithm == 'alpha':
        # Alpha Shape Algorithm
        if force_rerun or not check_existing_output(alpha_out_folder, tile_name, '.json'):
            logging.info("Running Alpha Shape algorithm to detect buildings...")
            buildings = alpha_shape_detection(raster_path)
            # Save the detected buildings (you may need to implement this function)
            save_buildings(buildings, alpha_out_folder, tile_name)
        else:
            logging.info("Alpha Shape output found. Loading existing buildings...")
            buildings = load_buildings(alpha_out_folder, tile_name)

        simplified_buildings = buildings  # No simplification step for Alpha Shape
        eval_folder = alpha_out_folder

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    if not simplified_buildings:
        logging.warning("No buildings detected. Skipping evaluation.")
        return None

    # Evaluation
    logging.info("Running evaluation...")
    eval_results = main_eval(res_folder=eval_folder,
                             res_type=".json",
                             gt_shp_path=gt_shp_path,
                             out_folder=eval_out_folder,
                             tile_name=tile_name,
                             use_v2_hausdorff=True)

    # Plot GT vs Simplified with statistics
    gt_gdf = gpd.read_file(gt_shp_path)
    simplified_gdf = gpd.GeoDataFrame(geometry=simplified_buildings)
    plot_gt_vs_simplified(gt_gdf, simplified_gdf, eval_results, 
                          os.path.join(eval_out_folder, f"{tile_name}_{algorithm}_gt_vs_simplified_with_stats.png"), tile_name)

    logging.info("Processing complete.")
    return eval_results

if __name__ == "__main__":
    tile_name = input("Enter the tile name (e.g., tile_26_9): ")
    force_rerun = input("Force rerun of all steps? (y/n): ").lower() == 'y'
    algorithm = input("Choose algorithm (mdl1/alpha): ").lower()
    
    results = main_all_gu(tile_name, force_rerun, algorithm)
    if results is not None:
        print(f"Evaluation Results for {tile_name} using {algorithm}:")
        print(f"Mean IoU: {results['IOU'].mean():.4f}")
        print(f"Mean Hausdorff Distance: {results['HD'].mean():.4f}")
    else:
        print("Processing completed with no results.")