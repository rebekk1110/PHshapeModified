import os
import sys
import logging
import traceback 
import pprint
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'config.yaml')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdl1_bolPH_gu import main_basicOL
from mdl2_simp_bol import main_simp_ol
from mdl_eval import main_eval
from utils.mdl_io import get_raster_path
from alpha_shape import process_tile_alpha

from utils.mdl_io import (get_raster_path, get_output_folder, get_gt_shp_path, load_raster, 
                        save_buildings, load_buildings, load_config, get_tile_list, get_tile_types)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add tqdm import with error handling
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm module not found. Please install it using 'pip install tqdm'")
    # Define a simple replacement for tqdm if it's not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

def check_existing_output(out_folder, tile_name, file_pattern):
    return any(f.startswith(tile_name) and f.endswith(file_pattern) for f in os.listdir(out_folder))

def process_tile(tile_name, config, algorithm, force_rerun, is_random):
    try:
        raster_path = get_raster_path(tile_name)
        
        if is_random:
            out_folder = config['data']['output']['out_rand_folder']['path']
        else:
            out_folder = config['data']['output']['out_spes_folder']['path']

        mdl1_out_folder = get_output_folder("mdl1", is_random)
        mdl2_out_folder = get_output_folder("mdl2", is_random)
        alpha_out_folder = get_output_folder("alpha", is_random)
        eval_out_folder = get_output_folder("evaluation", is_random)
        gt_shp_path = get_gt_shp_path(tile_name)

        os.makedirs(mdl1_out_folder, exist_ok=True)
        os.makedirs(mdl2_out_folder, exist_ok=True)
        os.makedirs(alpha_out_folder, exist_ok=True)
        os.makedirs(eval_out_folder, exist_ok=True)
        
        if algorithm == 'ph-shape':
            # PH-Shape Algorithm (MDL1 + MDL2)
            mdl1_output_file = os.path.join(mdl1_out_folder, f"{tile_name}_buildings.pkl")
            mdl2_output_file = os.path.join(mdl2_out_folder, f"{tile_name}_ph-shape_simplified_buildings.geojson")
            logging.info(f"Checking for existing PH-Shape output for {tile_name}")
            logging.info(f"MDL1 output file: {mdl1_output_file}")
            logging.info(f"File exists: {os.path.exists(mdl1_output_file)}")

            if force_rerun or not os.path.exists(mdl1_output_file):
                logging.info(f"Running PH-Shape detection for {tile_name}")
                buildings = main_basicOL(raster_path, mdl1_out_folder, tile_name, 
                               down_sample_num=config['params']['down_sample_factor'],
                               bfr_tole=config['params']['bfr_tole'],
                               bfr_otdiff=config['params']['bfr_otdiff'],
                               is_use_saved_bfr=config['params'].get('use_saved_bfr', False),
                               savename_bfr=os.path.join(mdl1_out_folder, f"{tile_name}_buffer_radii.csv"),
                               is_unrefresh_save=config['params'].get('unrefresh_save', False),
                               is_Debug=True,
                               is_random=is_random,
                               pre_cloud_num=config['params']['pre_raster_size']
                              )
                save_buildings(buildings, mdl1_out_folder, tile_name)
                logging.info(f"Saved PH-Shape detection results for {tile_name}")
            else:
                logging.info(f"PH-Shape output found. Loading existing buildings for {tile_name}")
                buildings = load_buildings(mdl1_out_folder, tile_name)
                logging.info(f"Loaded {len(buildings)} existing buildings for {tile_name}")

            if not buildings:
                logging.warning(f"No buildings detected for {tile_name}")
                return None

            if force_rerun or not os.path.exists(mdl2_output_file):
                simplified_buildings = main_simp_ol(buildings, mdl2_out_folder, tile_name, raster_path,
                                                    bfr_tole=config['params']['bfr_tole'],
                                                    bfr_otdiff=config['params']['bfr_otdiff'],
                                                    simp_method=config['params']['simp']['type'],
                                                    is_save_fig=False,
                                                    is_Debug=True,
                                                    is_random=is_random)
                # Save as GeoJSON
                gdf = gpd.GeoDataFrame(geometry=simplified_buildings)
                gdf.to_file(os.path.join(mdl2_out_folder, f"{tile_name}_ph-shape_simplified_buildings.geojson"), driver="GeoJSON")
            else:
                logging.info(f"Simplified output found. Loading existing simplified buildings for {tile_name}")
                gdf = gpd.read_file(os.path.join(mdl2_out_folder, f"{tile_name}_ph-shape_simplified_buildings.geojson"))
                simplified_buildings = gdf.geometry.tolist()

            eval_folder = mdl2_out_folder
            res_type = ".geojson"

        elif algorithm == 'alpha':
                # Alpha Shape Algorithm
                alpha_params = config['params']['alpha']
                alpha_output_file = os.path.join(alpha_out_folder, f"{tile_name}_alpha_simplified_buildings.geojson")
                
                if force_rerun or not os.path.exists(alpha_output_file):
                    logging.info(f"Running Alpha Shape detection for {tile_name}")
                    buildings = process_tile_alpha(
                        raster_path=raster_path,
                        out_folder=alpha_out_folder,
                        tile_name=tile_name,
                        params=alpha_params
                    )
                    logging.info(f"Number of buildings after Alpha Shape detection: {len(buildings)}")
                    
                    # Save the buildings to a GeoJSON file
                   # gdf = gpd.GeoDataFrame(geometry=buildings)
                    #gdf.to_file(alpha_output_file, driver="GeoJSON")
                else:
                    logging.info(f"Alpha Shape output found. Loading existing buildings for {tile_name}")
                    buildings = gpd.read_file(alpha_output_file)
                    logging.info(f"Loaded {len(buildings)} existing buildings for {tile_name}")

                eval_folder = alpha_out_folder
                res_type = ".geojson"

        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Evaluation
        eval_output_file = os.path.join(eval_out_folder, f"{tile_name}_{algorithm}_evaluation.csv")
        if force_rerun or not os.path.exists(eval_output_file):
            logging.info(f"Running evaluation for {tile_name} with {algorithm} algorithm")
            eval_results = main_eval(res_folder=eval_folder,
                                     res_type=res_type,
                                     gt_shp_path=gt_shp_path,
                                     out_folder=eval_out_folder,
                                     tile_name=f"{tile_name}_{algorithm}",
                                     use_v2_hausdorff=True,
                                     is_random=is_random)

            if eval_results is None or eval_results.empty:
                logging.warning(f"No valid evaluation results for {tile_name} with {algorithm} algorithm.")
                return None

        else:
            logging.info(f"Evaluation results found. Loading existing results for {tile_name} with {algorithm} algorithm")
            eval_results = pd.read_csv(eval_output_file)

        logging.info(f"Processing complete for {tile_name} with {algorithm} algorithm")
        if eval_results is not None and not eval_results.empty:
            logging.info(f"Evaluation results generated for {tile_name} with {algorithm} algorithm")
            logging.info(f"Number of results: {len(eval_results)}")
        else:
            logging.warning(f"No evaluation results generated for {tile_name} with {algorithm} algorithm")
        return eval_results

    except Exception as e:
        logging.error(f"Error processing tile {tile_name} with algorithm {algorithm}: {str(e)}")
        logging.error("Traceback:")
        logging.error(traceback.format_exc())
        return None

def main_all_gu(config_path, force_rerun=False):
    config = load_config(config_path)
    if not config:
        logging.error("Failed to load configuration. Exiting.")
        return

    tile_types = get_tile_types(config)
    if not tile_types:
        logging.error("Failed to retrieve tile types. Exiting.")
        return

    print("Available tile types:")
    for i, tile_type in enumerate(tile_types, 1):
        print(f"{i}. {tile_type}")
    print(f"{len(tile_types) + 1}. All tile types")
    print(f"{len(tile_types) + 2}. Random tile")

    choice = int(input("Enter your choice (1-8): "))
    
    if choice == len(tile_types) + 1:
        tiles_to_process = get_tile_list(config)
    elif choice == len(tile_types) + 2:
        tiles_to_process = get_tile_list(config, random_tile=True)
    else:
        selected_type = tile_types[choice - 1]
        tiles_to_process = get_tile_list(config, tile_type=selected_type)

    logging.info(f"Tiles to process: {tiles_to_process}")

    results = {}
    for tile_name in tqdm(tiles_to_process, desc="Processing tiles"):
        tile_results = {}
        for algorithm in ['ph-shape', 'alpha']:
            logging.info(f"Processing {tile_name} with {algorithm} algorithm")
            eval_results = process_tile(tile_name, config, algorithm, force_rerun, is_random=(choice == len(tile_types) + 2))
            if eval_results is not None:
                tile_results[algorithm] = {
                    'Mean IoU': eval_results['IOU'].mean(),
                    'Mean Hausdorff Distance': eval_results['HD'].mean(),
                    'Mean Area': eval_results['Area'].mean() if 'Area' in eval_results.columns else None,
                    'Mean Perimeter': eval_results['Perimeter'].mean() if 'Perimeter' in eval_results.columns else None
                }
        results[tile_name] = tile_results

    # Print results
    for tile_name, tile_results in results.items():
        print(f"\nResults for {tile_name}:")
        for algorithm, metrics in tile_results.items():
            print(f"  {algorithm}:")
            for metric, value in metrics.items():
                if value is not None:
                    print(f"    {metric}: {value:.4f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    force_rerun = input("Force rerun of all steps? (y/n): ").lower() == 'y'
    
    main_all_gu(CONFIG_PATH, force_rerun)
