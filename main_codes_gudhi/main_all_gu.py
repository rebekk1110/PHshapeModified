"""
@File           : main_all_gu.py
@Author         : Assistant
@Time           : Current Date
------------------------------------------------------------------------------------------------------------------------
@Description    : Main script to run the entire building outline detection and simplification pipeline
"""

import os
import sys
import logging
import traceback 
import pprint
import pandas as pd
import geopandas as gpd

from shapely.geometry import Polygon

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'config.yaml')


# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdl1_bolPH_gu import main_basicOL
from mdl2_simp_bol import main_simp_ol
from mdl_eval import main_eval
from utils.mdl_io import get_raster_path
from alpha_shape import process_tile_alpha

from utils.mdl_io import (get_raster_path, get_output_folder, get_gt_shp_path, load_raster, 
                          save_buildings, load_buildings, load_config, get_tile_list, get_tile_types)

from utils.mdl_visual import plot_buildings, plot_simplified_buildings, plot_gt_vs_simplified

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
            mdl1_output_file = os.path.join(mdl1_out_folder, f"{tile_name}_buildings.json")
            mdl2_output_file = os.path.join(mdl2_out_folder, f"{tile_name}_simplified_buildings.json")

            logging.info(f"Checking for existing PH-Shape output for {tile_name}")
            logging.info(f"MDL1 output file: {mdl1_output_file}")
            logging.info(f"File exists: {os.path.exists(mdl1_output_file)}")

            if force_rerun or not os.path.exists(mdl1_output_file):
                logging.info(f"Running PH-Shape detection for {tile_name}")
                buildings = main_basicOL(raster_path, mdl1_out_folder, tile_name, 
                                         down_sample_num=config['params']['down_sample_factor'],
                                         bfr_tole=config['params']['bfr_tole'],
                                         bfr_otdiff=config['params']['bfr_otdiff'],
                                         is_Debug=True,
                                         is_random=is_random,
                                         pre_cloud_num=config['params']['pre_raster_size'],
                                         is_use_saved_bfr=config['params'].get('use_saved_bfr', False),
                                         savename_bfr=os.path.join(mdl1_out_folder, f"{tile_name}_buffer_radii.csv"),
                                         is_unrefresh_save=config['params'].get('unrefresh_save', False))
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
                logging.info(f"Running simplification for {tile_name}")
                simplified_buildings = main_simp_ol(buildings, mdl2_out_folder, tile_name, raster_path,
                                                    bfr_tole=config['params']['bfr_tole'],
                                                    bfr_otdiff=config['params']['bfr_otdiff'],
                                                    simp_method=config['params']['simp']['type'],
                                                    is_Debug=True,
                                                    is_random=is_random)
                save_buildings(simplified_buildings, mdl2_out_folder, f"{tile_name}_simplified")
            else:
                logging.info(f"Simplified output found. Loading existing simplified buildings for {tile_name}")
                simplified_buildings = load_buildings(mdl2_out_folder, f"{tile_name}_simplified")

            eval_folder = mdl2_out_folder
            res_type = ".json"

        elif algorithm == 'alpha':
            # Alpha Shape Algorithm
            alpha_output_file = os.path.join(alpha_out_folder, f"{tile_name}_buildings.geojson")
            if force_rerun or not os.path.exists(alpha_output_file):
                logging.info(f"Running Alpha Shape detection for {tile_name}")
                buildings = process_tile_alpha(tile_name, force_rerun, 
                                               alpha_value=config['params']['alpha']['alpha_value'], 
                                               min_area=config['params']['alpha']['min_area'])
                if not buildings:
                    logging.warning(f"No buildings detected for {tile_name} using Alpha Shape algorithm.")
                    return None
                
                simplified_buildings = buildings  # No simplification step for Alpha Shape
                
                # Save buildings as GeoJSON
                gdf = gpd.GeoDataFrame(geometry=simplified_buildings)
                gdf.to_file(alpha_output_file, driver='GeoJSON')
            else:
                logging.info(f"Alpha Shape output found. Loading existing buildings for {tile_name}")
                gdf = gpd.read_file(alpha_output_file)
                simplified_buildings = gdf.geometry.tolist()

            logging.info(f"Number of buildings after Alpha Shape detection: {len(simplified_buildings)}")

            eval_folder = alpha_out_folder
            res_type = ".geojson"

        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        if not simplified_buildings:
            logging.warning(f"No buildings to evaluate for {tile_name}.")
            return None

        # Evaluation
        eval_output_file = os.path.join(eval_out_folder, f"{tile_name}_{algorithm}_evaluation.csv")
        if force_rerun or not os.path.exists(eval_output_file):
            logging.info(f"Running evaluation for {tile_name}")
            logging.info(f"Number of buildings before evaluation: {len(simplified_buildings)}")
            eval_results = main_eval(res_folder=eval_folder,
                                     res_type=res_type,
                                     gt_shp_path=gt_shp_path,
                                     out_folder=eval_out_folder,
                                     tile_name=tile_name,
                                     use_v2_hausdorff=True,
                                     is_random=is_random)

            if eval_results is None or eval_results.empty:
                logging.warning(f"No valid evaluation results for {tile_name}.")
                return None

            logging.info(f"Number of buildings after evaluation: {len(eval_results)}")

            # Plot GT vs Simplified with statistics
            gt_gdf = gpd.read_file(gt_shp_path)
            simplified_gdf = gpd.GeoDataFrame(eval_results, geometry='geometry')
            plot_gt_vs_simplified(gt_gdf, simplified_gdf, eval_results, 
                                  os.path.join(eval_out_folder, f"{tile_name}_{algorithm}_gt_vs_simplified_with_stats.png"), tile_name)
        else:
            logging.info(f"Evaluation results found. Loading existing results for {tile_name}")
            eval_results = pd.read_csv(eval_output_file)

        logging.info(f"Processing complete for {tile_name}")
        return eval_results
    except Exception as e:
        logging.error(f"Error processing tile {tile_name} with algorithm {algorithm}: {str(e)}")
        logging.error("Traceback:")
        logging.error(traceback.format_exc())
        return None






def main_all_gu(config_path, force_rerun=False):
    try:
        print(f"Loading configuration from: {config_path}")
        config = load_config(config_path)
        if config is None:
            print("Failed to load configuration. Exiting.")
            return

        print("Configuration loaded successfully. Attempting to access 'data' key:")
        '''
        if 'data' in config:
            print("'data' key found in configuration.")
            if 'input' in config['data']:
                print("'input' key found in config['data'].")
                if 'specific_tiles' in config['data']['input']:
                    print("'specific_tiles' found in config['data']['input'].")
                    print("Contents of 'specific_tiles':")
                    pprint.pprint(config['data']['input']['specific_tiles'])
                else:
                    print("'specific_tiles' not found in config['data']['input'].")
            else:
                print("'input' key not found in config['data'].")
        else:
            print("'data' key not found in configuration.")
        '''
        # Get tile types from config
        tile_types = get_tile_types(config)
        print(f"Retrieved tile types: {tile_types}")

        if not tile_types:
            print("No tile types found in the configuration. Please check your config file.")
            return

        # Choose tile type(s) to process
        print("Available options:")
        for i, tile_type in enumerate(tile_types, 1):
            print(f"{i}. {tile_type}")
        print(f"{len(tile_types) + 1}. All tile types")
        print(f"{len(tile_types) + 2}. Random tile")
        
        choice = input(f"Enter your choice (1-{len(tile_types) + 2}): ")
        choice = int(choice.strip())

        tiles_to_process = []
        is_random = False

        if choice == len(tile_types) + 2:
            # Random tile option
            all_tiles = get_tile_list(config)
            tiles_to_process = [random.choice(all_tiles)]
            is_random = True
        elif choice == len(tile_types) + 1:
            # All tile types
            tiles_to_process = get_tile_list(config)
        elif 1 <= choice <= len(tile_types):
            # Specific tile type
            tiles_to_process = get_tile_list(config, tile_type=tile_types[choice-1])
        else:
            print("Invalid choice. Exiting.")
            return

        print(f"Tiles to process: {tiles_to_process}")

        if not tiles_to_process:
            print("No tiles to process. Exiting.")
            return

        results = {}
        algorithms = ['ph-shape', 'alpha']

        for tile_name in tqdm(tiles_to_process, desc="Processing tiles"):
            for algorithm in algorithms:
                print(f"Processing {tile_name} with {algorithm} algorithm")
                result = process_tile(tile_name, config, algorithm, force_rerun, is_random=is_random)
                if result is not None:
                    results.setdefault(tile_name, {})[algorithm] = result

        # Print summary results
        for tile_name, tile_results in results.items():
            print(f"\nResults for {tile_name}:")
            for algorithm, result in tile_results.items():
                print(f"  {algorithm}:")
                print(f"    Mean IoU: {result['IOU'].mean():.4f}")
                print(f"    Mean Hausdorff Distance: {result['HD'].mean():.4f}")
                if 'Area' in result.columns:
                    print(f"    Mean Area: {result['Area'].mean():.4f}")
                if 'Perimeter' in result.columns:
                    print(f"    Mean Perimeter: {result['Perimeter'].mean():.4f}")

    except Exception as e:
        print(f"An error occurred in main_all_gu: {str(e)}")
        print("Traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'config.yaml')
    
    print(f"Using config file: {CONFIG_PATH}")

    if not os.path.exists(CONFIG_PATH):
        print(f"Config file not found at: {CONFIG_PATH}")
        sys.exit(1)

    force_rerun = input("Force rerun of all steps? (y/n): ").lower() == 'y'
    
    main_all_gu(CONFIG_PATH, force_rerun)
