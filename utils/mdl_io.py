"""
@File           : mdl_io.py
@Author         : Gefei Kong (modified by Assistant)
@Time           : Current Date
------------------------------------------------------------------------------------------------------------------------
@Description    : Input/Output operations, path management, and tile selection
"""

import os
import numpy as np
import rasterio
import pickle
import json
import geopandas as gpd
import random
import logging
import pprint
import yaml

# Base project path
BASE_PATH = "/Users/Rebekka/GiHub/PHshapeModified"

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def load_raster(file_path):
    with rasterio.open(file_path) as src:
        image = src.read(1)  # Read the first band
        return image, src.transform

def load_data(file_path):
    if file_path.endswith('.tif'):
        return load_raster(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f)


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_shp(shp_path):
    gdf = gpd.read_file(shp_path)
    gdf['area'] = gdf.geometry.area
    return gdf
def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print("Configuration loaded successfully. Contents:")
        pprint.pprint(config)
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {str(e)}")
        return None
    except IOError as e:
        print(f"Error reading config file: {str(e)}")
        return None

def get_tile_types(config):
    try:
        return [tile_group['name'] for tile_group in config['data']['input']['specific_tiles']]
    except KeyError as e:
        logging.error(f"KeyError in get_tile_types: {str(e)}")
        logging.error("Unable to find 'specific_tiles' in the configuration. Check the config file structure.")
        return []
    except TypeError as e:
        logging.error(f"TypeError in get_tile_types: {str(e)}")
        logging.error("Unexpected type in configuration. 'specific_tiles' should be a list.")
        return []


def get_tile_list(config, tile_type=None, specific_tiles=None, random_tile=False):
    try:
        all_tiles = []
        for tile_group in config['data']['input']['specific_tiles']:
            all_tiles.extend(tile_group['files'])
        
        if random_tile:
            return [random.choice(all_tiles)]
        
        if specific_tiles:
            return [tile for tile in specific_tiles if tile in all_tiles]
        
        if tile_type:
            for tile_group in config['data']['input']['specific_tiles']:
                if tile_group['name'] == tile_type:
                    return tile_group['files']
        
        return all_tiles
    except KeyError as e:
        logging.error(f"KeyError in get_tile_list: {str(e)}")
        return []
    except TypeError as e:
        logging.error(f"TypeError in get_tile_list: {str(e)}")
        logging.error("Unexpected type in configuration. Check the structure of 'specific_tiles'.")
        return []

def get_output_folder(algorithm, is_special=False):
    base_folder = "output"
    if is_special:
        return os.path.join(base_folder, "special", algorithm)
    else:
        return os.path.join(base_folder, algorithm)


def get_raster_path(tile_name):
    return os.path.join(BASE_PATH, "output", "tiles", "tif_tiles", f"{tile_name}.tif")

def get_output_folder(module, is_random=False):
    if module in ['mdl1', 'mdl2', 'evaluation', 'visualizations']:
        return get_specific_output_folder(module, is_random)
    return os.path.join(BASE_PATH, "output", f"{module}_output")

def get_specific_output_folder(module, is_random=False):
    base_folder = "rand" if is_random else "spes"
    return os.path.join(BASE_PATH, "output", base_folder, module)

def get_gt_shp_path(tile_name):
    return os.path.join(BASE_PATH, "output", "tiles", "shp_tiles", f"{tile_name}.shp")

def save_buildings(buildings, output_folder, tile_name):
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, f"{tile_name}_buildings.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump(buildings, f)
    print(f"Buildings saved to {file_path}")

def load_buildings(input_folder, tile_name):
    file_path = os.path.join(input_folder, f"{tile_name}_buildings.pkl")
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            buildings = pickle.load(f)
        print(f"Buildings loaded from {file_path}")
        return buildings
    else:
        print(f"No saved buildings found for {tile_name}")
        return None

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)



