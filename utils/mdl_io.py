"""
@File           : mdl_io.py
@Author         : Gefei Kong (modified by Assistant)
@Time           : Current Date
------------------------------------------------------------------------------------------------------------------------
@Description    : Input/Output operations and path management
"""

import os
import numpy as np
import rasterio
import json
import geopandas as gpd

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

# New path functions
def get_raster_path(tile_name):
    return os.path.join(BASE_PATH, "output", "tiles", "tif_tiles", f"{tile_name}.tif")

def get_output_folder(module):
    return os.path.join(BASE_PATH, "output", f"{module}_output")

def get_gt_shp_path(tile_name):
    return os.path.join(BASE_PATH, "output", "tiles", "shp_tiles", f"{tile_name}.shp")