"""
@File           : mdl_io.py
@Author         : Gefei Kong
@Time:          : 20.04.2023 18:07
------------------------------------------------------------------------------------------------------------------------
@Description    : as below

"""

##Changed to raster from point cloud

import os
import numpy as np
import rasterio

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
    import json
    with open(file_path, 'w') as f:
        json.dump(data, f)

def load_json(file_path):
    import json
    with open(file_path, 'r') as f:
        return json.load(f)

def load_shp(shp_path):
    gdf = gpd.read_file(shp_path)
    gdf['area'] = gdf.geometry.area
    return gdf
