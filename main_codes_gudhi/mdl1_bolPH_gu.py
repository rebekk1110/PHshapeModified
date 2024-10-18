import json
import os
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.mdl_geo import poly2Geojson
from modules.get_basic_ol_v2_gu import get_autooptim_bf_radius_GU, get_build_bf

def load_tif(file_path):
    with rasterio.open(file_path) as src:
        # Read the first band (assuming it's a single-band image)
        data = src.read(1)
        
        # Get coordinates
        height, width = data.shape
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        
        # Create a 2D array of coordinates
        coords = np.column_stack((xs.ravel(), ys.ravel()))
        
        return coords, data

def visualize_result(data, polygon, output_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(data, cmap='viridis')
    
    # Plot the polygon
    x, y = polygon.exterior.xy
    plt.plot(x, y, color='red', linewidth=2)
    
    plt.title('Building Outline')
    plt.colorbar(label='Elevation')
    plt.savefig(output_path)
    plt.close()

def main_basicOL(tif_file_path:str,
                 out_folder:str,
                 bfr_tole:float=5e-1,
                 bfr_otdiff:float=1e-2,
                 is_Debug:bool=False):

    os.makedirs(out_folder, exist_ok=True)

    # Read TIF data
    bldi_2d, bldi_data = load_tif(tif_file_path)
    if is_Debug:
        print(f"Input data shape: {bldi_2d.shape}")

    # Get auto-optimized buffer radius
    bldi_bfr_optim, bldi_bfr_0d, bldi_bfr_1d, bldi_pers_1d \
        = get_autooptim_bf_radius_GU(bldi_2d, is_down=False, isDebug=is_Debug)

    if is_Debug:
        print(f"Optimized buffer radius: {bldi_bfr_optim}")

    # Get buffer polygon
    bldi_bf_optnew, bldi_bf_optim = get_build_bf(bldi_2d, bfr_optim=bldi_bfr_optim, bf_tole=bfr_tole, bf_otdiff=bfr_otdiff, isDebug=is_Debug)

    # Save result as Geojson
    bldi_olpoly_json = poly2Geojson(bldi_bf_optnew, round_precision=6)
    json_output_path = os.path.join(out_folder, "building_outline.json")
    with open(json_output_path, "w+") as sf:
        json.dump(bldi_olpoly_json, sf, indent=4)

    # Visualize the result
    visualize_result(bldi_data, bldi_bf_optnew, os.path.join(out_folder, "building_outline_visualization.png"))

    if is_Debug:
        print(f"Results saved in {out_folder}")

if __name__ == "__main__":
    tif_file_path = '/Users/Rebekka/GiHub/PHshapeModified/main_codes_gudhi/test_images/trd_1_crop_188_3-kopi.tif'
    out_folder = 'output'
    
    main_basicOL(tif_file_path, out_folder, is_Debug=True)