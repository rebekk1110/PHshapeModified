"""
@File           : mdl1_bolPH_gu.py
@Author         : Gefei Kong (modified by Rebekka)
@Time           : Current Date
------------------------------------------------------------------------------------------------------------------------
@Description    : Get basic building outlines from raster data using image processing and persistent homology
"""

import os
import sys
import time
import numpy as np
from shapely.geometry import Polygon
from scipy import ndimage
from tqdm import tqdm
import logging
import pandas as pd
import rasterio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mdl_io import load_raster, save_json, get_raster_path, get_output_folder
from utils.mdl_geo import poly2Geojson
from utils.mdl_procs import pre_downsampling
from utils.mdl_visual import plot_buildings, plot_initial_separation
from modules.get_basic_ol_v2_gu import get_autooptim_bf_radius_GU, get_build_bf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def separate_buildings(raster_data, min_size=100):
    binary = raster_data > 0
    labeled, _ = ndimage.label(binary)
    sizes = ndimage.sum(binary, labeled, range(1, labeled.max() + 1))
    mask_sizes = sizes < min_size
    labeled[np.isin(labeled, np.where(mask_sizes)[0] + 1)] = 0
    return ndimage.label(labeled > 0)[0]

def extract_building_polygon(labeled_buildings, label):
    coords = np.column_stack(np.where(labeled_buildings == label))
    return coords if coords.shape[0] >= 3 else None

def process_building(label, labeled_buildings, transform, down_sample_num, bfr_tole, bfr_otdiff, 
                     is_Debug, all_bfr_optim, is_use_saved_bfr):
    try:
        building_coords = extract_building_polygon(labeled_buildings, label)
        if building_coords is None:
            return None, None

        xs, ys = rasterio.transform.xy(transform, building_coords[:, 0], building_coords[:, 1])
        building_coords = np.column_stack((xs, ys))

        if is_Debug:
            logging.info(f"Building {label} coordinates shape: {building_coords.shape}")

        pre_cloud_num = 5000
        if building_coords.shape[0] > pre_cloud_num:
            building_coords, _ = pre_downsampling(building_coords, target_num=pre_cloud_num, 
                                                  start_voxel_size=0.5, isDebug=is_Debug)
            if is_Debug:
                logging.info(f"Pre-downsampled building {label} coordinates shape: {building_coords.shape}")

        if (not is_use_saved_bfr) or (label not in all_bfr_optim):
            bfr_optim, _, _, _ = get_autooptim_bf_radius_GU(building_coords, down_sample_num=down_sample_num, 
                                                            is_down=True, isDebug=is_Debug)
            all_bfr_optim[label] = bfr_optim
        else:
            bfr_optim = all_bfr_optim[label]
            if is_Debug:
                logging.info(f"Using saved bfr_optim for building {label}: {bfr_optim}")

        bf_optnew, _ = get_build_bf(building_coords, bfr_optim=bfr_optim, bf_tole=bfr_tole, 
                                    bf_otdiff=bfr_otdiff, isDebug=is_Debug)

        return bf_optnew, label
    except Exception as e:
        logging.error(f"Error processing building {label}: {str(e)}")
        return None, None
        

def main_basicOL(raster_path, out_folder, tile_name, down_sample_num=450, bfr_tole=5e-1, bfr_otdiff=1e-2, 
                 is_use_saved_bfr=False, savename_bfr="", is_unrefresh_save=False, is_Debug=False):
    start_time = time.time()
    os.makedirs(out_folder, exist_ok=True)

    raster_data, transform = load_raster(raster_path)
    if is_Debug:
        logging.info(f"Raster shape: {raster_data.shape}")
        logging.info(f"Transform: {transform}")

    labeled_buildings = separate_buildings(raster_data)

    # Plot initial separation
    initial_separation_plot_path = os.path.join(out_folder, f"{tile_name}_initial_separation.png")
    plot_initial_separation(raster_data, labeled_buildings, initial_separation_plot_path, transform)
    
    building_labels = np.unique(labeled_buildings)[1:]

    all_bfr_optim = {}
    if is_use_saved_bfr and os.path.exists(savename_bfr):
        all_bfr_optim = pd.read_csv(savename_bfr, index_col='label').to_dict()['bfr_optim']
        if is_Debug:
            logging.info("Loaded saved buffer radii")

    buildings = []
    for label in tqdm(building_labels, desc="Processing buildings"):
        savename = os.path.join(out_folder, f"{tile_name}_building_{label}.json")
        if is_unrefresh_save and os.path.exists(savename):
            continue

        building, processed_label = process_building(label, labeled_buildings, transform, down_sample_num, 
                                                     bfr_tole, bfr_otdiff, is_Debug, all_bfr_optim, is_use_saved_bfr)
        if building is not None:
            buildings.append(building)
            building_json = poly2Geojson(building, round_precision=6)
            save_json(building_json, savename)

    if buildings:
        plot_buildings(raster_data, buildings, os.path.join(out_folder, f"{tile_name}_buildings.png"), transform, tile_name)

    if not is_use_saved_bfr and all_bfr_optim:  
        if savename_bfr:
            df = pd.DataFrame.from_dict(all_bfr_optim, orient='index', columns=['bfr_optim'])
            df.index.name = 'label'
            df.to_csv(savename_bfr)
            logging.info(f"Saved buffer radii to {savename_bfr}")
        else:
            logging.warning("Buffer radii were not saved because savename_bfr was not provided.")

    logging.info(f"Total buildings detected: {len(buildings)}")
    logging.info(f"Total execution time: {time.time() - start_time:.2f} seconds")
    return buildings

if __name__ == "__main__":
    tile_name = "tile_test_1"
    raster_path = get_raster_path(tile_name)
    out_folder = get_output_folder("mdl1")
    savename_bfr = os.path.join(os.path.dirname(out_folder), "buffer_radii.csv")

    detected_buildings = main_basicOL(raster_path, out_folder, tile_name,
                                      is_use_saved_bfr=False,
                                      savename_bfr=savename_bfr, 
                                      is_unrefresh_save=True, 
                                      is_Debug=True)
    logging.info(f"Number of buildings detected: {len(detected_buildings)}")