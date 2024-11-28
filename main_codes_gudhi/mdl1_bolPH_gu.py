"""
@File           : mdl1_bolPH_gu.py
@Author         : Gefei Kong (modified by Rebekka)
@Time           : 27.11.2024
------------------------------------------------------------------------------------------------------------------------
@Description    : Get basic building outlines from raster data using image processing and persistent homology, new version

"""
import os
import sys
import time
import numpy as np
from shapely.geometry import Polygon
from scipy import ndimage
import matplotlib.pyplot as plt
import rasterio
from tqdm import tqdm
import logging
import pandas as pd


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mdl_io import load_raster, save_json
from utils.mdl_geo import poly2Geojson
from utils.mdl_procs import pre_downsampling
from modules.get_basic_ol_v2_gu import get_autooptim_bf_radius_GU, get_build_bf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def plot_buildings(raster_data, buildings, output_path, transform):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(raster_data, cmap='gray', alpha=0.5, extent=[0, raster_data.shape[1], raster_data.shape[0], 0])

    for building in buildings:
        coords = np.array(building.exterior.coords)
        rows, cols = rasterio.transform.rowcol(transform, coords[:, 0], coords[:, 1])
        ax.plot(cols, rows, color='red', linewidth=2)

    ax.set_title(f'Detected Buildings (Total: {len(buildings)})')
    ax.set_xlim(0, raster_data.shape[1])
    ax.set_ylim(raster_data.shape[0], 0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_initial_separation(raster_data, labeled_buildings, output_path, transform):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot original raster
    ax1.imshow(raster_data, cmap='gray', extent=[0, raster_data.shape[1], raster_data.shape[0], 0])
    ax1.set_title('Original Raster')
    ax1.set_xlim(0, raster_data.shape[1])
    ax1.set_ylim(raster_data.shape[0], 0)
    
    # Plot labeled buildings
    unique_labels = np.unique(labeled_buildings)
    num_labels = len(unique_labels) - 1  # Subtract 1 to exclude background
    cmap = plt.get_cmap('tab20')
    labeled_buildings_colored = np.zeros((*labeled_buildings.shape, 3))
    
    for i, label in enumerate(unique_labels[1:]):  # Skip background (0)
        mask = labeled_buildings == label
        color = cmap(i / num_labels)[:3]
        labeled_buildings_colored[mask] = color
    
    ax2.imshow(labeled_buildings_colored, extent=[0, raster_data.shape[1], raster_data.shape[0], 0])
    ax2.set_title(f'Separated Buildings (Total: {num_labels})')
    ax2.set_xlim(0, raster_data.shape[1])
    ax2.set_ylim(raster_data.shape[0], 0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Initial separation plot saved to {output_path}")


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


def main_basicOL(raster_path, out_folder, down_sample_num=500, bfr_tole=5e-1, bfr_otdiff=1e-2, 
                 is_use_saved_bfr=False, savename_bfr="", is_unrefresh_save=False, is_Debug=False):
    start_time = time.time()
    os.makedirs(out_folder, exist_ok=True)
    tile_name = os.path.splitext(os.path.basename(raster_path))[0]

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


    plot_buildings(raster_data, buildings, os.path.join(out_folder, f"{tile_name}_buildings.png"), transform)

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
    raster_path = "/Users/Rebekka/GiHub/PHshapeModified/output/tiles/tif_tiles/tile_test_1.tif"
    out_folder = "/Users/Rebekka/GiHub/PHshapeModified/output/visualizations"
    savename_bfr = "/Users/Rebekka/GiHub/PHshapeModified/output/buffer_radii.csv"

    detected_buildings = main_basicOL(raster_path, out_folder, 
                                      is_use_saved_bfr=False,  # Changed to False since we're not using saved buffer radii
                                      savename_bfr=savename_bfr, 
                                      is_unrefresh_save=True, 
                                      is_Debug=True)
    logging.info(f"Number of buildings detected: {len(detected_buildings)}")