"""
@File           : mdl2_eval.py
@Author         : Gefei Kong (modified by Rebekka)
@Time           : Current Date
------------------------------------------------------------------------------------------------------------------------
@Description    : Evaluation module for mdl2 using shp files as ground truth
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import logging


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.eval_basic_ol import load_shp_GT, load_result_polygons, intersection_union, hausdorff_dis, hausdorff_dis_v2
from utils.mdl_io import load_json
from utils.mdl_geo import obj2Geo

def load_shp_GT(shp_gt_path, tile_name):
    gdf = gpd.read_file(shp_gt_path)
    logging.info(f"Columns in the shapefile: {gdf.columns}")
    
    if 'tile' in gdf.columns:
        gdf = gdf[gdf['tile'] == tile_name]
    else:
        logging.warning("'tile' column not found in the shapefile. Using all geometries.")
    
    logging.info(f"Number of geometries loaded: {len(gdf)}")
    return gdf

def load_result_polygons(res_folder, res_type, tile_name):
    polygons = []
    for file in os.listdir(res_folder):
        if file.startswith(tile_name) and file.endswith(res_type):
            file_path = os.path.join(res_folder, file)
            data = load_json(file_path)
            poly = obj2Geo(data)
            polygons.append(poly)
    return polygons

def filter_overlapping_buildings(gt_gdf, result_polygons):
    overlapping_gt = []
    for idx, gt_geom in gt_gdf.geometry.items():
        if any(result_poly.intersects(gt_geom) for result_poly in result_polygons):
            overlapping_gt.append(idx)
    
    filtered_gt = gt_gdf.loc[overlapping_gt]
    logging.info(f"Filtered out {len(gt_gdf) - len(filtered_gt)} non-overlapping ground truth buildings")
    return filtered_gt

def main_eval(res_folder, res_type, gt_shp_path, out_folder, tile_name, is_save_res=True, use_v2_hausdorff=False):
    os.makedirs(out_folder, exist_ok=True)

    # Load ground truth data
    poly_gt_eval = load_shp_GT(gt_shp_path, tile_name)

    # Load result polygons
    result_polygons = load_result_polygons(res_folder, res_type, tile_name)

    # Filter out non-overlapping ground truth buildings
    poly_gt_eval = filter_overlapping_buildings(poly_gt_eval, result_polygons)

    results = []
    for i, poly in enumerate(tqdm(result_polygons, desc="Evaluating buildings")):
        iou = intersection_union(poly, poly_gt_eval, i)
        
        if use_v2_hausdorff:
            hd = hausdorff_dis_v2(poly, poly_gt_eval, i)
        else:
            hd = hausdorff_dis(poly, poly_gt_eval, i)
        
        results.append([i+1, poly, iou, hd])

    res_df = pd.DataFrame(results, columns=["bid", "geometry", "IOU", "HD"])
    res_df = res_df.dropna()

    logging.info(f"Mean IOU: {res_df['IOU'].mean():.4f}")
    logging.info(f"Mean HD: {res_df['HD'].mean():.4f}")

    if is_save_res:
        savename = os.path.join(out_folder, f"{tile_name}_evaluation")
        res_df.to_csv(f"{savename}.csv", index=False)
        gdf = gpd.GeoDataFrame(res_df, geometry=res_df.geometry)
        gdf.to_file(f"{savename}.shp")

    return res_df

if __name__ == "__main__":
    tile_name = "tile_7_10"  # Replace with your tile name
    mdl2_out_folder = get_output_folder("mdl2")
    eval_out_folder = get_output_folder("eval")
    gt_shp_path = get_gt_shp_path(tile_name)

    eval_results = main_eval(res_folder=mdl2_out_folder,
                             res_type=".json",
                             gt_shp_path=gt_shp_path,
                             out_folder=eval_out_folder,
                             tile_name=tile_name,
                             use_v2_hausdorff=True)

    print(f"Evaluation Results for {tile_name}:")
    print(f"Mean IoU: {eval_results['IOU'].mean():.4f}")
    print(f"Mean Hausdorff Distance: {eval_results['HD'].mean():.4f}")