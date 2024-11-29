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
from shapely.wkt import dumps  # Add this import
import seaborn as sns
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.eval_basic_ol import load_shp_GT, load_result_polygons, intersection_union, hausdorff_dis, hausdorff_dis_v2, calculate_metrics
from utils.mdl_io import load_json
from utils.mdl_geo import obj2Geo

def main_eval(res_folder, res_type, gt_shp_path, out_folder, tile_name, is_save_res=True, use_v2_hausdorff=False):
    os.makedirs(out_folder, exist_ok=True)

    # Load ground truth data
    poly_gt_eval = gpd.read_file(gt_shp_path)

    # Load result polygons
    result_polygons = load_result_polygons(res_folder, res_type, tile_name)

    results = []
    for i, pred_poly in enumerate(tqdm(result_polygons, desc="Evaluating buildings")):
        best_iou = 0
        best_metrics = None
        for _, gt_poly in poly_gt_eval.iterrows():
            metrics = calculate_metrics(pred_poly, gt_poly.geometry)
            if metrics[0] is not None and metrics[0] > best_iou:
                best_iou = metrics[0]
                best_metrics = metrics
        
        if best_metrics:
            results.append([i+1, pred_poly] + list(best_metrics))

    res_df = pd.DataFrame(results, columns=["bid", "geometry", "IOU", "HD", "Area", "Perimeter"])
    res_df = res_df.dropna()  # Remove any rows with None values

    if not res_df.empty:
        logging.info(f"Mean IOU: {res_df['IOU'].mean():.4f}")
        logging.info(f"Mean HD: {res_df['HD'].mean():.4f}")
        logging.info(f"Mean Area: {res_df['Area'].mean():.4f}")
        logging.info(f"Mean Perimeter: {res_df['Perimeter'].mean():.4f}")

        if is_save_res:
            savename = os.path.join(out_folder, f"{tile_name}_evaluation")
            
            # Save as CSV with WKT geometries
            res_df['geometry_wkt'] = res_df['geometry'].apply(lambda geom: dumps(geom))
            res_df.drop('geometry', axis=1).to_csv(f"{savename}_with_wkt.csv", index=False)
            logging.info(f"Evaluation results with WKT geometries saved to {savename}_with_wkt.csv")

            try:
                # Try to save as GeoJSON
                gdf = gpd.GeoDataFrame(res_df, geometry='geometry')
                gdf.to_file(f"{savename}.geojson", driver='GeoJSON')
                logging.info(f"Evaluation results saved to {savename}.geojson")
            except Exception as e:
                logging.warning(f"Unable to save as GeoJSON due to: {str(e)}")
    else:
        logging.warning("No valid evaluation results. The DataFrame is empty.")

    return res_df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Add test code here if needed
    tile_name = "tile_26_9"  # Replace with your tile name
    mdl2_out_folder = "path/to/mdl2_output"  # Replace with actual path
    eval_out_folder = "path/to/eval_output"  # Replace with actual path
    gt_shp_path = "path/to/ground_truth.shp"  # Replace with actual path

    eval_results = main_eval(res_folder=mdl2_out_folder,
                             res_type=".json",
                             gt_shp_path=gt_shp_path,
                             out_folder=eval_out_folder,
                             tile_name=tile_name,
                             use_v2_hausdorff=True)

    if eval_results is not None and not eval_results.empty:
        print(f"Evaluation Results for {tile_name}:")
        print(f"Mean IoU: {eval_results['IOU'].mean():.4f}")
        print(f"Mean Hausdorff Distance: {eval_results['HD'].mean():.4f}")
        print(f"Mean Area: {eval_results['Area'].mean():.4f}")
        print(f"Mean Perimeter: {eval_results['Perimeter'].mean():.4f}")
    else:
        print("No valid evaluation results.")
