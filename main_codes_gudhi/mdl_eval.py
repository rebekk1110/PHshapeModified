"""
@File           : mdl_eval.py
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
from shapely.wkt import loads, dumps
import seaborn as sns
import logging
import glob
from shapely.geometry import Polygon, MultiPolygon, shape

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.eval_basic_ol import load_result_polygons,load_shp_GT
from utils.mdl_io import load_json, get_specific_output_folder
from utils.mdl_geo import obj2Geo
from utils.mdl_visual import plot_gt_vs_simplified

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def calculate_metrics(pred_poly, gt_poly):
    """Calculate evaluation metrics between predicted and ground truth polygons"""
    try:
        iou = intersection_union(pred_poly, gt_poly)
        hd = hausdorff_dis(pred_poly, gt_poly)
        area = pred_poly.area
        perimeter = pred_poly.length
        return iou, hd, area, perimeter
    except Exception as e:
        logging.error(f"Error calculating metrics: {str(e)}")
        logging.debug(f"Predicted polygon: {pred_poly.wkt}")
        logging.debug(f"Ground truth polygon: {gt_poly.wkt}")
        return None, None, None, None

def intersection_union(polygon1, polygon2):
    """Calculate Intersection over Union (IoU) for two polygons"""
    try:
        if not polygon1.is_valid:
            polygon1 = polygon1.buffer(0)
        if not polygon2.is_valid:
            polygon2 = polygon2.buffer(0)
        intersection = polygon1.intersection(polygon2).area
        union = polygon1.union(polygon2).area
        return intersection / union if union > 0 else 0
    except Exception as e:
        logging.error(f"Error calculating IoU: {str(e)}")
        return 0


def hausdorff_dis(polygon1, polygon2):
    """Calculate Hausdorff distance between two polygons"""
    return polygon1.hausdorff_distance(polygon2)

def main_eval(res_folder, res_type, gt_shp_path, out_folder, tile_name, is_save_res=True, use_v2_hausdorff=False, is_random=False):
    # Update the output folder based on whether it's a random tile or not
    out_folder = get_specific_output_folder("evaluation", is_random)
    os.makedirs(out_folder, exist_ok=True)

    # Load ground truth data
    poly_gt_eval = load_shp_GT(gt_shp_path, tile_name)
    if poly_gt_eval is None:
        logging.error(f"Failed to load ground truth data for {tile_name}")
        return None
    logging.info(f"Loaded {len(poly_gt_eval)} ground truth polygons")

    # Load result polygons
    result_polygons = load_result_polygons(res_folder, res_type, tile_name)
    logging.info(f"Loaded {len(result_polygons)} result polygons")

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
        else:
            logging.warning(f"No valid metrics found for building {i+1}")
            logging.debug(f"Predicted polygon for building {i+1}: {pred_poly.wkt}")
            logging.debug(f"Ground truth polygons: {[p.wkt for p in poly_gt_eval.geometry]}")

    res_df = pd.DataFrame(results, columns=["bid", "geometry", "IOU", "HD", "Area", "Perimeter"])
    res_df = res_df.dropna()  # Remove any rows with None values

    if not res_df.empty:
        logging.info(f"Number of buildings evaluated: {len(res_df)}")
        logging.info(f"Mean IOU: {res_df['IOU'].mean():.4f}")
        logging.info(f"Mean HD: {res_df['HD'].mean():.4f}")
        logging.info(f"Mean Area: {res_df['Area'].mean():.4f}")
        logging.info(f"Mean Perimeter: {res_df['Perimeter'].mean():.4f}")

        if is_save_res:
            savename = os.path.join(out_folder, f"{tile_name}_evaluation")
            
            # Save as CSV with WKT geometries
            res_df['geometry_wkt'] = res_df['geometry'].apply(lambda geom: dumps(geom))
            res_df.drop('geometry', axis=1).to_csv(f"{savename}_with_wkt.csv", index=False)

            try:
                # Try to save as GeoJSON
                gdf = gpd.GeoDataFrame(res_df, geometry='geometry')
                gdf.to_file(f"{savename}.geojson", driver='GeoJSON')
                logging.info(f"Evaluation results saved to {savename}.geojson")
            except Exception as e:
                logging.warning(f"Unable to save as GeoJSON due to: {str(e)}")

        # Generate visualization
        gt_gdf = gpd.read_file(gt_shp_path)
        simplified_gdf = gpd.GeoDataFrame(res_df, geometry='geometry')
        vis_folder = os.path.join(out_folder, 'visualizations')
        os.makedirs(vis_folder, exist_ok=True)
        vis_path = os.path.join(vis_folder, f"{tile_name}_gt_vs_simplified_with_stats.png")
        plot_gt_vs_simplified(gt_gdf, simplified_gdf, res_df, vis_path, tile_name)
        logging.info(f"Visualization saved to {vis_path}")

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
        print(f"Number of buildings evaluated: {len(eval_results)}")
        print(f"Mean IoU: {eval_results['IOU'].mean():.4f}")
        print(f"Mean Hausdorff Distance: {eval_results['HD'].mean():.4f}")
        print(f"Mean Area: {eval_results['Area'].mean():.4f}")
        print(f"Mean Perimeter: {eval_results['Perimeter'].mean():.4f}")
    else:
        print("No valid evaluation results.")

