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

def plot_gt_vs_simp(gt_gdf, simp_gdf, output_path):
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot ground truth buildings
    gt_gdf.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2, label='Ground Truth')
    
    # Plot simplified buildings
    simp_gdf.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=2, label='Simplified')
    
    ax.set_title('Ground Truth vs Simplified Buildings')
    ax.legend()
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Comparison plot saved to {output_path}")

def main_eval(res_folder, res_type, shp_gt_path, out_folder, tile_name, is_save_res=True, use_v2_hausdorff=False):
    os.makedirs(out_folder, exist_ok=True)

    # Load ground truth data
    poly_gt_eval = load_shp_GT(shp_gt_path, tile_name)

    # Load result polygons
    result_polygons = load_result_polygons(res_folder, res_type, tile_name)

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

        summary = {
            "mean_IoU": res_df['IOU'].mean(),
            "mean_HD": res_df['HD'].mean()
        }
        with open(f"{savename}_summary.json", "w") as f:
            json.dump(summary, f, indent=4)

        # Create and save the comparison plot
        simp_gdf = gpd.GeoDataFrame(geometry=result_polygons)
        plot_gt_vs_simp(poly_gt_eval, simp_gdf, os.path.join(out_folder, f"{tile_name}_gt_vs_simp_comparison.png"))

    return res_df