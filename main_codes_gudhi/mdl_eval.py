"""
@File           : mdl_eval.py
@Author         : Gefei Kong
@Time:          : 18.05.2023 14:15
------------------------------------------------------------------------------------------------------------------------
@Description    : as below
evaluation module
"""
import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from modules.eval_basic_ol import intersection_union, hausdorff_dis_v2

def load_ground_truth(shp_gt_path, bld_list):
    gdf = gpd.read_file(shp_gt_path)
    return gdf[gdf['id'].isin(bld_list)]

def evaluate_building(pred_poly, gt_poly):
    iou = intersection_union(pred_poly, gt_poly)
    hd = hausdorff_dis_v2(pred_poly, gt_poly)
    return iou, hd

def main_eval(res_folder, res_type, shp_gt_path, dataset_type, out_folder, res_base, bld_list, is_save_res):
    gt_data = load_ground_truth(shp_gt_path, bld_list)
    
    results = []
    for bldi in bld_list:
        pred_path = os.path.join(res_folder, f"{bldi}{res_type}")
        with open(pred_path, 'r') as f:
            pred_data = json.load(f)
        
        pred_poly = Polygon(pred_data['coordinates'][0])
        gt_poly = gt_data[gt_data['id'] == bldi].geometry.iloc[0]
        
        iou, hd = evaluate_building(pred_poly, gt_poly)
        results.append([bldi, pred_poly, iou, hd])
    
    res_df = pd.DataFrame(results, columns=["bid", "geometry", "IOU", "HD"])
    
    print(f"{dataset_type}'s mean_IOU: {res_df['IOU'].mean()}")
    print(f"{dataset_type}'s mean_HD: {res_df['HD'].mean()}")
    
    if is_save_res:
        savename = os.path.join(out_folder, f"{dataset_type}_{res_base}.shp")
        res_overall_json = {"mean_IoU": res_df['IOU'].mean(), "mean_HD": res_df['HD'].mean()}
        with open(savename.replace(".shp", "_overall.json"), "w") as res_js:
            json.dump(res_overall_json, res_js, indent=4)
        res_df.to_csv(savename.replace(".shp", ".csv"), index=False)
        gdf = gpd.GeoDataFrame(res_df, geometry=res_df.geometry)
        gdf.to_file(savename)

    return res_df