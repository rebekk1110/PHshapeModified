import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import os
import json
import numpy as np
from shapely.ops import unary_union
import logging
from shapely.measurement import hausdorff_distance



def load_shp_GT(shp_path, tile_name):
    gdf = gpd.read_file(shp_path)
    logging.info(f"Columns in the shapefile: {gdf.columns}")
    if 'tile' in gdf.columns:
        gdf = gdf[gdf['tile'] == tile_name]
    else:
        logging.warning("'tile' column not found in the shapefile. Using all geometries.")
    
    # Ensure the GeoDataFrame has a unique index
    gdf = gdf.reset_index(drop=True)
    logging.info(f"Number of geometries loaded: {len(gdf)}")
    return gdf


def load_result_polygons(res_folder, res_type, tile_name):
    polygons = []
    for file in os.listdir(res_folder):
        if file.startswith(tile_name) and file.endswith(res_type):
            with open(os.path.join(res_folder, file), 'r') as f:
                data = json.load(f)
                polygons.append(Polygon(data['coordinates'][0]))
    return polygons

def make_valid(polygon):
    # simple_value = 0.5
    # if (not polygon.is_valid):
    buffer_size = 0
    while True:
        if (buffer_size > 2):
            return None
        pp2 = polygon.buffer(buffer_size, cap_style=3)
        if (pp2.geom_type == "Polygon"):
            potential_polygon = Polygon(list(pp2.exterior.coords))
            potential_polygon = potential_polygon.buffer(-buffer_size, cap_style=3)
            return potential_polygon
        else:
            buffer_size = buffer_size + 0.05


def intersection_union(pred_poly: Polygon, poly_gt_eval: gpd.GeoDataFrame, bid: int) -> float or None:
    if pred_poly is None:
        return None

    if not pred_poly.is_valid:
        pred_poly = make_valid(pred_poly)

    # Check if bid is within the valid range of indices
    if bid < 0 or bid >= len(poly_gt_eval):
        logging.warning(f"Building ID {bid} is out of range. Total geometries: {len(poly_gt_eval)}")
        return None

    gt_poly = poly_gt_eval.iloc[bid].geometry
    if gt_poly is None:
        logging.warning(f"No geometry found for building ID {bid}")
        return None

    polygon_intersection = gt_poly.intersection(pred_poly).area
    polygon_union = gt_poly.union(pred_poly).area
    
    if polygon_union == 0:
        logging.warning(f"Union area is zero for building ID {bid}")
        return 0
    
    IOU = polygon_intersection / polygon_union
    return IOU


def hausdorff_dis(pred_poly:Polygon, poly_gt_eval: gpd.GeoDataFrame, bid:float or int or str) -> float or None:
    """
    calculate IoU between gt and pred result.
    Codes is from Jakob.
    :param pred_poly:
    :param poly_gt_eval:
    :param bid:
    :return:
    """
    if(pred_poly is None):
        return None

    if not pred_poly.is_valid:
        pred_poly = make_valid(pred_poly)

    gt_poly = poly_gt_eval.loc[bid].geometry
    hd = gt_poly.hausdorff_distance(pred_poly)
    return hd


def hausdorff_dis_v2(pred_poly: Polygon, poly_gt_eval: gpd.GeoDataFrame, index: int) -> float or None:
    if pred_poly is None:
        return None

    if not pred_poly.is_valid:
        pred_poly = make_valid(pred_poly)

    # Check if index is within the valid range of indices
    if index < 0 or index >= len(poly_gt_eval):
        logging.warning(f"Index {index} is out of range. Total geometries: {len(poly_gt_eval)}")
        return None

    gt_poly = poly_gt_eval.iloc[index].geometry
    if gt_poly is None:
        logging.warning(f"No geometry found for index {index}")
        return None

    hd = hausdorff_distance(gt_poly, pred_poly, densify=0.1)
    return hd