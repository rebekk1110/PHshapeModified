import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import os
import json
import numpy as np
from shapely.measurement import hausdorff_distance
from shapely.ops import unary_union
import logging


'''
gammel load 
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

'''
def load_shp_GT(shp_gt_path, tile_name):
    gdf = gpd.read_file(shp_gt_path)
    logging.info(f"Columns in the shapefile: {gdf.columns}")
    
    if 'tile' in gdf.columns:
        gdf = gdf[gdf['tile'] == tile_name]
    else:
        logging.warning("'tile' column not found in the shapefile. Using all geometries.")
    gdf = gdf.reset_index(drop=True)
    logging.info(f"Number of geometries loaded: {len(gdf)}")
    return gdf

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

##vuredere å endre noe i alpha som får den til å lagre i UTF-8
def load_result_polygons(res_folder, res_type, tile_name):
    polygons = []
    for file in os.listdir(res_folder):
        if file.startswith(tile_name) and file.endswith(res_type):
            file_path = os.path.join(res_folder, file)
            try:
                if res_type.lower() == '.json':
                    # Try UTF-8 first
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                    except UnicodeDecodeError:
                        # If UTF-8 fails, try with 'latin-1' encoding
                        with open(file_path, 'r', encoding='latin-1') as f:
                            data = json.load(f)
                    
                    if isinstance(data, dict) and 'coordinates' in data:
                        poly = Polygon(data['coordinates'][0])
                        polygons.append(poly)
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and 'coordinates' in item:
                                poly = Polygon(item['coordinates'][0])
                                polygons.append(poly)
                    
                elif res_type.lower() in ['.shp', '.geojson']:
                    gdf = gpd.read_file(file_path)
                    for geom in gdf.geometry:
                        if isinstance(geom, Polygon):
                            polygons.append(geom)
                        elif isinstance(geom, MultiPolygon):
                            polygons.extend(list(geom.geoms))
                else:
                    logging.warning(f"Unsupported file type: {res_type}")
                    continue
            except Exception as e:
                logging.error(f"Error loading file {file_path}: {str(e)}")
    
    logging.info(f"Loaded {len(polygons)} polygons from {res_folder}")
    return polygons



def calculate_metrics(pred_poly: Polygon, gt_poly: Polygon) -> tuple:
    # Attempt to fix invalid geometries
    if not pred_poly.is_valid:
        pred_poly = pred_poly.buffer(0)
    if not gt_poly.is_valid:
        gt_poly = gt_poly.buffer(0)
    
    # Check if the geometries are still invalid
    if not pred_poly.is_valid or not gt_poly.is_valid:
        logging.warning("Unable to fix invalid geometry. Skipping this comparison.")
        return None, None, None, None

    try:
        intersection = pred_poly.intersection(gt_poly).area
        union = pred_poly.union(gt_poly).area
        iou = intersection / union if union > 0 else 0
        hd = pred_poly.hausdorff_distance(gt_poly)
        area = pred_poly.area
        perimeter = pred_poly.length
        return iou, hd, area, perimeter
    except Exception as e:
        logging.error(f"Error calculating metrics: {str(e)}")
        return None, None, None, None

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