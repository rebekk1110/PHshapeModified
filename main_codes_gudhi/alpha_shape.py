import os
import sys
import numpy as np
from alphashape import alphashape
from shapely.geometry import MultiPolygon, Polygon, Point, box
import logging
import geopandas as gpd
from shapely.validation import make_valid
from shapely.ops import unary_union
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import rasterio
from rasterio.warp import transform_geom
from rasterio.crs import CRS

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mdl_io import load_raster, save_buildings, load_buildings, get_raster_path, get_output_folder

def preprocess_points(points, eps=1.5, min_samples=4):
    """Preprocess points using DBSCAN clustering to remove noise and group nearby points"""
    logging.info("Preprocessing points with DBSCAN clustering")
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    
    clusters = {}
    for i, label in enumerate(labels):
        if label != -1:  # Ignore noise points
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(points[i])
    
    return clusters

def process_cluster(points, alpha):
    """Process a single cluster of points to create an alpha shape"""
    if len(points) < 3:
        return None
    
    try:
        points_array = np.array(points)
        shape = alphashape(points_array, alpha)
        
        if shape is not None:
            shape = make_valid(shape)
            return shape
    except Exception as e:
        logging.warning(f"Error processing cluster: {str(e)}")
    
    return None

def split_large_polygons(polygons, max_area=600):
    """Split large polygons into smaller ones"""
    result = []
    for poly in polygons:
        if poly.area > max_area:
            # Split the polygon using its bounding box
            minx, miny, maxx, maxy = poly.bounds
            midx, midy = (minx + maxx) / 2, (miny + maxy) / 2
            
            # Create four smaller polygons
            top_left = Polygon([(minx, midy), (midx, midy), (midx, maxy), (minx, maxy)])
            top_right = Polygon([(midx, midy), (maxx, midy), (maxx, maxy), (midx, maxy)])
            bottom_left = Polygon([(minx, miny), (midx, miny), (midx, midy), (minx, midy)])
            bottom_right = Polygon([(midx, miny), (maxx, miny), (maxx, midy), (midx, midy)])
            
            # Intersect with the original polygon and add to results
            for small_poly in [top_left, top_right, bottom_left, bottom_right]:
                intersection = poly.intersection(small_poly)
                if not intersection.is_empty:
                    result.extend(split_large_polygons([intersection], max_area))
        else:
            result.append(poly)
    return result

def alpha_shape_detection(raster_path, tile_name, params):
    """Improved alpha shape detection with preprocessing and clustering"""
    logging.info(f"Starting Alpha Shape detection for {raster_path}")
    
    # Load raster data
    with rasterio.open(raster_path) as src:
        image = src.read(1)
        transform = src.transform
        raster_crs = src.crs
        bounds = src.bounds
    
    logging.info(f"Raster CRS: {raster_crs}")
    logging.info(f"Raster bounds: {bounds}")
    
    # Extract non-zero data points
    mask = image != 0
    rows, cols = np.where(mask)
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    points = np.column_stack((xs, ys))
    
    logging.info(f"Number of non-zero points: {len(points)}")
    logging.info(f"Sample points: {points[:5]}")
    
    if len(points) < 3:
        logging.warning(f"Not enough points found in {raster_path}")
        return gpd.GeoDataFrame(geometry=[], crs=raster_crs)
    
    # Preprocess points using DBSCAN clustering
    clusters = preprocess_points(points, eps=params['eps'], min_samples=params['min_samples'])
    logging.info(f"Found {len(clusters)} point clusters")
    
    # Process each cluster
    buildings = []
    for i, cluster_points in enumerate(clusters.values()):
        shape = process_cluster(cluster_points, params['alpha_value'])
        if shape is not None:
            if isinstance(shape, Polygon):
                buildings.append(shape)
            elif isinstance(shape, MultiPolygon):
                buildings.extend(list(shape.geoms))
        logging.info(f"Processed cluster {i+1}: resulting shape type: {type(shape)}")
    
    # Split large polygons
    buildings = split_large_polygons(buildings, max_area=params['max_area'])
    
    # Filter by area and buffer
    buildings = [b.buffer(params['buffer_distance']) for b in buildings if b.area >= params['min_area']]
    
    # Merge overlapping buildings
    if buildings:
        buildings = list(unary_union(buildings).geoms)
    
    logging.info(f"Detected {len(buildings)} buildings")
    
    # Create GeoDataFrame with the correct CRS
    gdf = gpd.GeoDataFrame(geometry=buildings, crs=raster_crs)
    
    # Clip the geometries to the raster bounds
    gdf = gdf.clip(box(*bounds))
    
    # Ensure the GeoDataFrame is in the correct CRS
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)
    
    logging.info(f"Final GeoDataFrame CRS: {gdf.crs}")
    logging.info(f"Final GeoDataFrame bounds: {gdf.total_bounds}")
    
    return gdf

def process_tile_alpha(raster_path, out_folder, tile_name, params):
    logging.info(f"Processing tile {tile_name} with alpha shape method")
    
    buildings_gdf = alpha_shape_detection(raster_path, tile_name, params)
    
    # Save to file with parameters in filename
    output_file = os.path.join(
        out_folder, 
        f"{tile_name}_alpha_simplified_buildings.geojson"
    )
    buildings_gdf.to_file(output_file, driver="GeoJSON")
    logging.info(f"Saved {len(buildings_gdf)} buildings to {output_file}")
    
    return buildings_gdf

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_tile = "tile_0_7"
    
    params = {
        'alpha_value': 0.85,
        'min_area': 3,
        'buffer_distance': 0.1,
        'eps': 1.5,
        'min_samples': 4,
        'max_area': 600
    }
    
    logging.info(f"Testing with parameters: {params}")
    
    raster_path = get_raster_path(test_tile)
    out_folder = get_output_folder()
    
    buildings_gdf = main_alpha(raster_path, out_folder, test_tile, params)
    print(f"Number of buildings detected: {len(buildings_gdf)}")

