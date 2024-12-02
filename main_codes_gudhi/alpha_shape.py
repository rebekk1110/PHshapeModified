import os
import sys
import numpy as np
import alphashape
import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
import logging
from shapely.validation import make_valid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mdl_io import load_raster, save_buildings, load_buildings, get_raster_path, get_output_folder

def alpha_shape_detection(raster_path, alpha=1.0, min_area=10):
    logging.info(f"Starting Alpha Shape detection for {raster_path}")
    
    # Load raster data
    image, transform = load_raster(raster_path)

    # Extract non-zero data points and convert to geographic coordinates
    mask = image != 0
    coords = np.column_stack(np.where(mask))
    points = np.array([(transform * (x, y))[0:2] for y, x in coords])

    if len(points) < 3:
        logging.warning(f"Not enough points found in {raster_path} for Alpha Shape detection")
        return []

    # Apply alpha shape for building outlines
    alpha_shape = alphashape.alphashape(points, alpha)

    # Convert to list of Polygon objects and filter small areas
    buildings = []
    if isinstance(alpha_shape, MultiPolygon):
        for geom in alpha_shape.geoms:
            if geom.area >= min_area:
                valid_geom = make_valid(geom)
                if valid_geom.geom_type == 'Polygon':
                    buildings.append(valid_geom)
                elif valid_geom.geom_type == 'MultiPolygon':
                    buildings.extend(list(valid_geom.geoms))
    elif isinstance(alpha_shape, Polygon):
        if alpha_shape.area >= min_area:
            valid_geom = make_valid(alpha_shape)
            if valid_geom.geom_type == 'Polygon':
                buildings.append(valid_geom)
            elif valid_geom.geom_type == 'MultiPolygon':
                buildings.extend(list(valid_geom.geoms))

    logging.info(f"Detected {len(buildings)} buildings using Alpha Shape algorithm")
    return buildings

def process_tile_alpha(tile_name, force_rerun=False, alpha_value=1.0, min_area=10):
    raster_path = get_raster_path(tile_name)
    alpha_out_folder = get_output_folder("alpha")
    output_file = os.path.join(alpha_out_folder, f"{tile_name}_buildings.geojson")
    
    if force_rerun or not os.path.exists(output_file):
        logging.info(f"Running Alpha Shape detection for {tile_name}")
        buildings = alpha_shape_detection(raster_path, alpha=alpha_value, min_area=min_area)
        if buildings:
            gdf = gpd.GeoDataFrame(geometry=buildings)
            gdf.to_file(output_file, driver='GeoJSON')
            logging.info(f"Saved {len(buildings)} buildings for {tile_name} to {output_file}")
        else:
            logging.warning(f"No buildings detected for {tile_name}")
    else:
        logging.info(f"Alpha Shape output found. Loading existing buildings for {tile_name}")
        gdf = gpd.read_file(output_file)
        buildings = gdf.geometry.tolist()
        logging.info(f"Loaded {len(buildings)} existing buildings for {tile_name} from {output_file}")
    
    return buildings

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_tile = "tile_26_9"
    buildings = process_tile_alpha(test_tile, force_rerun=True)
    print(f"Number of buildings detected: {len(buildings)}")