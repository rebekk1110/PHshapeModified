import os 
import sys
import numpy as np
import alphashape
from shapely.geometry import MultiPolygon, Polygon

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mdl_io import load_raster, save_buildings, load_buildings
import logging

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
    if isinstance(alpha_shape, MultiPolygon):
        buildings = [geom for geom in alpha_shape.geoms if geom.area >= min_area]
    elif isinstance(alpha_shape, Polygon):
        buildings = [alpha_shape] if alpha_shape.area >= min_area else []
    else:
        buildings = []

    logging.info(f"Detected {len(buildings)} buildings using Alpha Shape algorithm")
    return buildings

def process_tile_alpha(tile_name, force_rerun=False):
    from utils.mdl_io import get_raster_path, get_output_folder
    
    raster_path = get_raster_path(tile_name)
    alpha_out_folder = get_output_folder("alpha")
    
    if force_rerun or not load_buildings(alpha_out_folder, tile_name):
        logging.info(f"Running Alpha Shape detection for {tile_name}")
        buildings = alpha_shape_detection(raster_path)
        if buildings:
            save_buildings(buildings, alpha_out_folder, tile_name)
            logging.info(f"Saved {len(buildings)} buildings for {tile_name}")
        else:
            logging.warning(f"No buildings detected for {tile_name}")
    else:
        buildings = load_buildings(alpha_out_folder, tile_name)
        logging.info(f"Loaded {len(buildings)} existing buildings for {tile_name}")
    
    return buildings

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_tile = "tile_26_9"
    buildings = process_tile_alpha(test_tile, force_rerun=True)
    print(f"Number of buildings detected: {len(buildings)}")

