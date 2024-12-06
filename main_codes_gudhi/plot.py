import geopandas as gpd
import logging

logging.basicConfig(level=logging.INFO)

def check_building_coordinates(geojson_path):
    gdf = gpd.read_file(geojson_path)
    logging.info(f"Total buildings: {len(gdf)}")
    
    for idx, row in gdf.iterrows():
        centroid = row.geometry.centroid
        logging.info(f"Building {idx + 1}: Centroid coordinates: ({centroid.x}, {centroid.y})")
        logging.info(f"Building {idx + 1}: Bounding box: {row.geometry.bounds}")

    logging.info(f"Overall bounding box: {gdf.total_bounds}")

# Usage
alpha_results_path = "/Users/Rebekka/GiHub/PHshapeModified/output/spes/evaluation/tile_26_9_alpha_evaluation.geojson"
check_building_coordinates(alpha_results_path)

