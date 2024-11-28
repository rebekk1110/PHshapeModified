import os
import json
import rasterio
import geopandas as gpd
from tqdm import tqdm
from shapely.geometry import mapping
import traceback
import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from utils.config_loader import get_config
from utils.utils import get_project_paths, create_directory

# Now import the local modules
from main_codes_gudhi.mdl1_bolPH_gu import get_building_outlines_from_raster
from main_codes_gudhi.mdl2_simp_bol import main_simp_ol
from main_codes_gudhi.vis_comp import visualize_tile

def process_tile(tile_path, output_folder, vis_folder, gt_shapefile_dir, building_counter, config):
    try:
        with rasterio.open(tile_path) as src:
            raster_data = src.read(1)  # Read the first band
            transform = src.transform
            raster_crs = src.crs

            # Detect buildings
            buildings = get_building_outlines_from_raster(raster_data, transform)
            print(f"Detected {len(buildings)} buildings in tile {os.path.basename(tile_path)}")

            if buildings:
                # Simplify outlines
                bld_list = [os.path.splitext(os.path.basename(tile_path))[0]]
                simplified_buildings = main_simp_ol(buildings, config, bld_list=bld_list)
                print(f"Simplified {len(simplified_buildings)} buildings")

                # Assign unique IDs and create GeoJSON features
                features = []
                for building in simplified_buildings:
                    feature = {
                        "type": "Feature",
                        "geometry": mapping(building),
                        "properties": {"id": f"B{building_counter}"}
                    }
                    features.append(feature)
                    building_counter += 1

                # Create GeoDataFrame from features
                gdf = gpd.GeoDataFrame.from_features(features)
                gdf.set_crs(raster_crs, inplace=True)

                # Check if all elements have the same CRS
                if not all(gdf.crs == raster_crs for geom in gdf.geometry):
                    print(f"WARNING: Not all elements in tile {os.path.basename(tile_path)} have the same CRS")

                # Save as JSON
                output_name = f"{os.path.splitext(os.path.basename(tile_path))[0]}_buildings.json"
                output_path = os.path.join(output_folder, output_name)
                with open(output_path, 'w') as f:
                    json.dump({
                        "type": "FeatureCollection",
                        "features": features,
                        "crs": str(raster_crs)
                    }, f, indent=2)
                print(f"Saved JSON file: {output_path}")

                # Create visualization
                vis_name = f"{os.path.splitext(os.path.basename(tile_path))[0]}_visualization.png"
                vis_path = os.path.join(vis_folder, vis_name)
                
                # Get the corresponding ground truth shapefile
                gt_shapefile_name = f"{os.path.splitext(os.path.basename(tile_path))[0]}.shp"
                gt_shapefile_path = os.path.join(gt_shapefile_dir, gt_shapefile_name)
                
                if os.path.exists(gt_shapefile_path):
                    visualize_tile(raster_data, transform, features, gt_shapefile_path, vis_path, config, tile_path)
                    print(f"Created visualization with ground truth: {vis_path}")
                else:
                    print(f"Ground truth shapefile not found: {gt_shapefile_path}")
                    visualize_tile(raster_data, transform, features, None, vis_path, config, tile_path)
                    print(f"Created visualization without ground truth: {vis_path}")

                return len(simplified_buildings)
            else:
                return 0
    except Exception as e:
        print(f"Error processing tile {tile_path}: {str(e)}")
        traceback.print_exc()
        return 0

def main(cfg_path):
    # Load configuration
    config = get_config(cfg_path)

    # Get project paths
    tif_tiles_dir = config['data']['output']['out_tif_tiles_folder']
    output_folder = config['data']['output']['out_simp_folder']
    vis_folder = config['data']['output']['out_vis_folder']
    gt_shapefile_dir = config['data']['input']['shapefile_folder']

    # Print paths for debugging
    print(f"TIF tiles directory: {tif_tiles_dir}")
    print(f"Output folder: {output_folder}")
    print(f"Visualization folder: {vis_folder}")
    print(f"Ground truth shapefile directory: {gt_shapefile_dir}")

    # Add output_folder to config
    config['output_folder'] = output_folder

    # Create output folders if they don't exist
    create_directory(output_folder)
    create_directory(vis_folder)

    # Initialize total buildings counter
    total_buildings = 0

    # Process each tile
    tile_files = [f for f in os.listdir(tif_tiles_dir) if f.endswith('.tif')]

    for tile_file in tqdm(tile_files, desc="Processing tiles"):
        tile_path = os.path.join(tif_tiles_dir, tile_file)
        buildings_in_tile = process_tile(tile_path, output_folder, vis_folder, gt_shapefile_dir, total_buildings, config)
        total_buildings += buildings_in_tile

    # Save total building count
    try:
        with open(os.path.join(output_folder, 'building_count.txt'), 'w') as f:
            f.write(f"Total buildings detected: {total_buildings}")
        print(f"Total buildings detected: {total_buildings}")
    except Exception as e:
        print(f"Error saving building count: {str(e)}")

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config_raster.yaml")
    main(config_path)