import os
import json
import rasterio
from rasterio.plot import show
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import shape, box
import yaml
from pathlib import Path
import traceback

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def visualize_tile(raster_data, transform, features, gt_shapefile_path, output_path, config, tif_path):
    print("\n--- Visualization Tile Information ---")
    print(f"TIF file: {tif_path}")
    print(f"Output path: {output_path}")
    print(f"Ground truth shapefile path: {gt_shapefile_path}")

    # TIF file information
    with rasterio.open(tif_path) as src:
        print("\nTIF File Details:")
        print(f"  Driver: {src.driver}")
        print(f"  Width: {src.width}")
        print(f"  Height: {src.height}")
        print(f"  Coordinate Reference System: {src.crs}")
        print(f"  Transform: {src.transform}")
        print(f"  Bounds: {src.bounds}")

    fig, ax = plt.subplots(figsize=(10, 10))
    show(raster_data, transform=transform, ax=ax, cmap='gray')

    # Create GeoDataFrame from detected features
    detected_gdf = gpd.GeoDataFrame.from_features(features)
    
    detected_crs = 'PROJCS["NZGD2000_New_Zealand_Transverse_Mercator_2000",GEOGCS["NZGD2000",DATUM["New_Zealand_Geodetic_Datum_2000",SPHEROID["GRS 1980",6378137,298.257222101004,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6167"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",173],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",1600000],PARAMETER["false_northing",10000000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
    detected_gdf.set_crs(detected_crs, inplace=True)

    print("\nDetected Features Details:")
    print(f"  Number of features: {len(detected_gdf)}")
    print(f"  Coordinate Reference System: {detected_gdf.crs}")
    print(f"  Geometry types: {detected_gdf.geom_type.unique()}")
    print(f"  Bounding box: {detected_gdf.total_bounds}")

    # Read ground truth shapefile
    if gt_shapefile_path and os.path.exists(gt_shapefile_path):
        gt_shapefile = gpd.read_file(gt_shapefile_path)
        print("\nGround Truth Shapefile Details:")
        print(f"  Number of features: {len(gt_shapefile)}")
        print(f"  Coordinate Reference System: {gt_shapefile.crs}")
        print(f"  Geometry types: {gt_shapefile.geom_type.unique()}")
        print(f"  Bounding box: {gt_shapefile.total_bounds}")

        # Transform ground truth to match detected CRS
        if gt_shapefile.crs != detected_gdf.crs:
            print("Transforming ground truth CRS to match detected CRS")
            gt_shapefile = gt_shapefile.to_crs(detected_gdf.crs)
        else:
            print("CRS already match, no transformation needed")

        print("\nCRS after transformation:")
        print(f"Ground Truth CRS: {gt_shapefile.crs}")

        # Check bounding boxes
        detected_bbox = box(*detected_gdf.total_bounds)
        gt_bbox = box(*gt_shapefile.total_bounds)

        print(f"Detected bounding box: {detected_bbox.bounds}")
        print(f"Ground Truth bounding box: {gt_bbox.bounds}")

        if not detected_bbox.intersects(gt_bbox):
            print("WARNING: No overlap between detected features and ground truth")
        else:
            # Clip ground truth to the extent of the detected features
            gt_shapefile = gt_shapefile[gt_shapefile.intersects(detected_bbox)]
            print(f"Number of ground truth buildings in the tile: {len(gt_shapefile)}")

        # Plot ground truth buildings in blue
        gt_shapefile.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=0.5)
    else:
        print("Ground truth shapefile not found or not provided")

    # Plot detected buildings in red
    detected_gdf.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=0.5)

    # Create a custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=0.5, label='Detected'),
        Line2D([0], [0], color='blue', lw=0.5, label='Ground Truth')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.title(f"Detected vs Ground Truth - {os.path.basename(output_path)}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\nVisualization saved: {output_path}")
    print("--- End of Visualization Tile Information ---\n")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, 'config', 'config_raster.yaml')
    
    config = load_config(config_path)
    
    tif_folder = Path(config['data']['output']['out_tif_tiles_folder'])
    shp_folder = Path(config['data']['input']['shapefile_folder'])
    json_folder = Path(config['data']['output']['out_simp_folder'])
    vis_folder = Path(config['data']['output']['out_vis_folder'])
    
    vis_folder.mkdir(parents=True, exist_ok=True)
    
    tif_files = list(tif_folder.glob('*.tif'))
    
    for tif_file in tif_files:
        base_name = tif_file.stem
        shp_file = shp_folder / f"{base_name}.shp"
        json_file = json_folder / f"{base_name}.json"
        output_file = vis_folder / f"{base_name}_visualization.png"
        
        print(f"\nProcessing: {tif_file}")
        print(f"Shapefile: {shp_file}")
        print(f"JSON file: {json_file}")
        print(f"Output: {output_file}")
        
        if not os.path.exists(str(json_file)):
            print(f"Warning: JSON file not found for {tif_file}. Skipping this tile.")
            continue
        
        try:
            # Read the raster data
            with rasterio.open(str(tif_file)) as src:
                raster_data = src.read(1)
                transform = src.transform

            # Read the JSON file with detected features
            with open(str(json_file), 'r') as f:
                json_data = json.load(f)
                detected_features = json_data.get('features', [])
            
            if not detected_features:
                print(f"Warning: No features found in JSON file for {tif_file}. Skipping this tile.")
                continue

            # Visualize the tile
            visualize_tile(raster_data, transform, detected_features, str(shp_file), str(output_file), config, str(tif_file))
            print(f"Visualization saved: {output_file}")
            
            # Clear matplotlib figure to free up memory
            plt.close('all')
            
        except Exception as e:
            print(f"Error processing {tif_file}: {str(e)}")
            print("Detailed error information:")
            traceback.print_exc()
    
    print(f"\nVisualizations saved in {vis_folder}")

if __name__ == "__main__":
    main()