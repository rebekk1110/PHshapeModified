import os
import yaml
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def normalize_raster(raster_data):
    data = raster_data.astype(np.float32)
    data[~np.isfinite(data)] = np.nan
    raster_min, raster_max = np.nanmin(data), np.nanmax(data)
    
    if raster_min == raster_max:
        return np.zeros_like(data, dtype=np.float32)
    
    normalized = (data - raster_min) / (raster_max - raster_min)
    normalized[~np.isfinite(normalized)] = 0
    
    return normalized

def visualize_tile(tif_path, shp_path, output_path):
    try:
        with rasterio.open(tif_path) as src:
            raster_data = src.read(1)
            print(f"Raster shape: {raster_data.shape}")
            print(f"Raster bounds: {src.bounds}")
            print(f"Raster CRS: {src.crs}")
            print(f"Raster min: {np.min(raster_data)}, max: {np.max(raster_data)}")
            
          #  raster_data = normalize_raster(raster_data)
            print(f"Normalized raster min: {np.min(raster_data)}, max: {np.max(raster_data)}")
            
            fig, ax = plt.subplots(figsize=(10, 10))
            show(raster_data, ax=ax, cmap='viridis')  # Changed cmap to 'viridis' for better visibility
            plt.colorbar(ax.images[0], ax=ax, label='Normalized pixel value')
            
            if os.path.exists(shp_path):
                gdf = gpd.read_file(shp_path)
                print(f"Shapefile CRS: {gdf.crs}")
                print(f"Number of features in shapefile: {len(gdf)}")
                
                if not gdf.empty:
                    # Ensure the GeoDataFrame is in the same CRS as the raster
                    if gdf.crs != src.crs:
                        gdf = gdf.to_crs(src.crs)
                    
                    gdf.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=0.5)
                    print(f"Plotted {len(gdf)} features from shapefile")
                else:
                    print(f"Warning: Shapefile {shp_path} is empty")
            else:
                print(f"Warning: Shapefile {shp_path} does not exist")
            
            plt.title(f"Tile: {os.path.basename(tif_path)}")
            plt.axis('on')  # Turn on axis to see the extent
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Verify that the output file was created and has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"Visualization created for {os.path.basename(tif_path)}")
                print(f"Output file size: {os.path.getsize(output_path)} bytes")
            else:
                print(f"Error: Failed to create visualization for {os.path.basename(tif_path)}")
    except Exception as e:
        print(f"Error visualizing {os.path.basename(tif_path)}: {str(e)}")

def main():
    # Load configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, 'config', 'config_raster.yaml')
    
    try:
        config = load_config(config_path)
        print(f"Configuration loaded from: {config_path}")
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        return

    # Set up paths
    tif_folder = Path(config['data']['output']['out_tif_tiles_folder'])
    shp_folder = Path(config['data']['output']['out_shp_tiles_folder'])
    vis_folder = Path(config['data']['output']['out_vis_folder'])
    
    print(f"TIF folder: {tif_folder}")
    print(f"SHP folder: {shp_folder}")
    print(f"Visualization folder: {vis_folder}")
    
    # Create visualization folder if it doesn't exist
    vis_folder.mkdir(parents=True, exist_ok=True)
    
    # Get list of TIF files
    tif_files = list(tif_folder.glob('*.tif'))
    print(f"Found {len(tif_files)} TIF files")
    
    # Select 5 random tiles
    num_tiles = min(5, len(tif_files))
    selected_tiles = random.sample(tif_files, num_tiles)
    
    # Visualize selected tiles
    for tif_file in selected_tiles:
        base_name = tif_file.stem
        shp_file = shp_folder / f"{base_name}.shp"
        output_file = vis_folder / f"{base_name}_visualization_shp_tiles.png"
        
        print(f"\nProcessing: {tif_file}")
        print(f"Shapefile: {shp_file}")
        print(f"Output: {output_file}")
        
        visualize_tile(str(tif_file), str(shp_file), str(output_file))
    
    print(f"\nVisualizations saved in {vis_folder}")

if __name__ == "__main__":
    main()