import os
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import box
import numpy as np
import sys
import argparse


##FUNket ikke, prøver å lage ny med feilmelding i v0###

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import get_project_paths, create_directory
from utils.config_loader import get_config

def get_tile_bounds(tif_path, tile_size_pixels):
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        transform = src.transform
        crs = src.crs
        
        width = src.width
        height = src.height
        
        tiles = []
        
        for row in range(0, height, tile_size_pixels):
            for col in range(0, width, tile_size_pixels):
                window = Window(col, row, 
                                min(tile_size_pixels, width - col),
                                min(tile_size_pixels, height - row))
                window_transform = rasterio.windows.transform(window, transform)
                window_bounds = rasterio.windows.bounds(window, transform)
                
                tiles.append({
                    'row': row // tile_size_pixels,
                    'col': col // tile_size_pixels,
                    'window': window,
                    'transform': window_transform,
                    'bounds': window_bounds
                })
        
        return tiles, crs

def create_tif_tile(src, window, transform, output_path):
    """Create a single TIF tile"""
    data = src.read(window=window)
    
    profile = src.profile.copy()
    profile.update({
        'height': window.height,
        'width': window.width,
        'transform': transform,
        'driver': 'GTiff',
        'compress': 'lzw'
    })
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data)
    
    print(f"Created tile: {output_path}")
    print(f"Dimensions: {window.height} x {window.width}")
    print(f"Transform: {transform}")

def create_shapefile_tile(buildings_gdf, bounds, output_path):
    """Create a single shapefile tile"""
    tile_polygon = box(*bounds)
    tile_buildings = buildings_gdf.clip(tile_polygon)
    
    if not tile_buildings.empty:
        tile_buildings.to_file(output_path)
        return len(tile_buildings)
    return 0

def create_matching_tiles(tif_paths, shapefile_path, output_dir, tif_tiles_dir, image_tiles_dir, shp_tiles_dir, tile_size_pixels=1000):
    try:
        print("Starting tile creation process...")
        
        # Create output directories
        create_directory(tif_tiles_dir)
        create_directory(image_tiles_dir)
        create_directory(shp_tiles_dir)
        print(f"TIF tiles will be saved in: {tif_tiles_dir}")
        print(f"Image tiles will be saved in: {image_tiles_dir}")
        print(f"Shapefiles will be saved in: {shp_tiles_dir}")
        
        # Read the shapefile
        print("Reading shapefile...")
        buildings_gdf = gpd.read_file(shapefile_path)
        
        for tif_info in tif_paths:
            tif_name = tif_info['name']
            tif_path = tif_info['path']
            print(f"\nProcessing {tif_name}...")
        
            # Get tile bounds from TIF
            print("Calculating tile bounds...")
            tiles, tif_crs = get_tile_bounds(tif_path, tile_size_pixels)
            
            # Ensure the shapefile is in the same CRS as the TIF
            if buildings_gdf.crs != tif_crs:
                print(f"Converting shapefile from {buildings_gdf.crs} to {tif_crs}")
                buildings_gdf = buildings_gdf.to_crs(tif_crs)
            
            # Process each tile
            print(f"Creating {len(tiles)} tiles for {tif_name}...")
            
            with rasterio.open(tif_path) as src:
                for tile in tiles:
                    tile_id = f"{tile['row']}_{tile['col']}"
                    
                    if "image" in tif_name.lower():
                        # Create image tile
                        image_tile_path = os.path.join(image_tiles_dir, f"tile_{tile_id}.tif")
                        create_tif_tile(src, tile['window'], tile['transform'], image_tile_path)
                        print(f"Created image tile {tile_id}")
                    else:
                        # Create TIF tile
                        tif_tile_path = os.path.join(tif_tiles_dir, f"tile_{tile_id}.tif")
                        create_tif_tile(src, tile['window'], tile['transform'], tif_tile_path)
                        print(f"Created TIF tile {tile_id}")
                    
                    # Create shapefile tile (for all TIF files)
                    shp_tile_path = os.path.join(shp_tiles_dir, f"tile_{tile_id}.shp")
                    num_buildings = create_shapefile_tile(buildings_gdf, tile['bounds'], shp_tile_path)
                    print(f"Created shapefile tile {tile_id} with {num_buildings} buildings")
        
        print("\nTile creation completed successfully!")
        print("\nSummary of created files:")
        print(f"TIF tiles created: {len([f for f in os.listdir(tif_tiles_dir) if f.endswith('.tif')])}")
        print(f"Image tiles created: {len([f for f in os.listdir(image_tiles_dir) if f.endswith('.tif')])}")
        print(f"Shapefiles created: {len([f for f in os.listdir(shp_tiles_dir) if f.endswith('.shp')])}")
        return True
        
    except Exception as e:
        print(f"Error creating tiles: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Create tiles from geospatial data')
    parser.add_argument('--tile-size', type=int, default=750, help='Tile size in pixels')
    args = parser.parse_args()

    config = get_config()
    paths = get_project_paths()
    
    # Use the tile size from command-line argument
    tile_size = args.tile_size
    
    # Input paths
    tif_paths = paths['tif_paths']
    shapefile_path = paths['shapefile_path']
    output_dir = paths['tiles_dir']
    tif_tiles_dir = paths['tif_tiles_dir']
    image_tiles_dir = paths['image_tiles_dir']
    shp_tiles_dir = paths['shp_tiles_dir']
    
    # Parameters
    
    
    # Create the tiles
    success = create_matching_tiles(
        tif_paths=tif_paths,
        shapefile_path=shapefile_path,
        output_dir=output_dir,
        tif_tiles_dir=tif_tiles_dir,
        image_tiles_dir=image_tiles_dir,
        shp_tiles_dir=shp_tiles_dir,
        tile_size_pixels=tile_size
    )
    
    if success:
        print(f"\nAll files have been created in {output_dir}")
        print(f"- TIF tiles: {tif_tiles_dir}")
        print(f"- Image tiles: {image_tiles_dir}")
        print(f"- Shapefiles: {shp_tiles_dir}")
    else:
        print("Failed to create tiles")

if __name__ == "__main__":
    main()