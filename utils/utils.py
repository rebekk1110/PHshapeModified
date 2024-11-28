import os
import sys
import rasterio
from rasterio.plot import show
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box, Polygon, MultiPolygon
import numpy as np
from pyproj import CRS, Transformer
from shapely.validation import make_valid
from pathlib import Path
import json

#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .config_loader import get_config

def get_tif_bounds_and_info(tif_path):
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        original_crs = src.crs
        
        transformer = Transformer.from_crs(
            original_crs,
            CRS.from_epsg(4326),
            always_xy=True
        )
        
        minx, miny = transformer.transform(bounds.left, bounds.bottom)
        maxx, maxy = transformer.transform(bounds.right, bounds.top)
        
        bounds_wgs84 = Polygon([
            (minx, miny),
            (maxx, miny),
            (maxx, maxy),
            (minx, maxy),
            (minx, miny)
        ])
        
        return bounds_wgs84, original_crs, src.transform, src.read(1)

def is_valid_polygon(geom):
    if geom is None:
        return False
    if not isinstance(geom, (Polygon, MultiPolygon)):
        return False
    try:
        valid_geom = make_valid(geom)
        return valid_geom.is_valid and not valid_geom.is_empty
    except Exception:
        return False

def create_visualization(tif_data, transform, buildings_gdf, output_path):
    fig, ax = plt.subplots(figsize=(20, 20))
    
    tif_normalized = normalize_raster(tif_data)
    show(tif_normalized, transform=transform, ax=ax, cmap='gray')
    
    if buildings_gdf is not None and not buildings_gdf.empty:
        buildings_gdf.boundary.plot(ax=ax, color='red', linewidth=0.5, alpha=0.7)
    
    plt.title('Building Outlines overlaid on TIF image')
    ax.set_axis_off()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def get_project_paths():
    base_dir = Path(__file__).resolve().parent.parent

    config = get_config()
    print('got config')
    raster_folder = config['data']['input']['big_raster_folder']
    raster_files = config['data']['input']['raster_files']
    
    # Find the raster file to use for bounds
    bounds_raster = next((raster for raster in raster_files if raster.get('use_for_bounds')), None)
    if not bounds_raster:
        raise ValueError("No raster file specified for bounds in configuration")
    
    bounds_tif_path = os.path.join(raster_folder, bounds_raster['file'])
    
    # Create a list of all TIF paths
    tif_paths = [{'name': raster['name'], 'path': os.path.join(raster_folder, raster['file'])} for raster in raster_files]
    
    output_dir = config['data']['output']['out_root_folder']
    tiles_dir = config['data']['output']['out_tiles_folder']
    tif_tiles_dir = config['data']['output']['out_tif_tiles_folder']
    image_tiles_dir = config['data']['output']['out_image_tiles_folder']
    shp_tiles_dir = config['data']['output']['out_shp_tiles_folder']
    shapefile_path = os.path.join(output_dir, "buildings.shp")
    visualization_path = os.path.join(output_dir, "buildings_overlay.png")
    

    return {
        'bounds_tif_path': bounds_tif_path,
        'tif_paths': tif_paths,
        'output_dir': output_dir,
        'tiles_dir': tiles_dir,
        'tif_tiles_dir': tif_tiles_dir,
        'image_tiles_dir': image_tiles_dir,
        'shp_tiles_dir': shp_tiles_dir,
        'shapefile_path': shapefile_path,
        'visualization_path': visualization_path,
        'tif_tiles_dir': os.path.join(base_dir, config['data']['output']['out_tif_tiles_folder']),
        'out_simp_folder': os.path.join(base_dir, config['data']['output']['out_simp_folder']),
        'out_vis_folder': os.path.join(base_dir, config['data']['output']['out_vis_folder']),
        'out_shp_tiles_folder': os.path.join(base_dir, config['data']['output']['out_shp_tiles_folder']),
        
    }

def normalize_raster(raster_data):
    data = raster_data.astype(np.float32)
    data[~np.isfinite(data)] = np.nan
    raster_min, raster_max = np.nanmin(data), np.nanmax(data)
    
    if raster_min == raster_max:
        return np.zeros_like(data, dtype=np.float32)
    
    normalized = np.zeros_like(data, dtype=np.float32)
    mask = ~np.isnan(data)
    normalized[mask] = (data[mask] - raster_min) / (raster_max - raster_min)
    normalized[~np.isfinite(normalized)] = 0
    
    return normalized

def create_directory(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)

def load_tile_info(tiles_dir):
    info_path = os.path.join(tiles_dir, 'tile_info.json')
    try:
        with open(info_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise Exception(f"Error loading tile information: {str(e)}")

def save_tile_info(tiles_dir, tile_info):
    info_path = os.path.join(tiles_dir, 'tile_info.json')
    try:
        with open(info_path, 'w') as f:
            json.dump(tile_info, f, indent=2)
    except Exception as e:
        raise Exception(f"Error saving tile information: {str(e)}")

def validate_paths(paths_dict):
    for name, path in paths_dict.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required path not found - {name}: {path}")
    return True

class TileInfo:
    def __init__(self, tile_id, tif_path, shp_path, bounds, num_buildings=0):
        self.tile_id = tile_id
        self.tif_path = tif_path
        self.shp_path = shp_path
        self.bounds = bounds
        self.num_buildings = num_buildings
    
    def to_dict(self):
        return {
            'tile_id': self.tile_id,
            'tif_path': self.tif_path,
            'shp_path': self.shp_path,
            'bounds': list(self.bounds),
            'num_buildings': self.num_buildings
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            tile_id=data['tile_id'],
            tif_path=data['tif_path'],
            shp_path=data['shp_path'],
            bounds=data['bounds'],
            num_buildings=data['num_buildings']
        )
