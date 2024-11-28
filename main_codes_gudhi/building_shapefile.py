import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
from rasterio.plot import show
from shapely.geometry import box, Polygon, MultiPolygon
import numpy as np
from pyproj import CRS, Transformer
import fiona
from shapely.validation import make_valid
from utils.utils import get_config, get_project_paths, get_tif_bounds_and_info, is_valid_polygon, create_visualization

def get_buildings_for_area(bounds_wgs84):
    """Fetch buildings from OSM using a valid WGS84 polygon"""
    try:
        area_poly = box(*bounds_wgs84.bounds)
        
        print("Fetching buildings from OSM...")
        buildings = ox.features.features_from_polygon(
            area_poly,
            tags={'building': True}
        )
        
        if buildings is None or buildings.empty:
            print("No buildings found in the specified area")
            return None
        
        # Keep only essential columns
        essential_columns = ['geometry', 'building', 'name', 'height', 'levels']
        existing_columns = [col for col in essential_columns if col in buildings.columns]
        buildings = buildings[existing_columns]
        
        # Filter for valid polygons only
        valid_mask = buildings.geometry.apply(is_valid_polygon)
        buildings = buildings[valid_mask].copy()
        
        if buildings.empty:
            print("No valid building polygons found")
            return None
            
        print(f"Found {len(buildings)} valid building polygons")
        return buildings
        
    except Exception as e:
        print(f"Error fetching buildings: {str(e)}")
        return None

def save_buildings_shapefile(buildings_gdf, shapefile_path):
    """Save buildings to shapefile with proper geometry handling"""
    try:
        # Ensure all geometries are valid
        buildings_gdf['geometry'] = buildings_gdf.geometry.apply(make_valid)
        
        # Convert MultiPolygons to individual Polygons
        exploded = buildings_gdf.explode(index_parts=True)
        
        # Create schema for shapefile
        schema = {
            'geometry': 'Polygon',
            'properties': {
                'building': 'str',
                'name': 'str',
                'height': 'str',
                'levels': 'str'
            }
        }
        
        # Write to shapefile
        with fiona.open(
            shapefile_path,
            'w',
            driver='ESRI Shapefile',
            schema=schema,
            crs=buildings_gdf.crs
        ) as output:
            for idx, row in exploded.iterrows():
                if isinstance(row.geometry, (Polygon, MultiPolygon)):
                    feature = {
                        'geometry': row.geometry.__geo_interface__,
                        'properties': {
                            'building': str(row.get('building', '')),
                            'name': str(row.get('name', '')),
                            'height': str(row.get('height', '')),
                            'levels': str(row.get('levels', ''))
                        }
                    }
                    output.write(feature)
        
        print(f"Successfully saved buildings to {shapefile_path}")
        return True
    except Exception as e:
        print(f"Error saving shapefile: {str(e)}")
        return False


def main():
    
    config = get_config()
    paths = get_project_paths()
    
    bounds_tif_path = paths['bounds_tif_path']
    output_path = paths['visualization_path']
    shapefile_path = paths['shapefile_path']
    
    print(f"Using TIF file for bounds: {bounds_tif_path}")
    
    print("Reading TIF file and computing bounds...")
    bounds_wgs84, original_crs, transform, tif_data = get_tif_bounds_and_info(bounds_tif_path)
    
    buildings_gdf = get_buildings_for_area(bounds_wgs84)
    
    if buildings_gdf is not None and not buildings_gdf.empty:
        # Convert buildings to the same CRS as the TIF
        buildings_gdf = buildings_gdf.to_crs(original_crs)
        
        # Save buildings as shapefile
        if save_buildings_shapefile(buildings_gdf, shapefile_path):
            print("Creating visualization...")
            create_visualization(tif_data, transform, buildings_gdf, output_path)
            print(f"Visualization saved to {output_path}")
    else:
        print("No buildings were found or an error occurred")

if __name__ == "__main__":
    main()