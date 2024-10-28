import os
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from utils.mdl_geo import poly2Geojson
from utils.mdl_io import save_json

def simplify_polygon(polygon, tolerance):
    return polygon.simplify(tolerance)

def main_simp_ol(building_outlines, out_folder, bld_list, bfr_tole=0.5, bfr_otdiff=0.0, simp_method="haus",
                 savename_bfr="", is_unrefresh_save=False, is_save_fig=False, is_Debug=False):
    
    if not isinstance(building_outlines, list):
        building_outlines = [building_outlines]
    
    simplified_outlines = []
    for i, outline in enumerate(building_outlines):
        simplified_outline = simplify_polygon(outline, bfr_tole)
        simplified_outlines.append(simplified_outline)
    
    # Save all simplified outlines in a single GeoJSON file
    if bld_list and isinstance(bld_list[0], str):
        savename = os.path.join(out_folder, f"{bld_list[0]}.json")
    else:
        savename = os.path.join(out_folder, "simplified_outlines.json")
    
    # Create a MultiPolygon from the simplified outlines
    multi_polygon = MultiPolygon(simplified_outlines)
    
    all_outlines_json = poly2Geojson(multi_polygon, round_precision=6)
    
    # Wrap the GeoJSON in a FeatureCollection
    feature_collection = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": all_outlines_json,
            "properties": {}
        }]
    }
    
    save_json(feature_collection, savename)
    
    print(f"Saved {len(simplified_outlines)} simplified outlines to: {savename}")
    
    return simplified_outlines