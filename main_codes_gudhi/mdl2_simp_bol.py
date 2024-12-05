import os
import json
import numpy as np
import logging
from tqdm import tqdm
import rasterio
import rasterio
from shapely.geometry import Polygon, MultiPolygon
import geopandas as gpd

from modules.simp_basic_ol import simp_poly_Fd, simp_poly_Extmtd
from utils.mdl_io import load_raster, save_json, get_raster_path, get_output_folder, get_gt_shp_path, get_specific_output_folder
from utils.mdl_geo import poly2Geojson
from utils.mdl_visual import  drawmultipolygon
from mdl1_bolPH_gu import main_basicOL
from mdl_eval import main_eval


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main_simp_ol(buildings, out_folder, tile_name, raster_path, bfr_tole=0.5, bfr_otdiff=0.0, simp_method="haus", is_save_fig=False, is_Debug=False, is_random=False):
    logging.info("MDL2: Starting building simplification process")
    if not buildings:
        logging.warning("MDL2: No buildings to simplify.")
        return []

    # Update the output folder based on whether it's a random tile or not
    out_folder = get_specific_output_folder("mdl2", is_random)
    os.makedirs(out_folder, exist_ok=True)

    simplified_buildings = []

    for i, building in enumerate(tqdm(buildings, desc="MDL2: Simplifying buildings")):
        try:
            if simp_method == "haus":
                thres_haus = (1 - np.cos(30 / 180 * np.pi)) * (bfr_tole - bfr_otdiff)
                b_oli_simp_ext, b_oli_simp_ints, b_oli_simp = simp_poly_Fd(building, thres_mode="haus", thres_haus=thres_haus, isDebug=is_Debug)
            elif simp_method == "iou":
                thres_simparea = 0.95  # You may want to make this configurable
                b_oli_simp_ext, b_oli_simp_ints, b_oli_simp = simp_poly_Fd(building, thres_mode="iou", thres_simparea=thres_simparea, isDebug=is_Debug)
            else:
                b_oli_simp_ext, b_oli_simp_ints, b_oli_simp = simp_poly_Extmtd(building, bfr_otdiff=bfr_otdiff, bfr_tole=bfr_tole)

            simplified_buildings.append(b_oli_simp)

            # Check for interior holes
            if isinstance(b_oli_simp, Polygon):
                if len(b_oli_simp.interiors) > 0:
                    logging.info(f"MDL2: Simplified building {i+1} has {len(b_oli_simp.interiors)} interior hole(s)")
            elif isinstance(b_oli_simp, MultiPolygon):
                total_interiors = sum(len(poly.interiors) for poly in b_oli_simp.geoms)
                if total_interiors > 0:
                    logging.info(f"MDL2: Simplified building {i+1} (MultiPolygon) has {total_interiors} total interior hole(s)")

            # Save simplified building as GeoJSON
            b_oli_simp_json = poly2Geojson(b_oli_simp, round_precision=6)
            savename = os.path.join(out_folder, f"{tile_name}_simplified_building_{i+1}.json")
            save_json(b_oli_simp_json, savename)

            if is_save_fig:
                out_path_fig = os.path.join(out_folder, 'figures', f"{tile_name}_simplified_building_{i+1}")
                os.makedirs(out_path_fig, exist_ok=True)
                drawmultipolygon(b_oli_simp, title=f"{tile_name} Simplified Building {i+1}",
                                 savepath=os.path.join(out_path_fig, f"{tile_name}_simplified_building_{i+1}"))
        except Exception as e:
            logging.error(f"MDL2: Error processing building {i+1}: {str(e)}")

    # Save all simplified buildings as a single GeoJSON file
    gdf = gpd.GeoDataFrame(geometry=simplified_buildings)
    gdf.to_file(os.path.join(out_folder, f"{tile_name}_simplified_buildings.geojson"), driver="GeoJSON")

    logging.info(f"MDL2: Total simplified buildings: {len(simplified_buildings)}")
    logging.info("MDL2: Completed building simplification process")
    return simplified_buildings


if __name__ == "__main__":
    # Run mdl1 to get basic outlines
    tile_name = "tile_26_9"
    raster_path = get_raster_path(tile_name)
    mdl1_out_folder = get_output_folder("mdl1")
    mdl2_out_folder = get_output_folder("mdl2")
    gt_shp_path = get_gt_shp_path(tile_name)
    
    # Run MDL1 if necessary
    if not os.path.exists(mdl1_out_folder) or not any(f.startswith(tile_name) for f in os.listdir(mdl1_out_folder)):
        buildings = main_basicOL(raster_path, mdl1_out_folder, tile_name, is_Debug=True)
    else:
        # Load existing buildings from MDL1 output
        buildings = []
        for file in os.listdir(mdl1_out_folder):
            if file.startswith(tile_name) and file.endswith('.json'):
                with open(os.path.join(mdl1_out_folder, file), 'r') as f:
                    building_data = json.load(f)
                    buildings.append(Polygon(building_data['coordinates'][0]))
    
    #logging.info(f"Number of buildings detected: {len(buildings)}")

    if not buildings:
        logging.warning("No buildings detected. Exiting.")
        sys.exit(0)

    # Run mdl2 to simplify the outlines
    simplified_buildings = main_simp_ol(buildings, mdl2_out_folder, tile_name, raster_path, is_Debug=True)

 
    # Run evaluation
    logging.info("Running evaluation...")
    eval_results = main_eval(res_folder=mdl2_out_folder,
                             res_type=".json",
                             gt_shp_path=gt_shp_path,
                             out_folder=os.path.join(mdl2_out_folder, "eval_results"),
                             tile_name=tile_name,
                             use_v2_hausdorff=True)

    logging.info("Evaluation complete. Results saved in the output folder.")
    logging.info(f"Mean IoU: {eval_results['IOU'].mean():.4f}")
    logging.info(f"Mean Hausdorff Distance: {eval_results['HD'].mean():.4f}")