import os
import json
import geopandas as gpd
from shapely.geometry import shape, Polygon
from shapely.validation import make_valid
import yaml

def validate_geometry(geom):
    if isinstance(geom, (int, float)):
        print(f"Error: Unexpected numeric value: {geom}")
        return None
    if isinstance(geom, str):
        try:
            geom_dict = json.loads(geom)
            geom = shape(geom_dict)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON string: {geom[:50]}...")
            return None
    if not isinstance(geom, Polygon):
        print(f"Error: Geometry is not a Polygon: {type(geom)}")
        return None
    if not geom.is_valid:
        return make_valid(geom)
    return geom

def calculate_iou(polygon1, polygon2):
    try:
        polygon1 = validate_geometry(polygon1)
        polygon2 = validate_geometry(polygon2)
        if polygon1 is None or polygon2 is None:
            return 0
        intersection = polygon1.intersection(polygon2).area
        union = polygon1.union(polygon2).area
        return intersection / union if union > 0 else 0
    except Exception as e:
        print(f"Error calculating IoU: {e}")
        print(f"Polygon1 type: {type(polygon1)}, Polygon2 type: {type(polygon2)}")
        return 0

def match_outlines(gt_outlines, detected_outlines, iou_threshold=0.5):
    matches = []
    unmatched_gt = list(range(len(gt_outlines)))
    unmatched_detected = list(range(len(detected_outlines)))

    for i, gt in enumerate(gt_outlines):
        best_match = None
        best_iou = 0

        for j, detected in enumerate(detected_outlines):
            iou = calculate_iou(gt, detected)
            if iou > best_iou and iou >= iou_threshold:
                best_match = j
                best_iou = iou

        if best_match is not None:
            matches.append((i, best_match, best_iou))
            unmatched_gt.remove(i)
            unmatched_detected.remove(best_match)

    return matches, unmatched_gt, unmatched_detected

def process_tile(shp_path, json_path, output_path):
    print(f"Processing tile: {os.path.basename(shp_path)}")
    
    # Read ground truth outlines
    gt_gdf = gpd.read_file(shp_path)
    gt_outlines = [geom for geom in gt_gdf.geometry]

    # Read detected outlines
    try:
        with open(json_path, 'r') as f:
            detected_data = json.load(f)
        
        if isinstance(detected_data, list):
            print("JSON structure: List of features")
            print(f"Number of features: {len(detected_data)}")
            if detected_data:
                print(f"Keys in first item: {list(detected_data[0].keys())}")
                if 'geometry' in detected_data[0]:
                    print(f"Geometry type: {detected_data[0]['geometry'].get('type', 'Not specified')}")
                else:
                    print("Warning: 'geometry' key not found in the first item")
            detected_outlines = []
            for feature in detected_data:
                if 'geometry' in feature:
                    try:
                        detected_outlines.append(shape(feature['geometry']))
                    except Exception as e:
                        print(f"Error processing geometry: {e}")
                        print(f"Problematic geometry: {json.dumps(feature['geometry'], indent=2)}")
                else:
                    print(f"Warning: 'geometry' key not found in feature: {json.dumps(feature, indent=2)}")
        elif isinstance(detected_data, dict):
            print(f"JSON structure: {list(detected_data.keys())}")
            if 'features' in detected_data:
                detected_outlines = [shape(feature['geometry']) for feature in detected_data['features']]
            elif 'type' in detected_data and detected_data['type'] == 'FeatureCollection':
                detected_outlines = [shape(feature['geometry']) for feature in detected_data['features']]
            else:
                print(f"Error: Unexpected JSON structure in {json_path}")
                return
        else:
            print(f"Error: Unexpected JSON type in {json_path}: {type(detected_data)}")
            return

    except Exception as e:
        print(f"Error reading JSON file {json_path}: {e}")
        return

    print(f"Ground truth outlines: {len(gt_outlines)}")
    print(f"Detected outlines: {len(detected_outlines)}")

    # Match outlines
    matches, unmatched_gt, unmatched_detected = match_outlines(gt_outlines, detected_outlines)

    print(f"Matches: {len(matches)}")
    print(f"Unmatched ground truth: {len(unmatched_gt)}")
    print(f"Unmatched detected: {len(unmatched_detected)}")

    # Prepare output data
    output_data = {
        "matches": [
            {
                "gt_index": gt_idx,
                "detected_index": det_idx,
                "iou": iou
            } for gt_idx, det_idx, iou in matches
        ],
        "unmatched_gt": unmatched_gt,
        "unmatched_detected": unmatched_detected
    }

    # Save output
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

def main(config_path):
    # Load configuration
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Set up paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shp_folder = os.path.join(base_dir, config['data']['input'].get('shapefile_folder', 'input/shapefiles'))
    json_folder = os.path.join(base_dir, config['data']['output'].get('out_simp_folder', 'output/simplified'))
    output_folder = os.path.join(base_dir, config['data']['output'].get('out_match_folder', 'output/matched_outlines'))

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    print(f"Shapefile folder: {shp_folder}")
    print(f"JSON folder: {json_folder}")
    print(f"Output folder: {output_folder}")

    for shp_file in os.listdir(shp_folder):
        if shp_file.endswith('.shp'):
            base_name = os.path.splitext(shp_file)[0]
            shp_path = os.path.join(shp_folder, shp_file)
            json_path = os.path.join(json_folder, f"{base_name}.json")
            output_path = os.path.join(output_folder, f"{base_name}_matched.json")

            if os.path.exists(json_path):
                try:
                    process_tile(shp_path, json_path, output_path)
                    print(f"Processed {base_name}")
                except Exception as e:
                    print(f"Error processing {base_name}: {e}")
            else:
                print(f"Warning: JSON file not found for {base_name}")

    print("All tiles processed.")

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config_raster.yaml")
    main(config_path)