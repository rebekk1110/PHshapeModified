import numpy as np
from shapely.geometry import Polygon
from skimage import measure
import rasterio


def get_building_outlines_from_raster(raster_image, transform):
    # Ensure the image is binary
    threshold = raster_image.mean()
    binary_image = raster_image > threshold

    # Find contours
    contours = measure.find_contours(binary_image, 0.5)
    
    print(f"Number of contours detected: {len(contours)}")
    
    building_outlines = []
    for contour in contours:
        # Convert pixel coordinates to geospatial coordinates
        coords = [rasterio.transform.xy(transform, y, x) for y, x in contour]
        # Close the polygon
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        # Only add polygons with a minimum area (to filter out noise)
        poly = Polygon(coords)
        if poly.area > 10:  # Adjust this threshold as needed
            building_outlines.append(poly)
    
    print(f"Number of building outlines after filtering: {len(building_outlines)}")
    return building_outlines