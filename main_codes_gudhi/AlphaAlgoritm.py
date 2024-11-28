import numpy as np
import alphashape
import rasterio
from shapely.geometry import MultiPolygon, Polygon
from utils.mdl_io import load_raster

def alpha_shape_detection(raster_path, alpha=1.0):
    # Load raster data
    image, transform = load_raster(raster_path)

    # Extract non-zero data points and convert to geographic coordinates
    mask = image != 0
    coords = np.column_stack(np.where(mask))
    points = np.array([(transform * (x, y))[0:2] for y, x in coords])

    # Apply alpha shape for building outlines
    alpha_shape = alphashape.alphashape(points, alpha)

    # Convert to list of Polygon objects
    if isinstance(alpha_shape, MultiPolygon):
        buildings = list(alpha_shape.geoms)
    else:
        buildings = [alpha_shape]

    return buildings

if __name__ == "__main__":
    test = "/Users/Rebekka/GiHub/PHshapeModified/output/tiles/tif_tiles/tile_test_1.tif"
    detected_buildings = alpha_shape_detection(test)
    print(f"Number of buildings detected: {len(detected_buildings)}")