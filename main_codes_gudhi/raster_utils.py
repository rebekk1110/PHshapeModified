import numpy as np

def preprocess_raster(raster_image):
    # Replace NaN values with 0
    raster_image = np.nan_to_num(raster_image, nan=0.0)
    
    # Normalize the image
    if raster_image.max() != raster_image.min():
        raster_image = (raster_image - raster_image.min()) / (raster_image.max() - raster_image.min())
    
    return raster_image