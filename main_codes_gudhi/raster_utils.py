import numpy as np
from skimage import filters

def preprocess_raster(raster_image):
    # Normalize the image
    raster_image = (raster_image - np.min(raster_image)) / (np.max(raster_image) - np.min(raster_image))
    
    # Apply Gaussian smoothing
    smoothed = filters.gaussian(raster_image, sigma=1)
    
    # Apply Otsu's thresholding
    binary = smoothed > filters.threshold_otsu(smoothed)
    
    return binary

def raster_to_vector(binary_image, transform):
    from rasterio import features
    shapes = features.shapes(binary_image.astype('uint8'), transform=transform)
    return list(shapes)