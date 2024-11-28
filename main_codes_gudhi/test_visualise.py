import os
import random
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import numpy as np

def visualize_tiles(tile_directory, num_tiles=60):
    # Get all TIF files in the directory
    tif_files = [f for f in os.listdir(tile_directory) if f.endswith('.tif')]
    
    # Randomly select 30 tiles (or less if there are fewer tiles)
    selected_tiles = random.sample(tif_files, min(num_tiles, len(tif_files)))
    
    # Calculate the grid size for the subplot
    grid_size = int(np.ceil(np.sqrt(len(selected_tiles))))
    
    # Create a figure with subplots
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(30, 20))
    fig.suptitle("Visualization of 30 Random Tiles", fontsize=16)
    
    # Flatten the axes array for easier indexing
    axes = axes.flatten()
    
    for idx, tile_file in enumerate(selected_tiles):
        tile_path = os.path.join(tile_directory, tile_file)
        
        with rasterio.open(tile_path) as src:
            # Read the first band of the raster
            tile_data = src.read(1)
            
            # Plot the tile
            show(tile_data, ax=axes[idx], cmap='gray', title=f"Tile: {tile_file}")
            axes[idx].axis('off')
            axes[idx].set_title(f"Tile: {tile_file}", fontsize=8)
        
        print(f"Visualized tile: {tile_file}")
    
    # Remove any unused subplots
    for idx in range(len(selected_tiles), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Replace this with the actual path to your tile directory
    tile_directory = "/Users/Rebekka/GiHub/PHshapeModified/output/tiles/tif_tiles"
    
    visualize_tiles(tile_directory)