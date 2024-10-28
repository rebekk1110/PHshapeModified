import json
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
from shapely.geometry import shape, mapping

def count_buildings(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    if 'type' in data and data['type'] == 'FeatureCollection':
        return len(data['features'])
    else:
        # If it's a single geometry, return 1
        return 1

def visualize_results(tif_file, ph_shape_file, output_file):
    # Open the TIF file
    with rasterio.open(tif_file) as src:
        # Read the data
        data = src.read(1)

    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Show the TIF data
    show(data, ax=ax, cmap='gray')

    # Load the PH.Shape results
    with open(ph_shape_file, 'r') as f:
        ph_shape_data = json.load(f)

    # Count the number of buildings
    num_buildings = count_buildings(ph_shape_file)

    # Function to plot a single geometry
    def plot_geometry(geom, label):
        if geom['type'] == 'MultiPolygon':
            multi_polygon = shape(geom)
            for idx, polygon in enumerate(multi_polygon.geoms):
                x, y = polygon.exterior.xy
                ax.plot(x, y, color='red', linewidth=2)
                centroid = polygon.centroid
                ax.text(centroid.x, centroid.y, f"{label}-{idx+1}", ha='center', va='center', 
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        elif geom['type'] == 'Polygon':
            polygon = shape(geom)
            x, y = polygon.exterior.xy
            ax.plot(x, y, color='red', linewidth=2)
            centroid = polygon.centroid
            ax.text(centroid.x, centroid.y, label, ha='center', va='center', 
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

   # Plot each polygon
    if 'type' in ph_shape_data and ph_shape_data['type'] == 'FeatureCollection':
        for idx, feature in enumerate(ph_shape_data['features'], 1):
            plot_geometry(feature['geometry'], f"Building {idx}")
    else:
        # If it's a single geometry, plot it directly
        plot_geometry(ph_shape_data, "Building 1")

    # Set the title with the number of buildings detected
    ax.set_title(f'PH.Shape Results Overlay - {num_buildings} building{"s" if num_buildings > 1 else ""} detected')

    # Invert the y-axis
    ax.invert_yaxis()

    # Save the figure
    plt.savefig(output_file)
    plt.close()

    print(f"Visualization saved as {output_file}")
    print(f"Number of buildings detected: {num_buildings}")

# Example usage
# visualize_results('path/to/tif_file.tif', 'path/to/ph_shape_file.json', 'output_visualization.png')