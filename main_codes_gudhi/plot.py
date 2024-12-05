import json
import matplotlib.pyplot as plt

def plot_polygon_from_json(json_path):
    """
    Plots a polygon from a JSON file containing GeoJSON-like structure.

    Parameters:
        json_path (str): Path to the JSON file containing the polygon data.
    """
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Ensure it's a Polygon type
    if data['type'] != 'Polygon':
        raise ValueError("The JSON file does not contain a 'Polygon' geometry type.")

    # Extract coordinates
    coordinates = data['coordinates'][0]  # Assuming a single polygon (no holes)

    # Separate x and y coordinates
    x_coords, y_coords = zip(*coordinates)

    # Plot the polygon
    plt.figure(figsize=(8, 8))
    plt.plot(x_coords, y_coords, '-o', label='Polygon')
    plt.fill(x_coords, y_coords, alpha=0.3, label='Filled Polygon')  # Optional fill
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')
    plt.title('Polygon Plot from JSON')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')  # Equal scaling for proper visualization
    plt.show()


# Example usage
if __name__ == "__main__":
    # Replace 'your_file_path.json' with the path to your JSON file
    json_path = '/Users/Rebekka/GiHub/PHshapeModified/output/visualizations/tile_test_1_building_1.json'
    plot_polygon_from_json(json_path)
