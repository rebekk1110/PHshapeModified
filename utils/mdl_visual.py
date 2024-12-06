import logging
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import shapely
import geopandas as gpd
import pandas as pd

def plot_gt_vs_simplified(gt_gdf, simplified_gdf, eval_results, output_path, tile_name):
    logging.info(f"Starting plot generation for tile: {tile_name}")

    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    ax.set_aspect('equal', 'box')
    ax.autoscale_view()

    def plot_polygon(ax, polygon, color, linestyle='-'):
        if polygon.geom_type == 'Polygon':
            ax.plot(*polygon.exterior.xy, color=color, linestyle=linestyle, linewidth=2)
            for interior in polygon.interiors:
                ax.plot(*interior.xy, color=color, linestyle='--', linewidth=1.5)
        elif polygon.geom_type == 'MultiPolygon':
            for geom in polygon.geoms:
                plot_polygon(ax, geom, color, linestyle)

    # Plot ground truth
    gt_gdf.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2)
    for _, row in gt_gdf.iterrows():
        plot_polygon(ax, row['geometry'], 'red')
    logging.info(f"Plotted {len(gt_gdf)} ground truth polygons")

    # Plot simplified
    simplified_gdf.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=2)
    for _, row in simplified_gdf.iterrows():
        plot_polygon(ax, row['geometry'], 'blue')
    logging.info(f"Plotted {len(simplified_gdf)} simplified polygons")

    ax.set_xlim(gt_gdf.total_bounds[[0, 2]])
    ax.set_ylim(gt_gdf.total_bounds[[1, 3]])

    gt_line = plt.Line2D([], [], color='red', linestyle='-', label='Ground Truth')
    gt_interior = plt.Line2D([], [], color='red', linestyle='--', label='Ground Truth Interior')
    simp_line = plt.Line2D([], [], color='blue', linestyle='-', label='Simplified')
    simp_interior = plt.Line2D([], [], color='blue', linestyle='--', label='Simplified Interior')
    ax.legend(handles=[gt_line, gt_interior, simp_line, simp_interior], loc='upper right')

    ax.set_title(f'Ground Truth (Red) vs Simplified (Blue) Buildings - {tile_name}')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    stats_text = f"Overall Statistics:\nMean IoU: {eval_results['IOU'].mean():.4f}\nMean HD: {eval_results['HD'].mean():.4f}"
    if 'Area' in eval_results.columns:
        stats_text += f"\nMean Area: {eval_results['Area'].mean():.4f}"
    if 'Perimeter' in eval_results.columns:
        stats_text += f"\nMean Perimeter: {eval_results['Perimeter'].mean():.4f}"
    
    plt.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    for idx, row in eval_results.iterrows():
        centroid = row['geometry'].centroid
        building_stats = f"Building {idx+1}:\nIoU: {row['IOU']:.4f}\nHD: {row['HD']:.4f}"
        if 'Area' in eval_results.columns:
            building_stats += f"\nArea: {row['Area']:.4f}"
        if 'Perimeter' in eval_results.columns:
            building_stats += f"\nPerim: {row['Perimeter']:.4f}"
        plt.annotate(building_stats, (centroid.x, centroid.y), xytext=(10, 10), 
                     textcoords="offset points", fontsize=8, 
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.5"))

    plt.tight_layout()

    logging.info(f"Saving plot to: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Plot saved to {output_path}")

def plot_initial_separation(raster_data, labeled_buildings, output_path, transform):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot original raster
    ax1.imshow(raster_data, cmap='gray', extent=[0, raster_data.shape[1], raster_data.shape[0], 0])
    ax1.set_title('Original Raster')
    ax1.set_xlim(0, raster_data.shape[1])
    ax1.set_ylim(raster_data.shape[0], 0)
    
    # Plot labeled buildings
    unique_labels = np.unique(labeled_buildings)
    num_labels = len(unique_labels) - 1  # Subtract 1 to exclude background
    cmap = plt.get_cmap('tab20')
    labeled_buildings_colored = np.zeros((*labeled_buildings.shape, 3))
    
    for i, label in enumerate(unique_labels[1:]):  # Skip background (0)
        mask = labeled_buildings == label
        color = cmap(i / num_labels)[:3]
        labeled_buildings_colored[mask] = color
    
    ax2.imshow(labeled_buildings_colored, extent=[0, raster_data.shape[1], raster_data.shape[0], 0])
    ax2.set_title(f'Separated Buildings (Total: {num_labels})')
    ax2.set_xlim(0, raster_data.shape[1])
    ax2.set_ylim(raster_data.shape[0], 0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def drawmultipolygon(polygon:shapely.geometry, pts:np.ndarray=None, title:str="", savepath:str=""):
    fcolor=["r","b","g","c"]
    fig, axs = plt.subplots()
    if polygon.geom_type=="MultiPolygon":
        for gi, geom in enumerate(polygon.geoms):
            xs, ys = geom.exterior.xy
            axs.fill(xs, ys, alpha=0.5, fc=fcolor[gi%len(fcolor)], ec='black')
            inters = [list(inter.coords) for inter in geom.interiors]
            for inter in inters:
                inter = np.asarray(inter)
                axs.plot(inter[:, 0], inter[:, 1], c='yellow')
    else:
        xs, ys = polygon.exterior.xy
        axs.fill(xs, ys, alpha=0.5, fc='r', ec='black')
        inters = [list(inter.coords) for inter in polygon.interiors]
        for inter in inters:
            inter = np.asarray(inter)
            axs.plot(inter[:,0], inter[:,1], c='yellow')

    if pts is not None:
        axs.scatter(pts[:, 0], pts[:, 1], c='C1', edgecolor='black', s=2)

    plt.title(title)

    if savepath=="":
        plt.show()
    else:
        plt.savefig(savepath, dpi=300)
    plt.close()

def show_ifd_shape(org_data:np.ndarray, ifd_topP_coords:np.ndarray, title:str="", savepath:str=""):
    fig = plt.figure(figsize=(10, 10))

    ax1 = fig.add_subplot(111)
    ax1.plot(org_data[:, 0], org_data[:, 1], c="blue", marker="o", mfc="blue")
    for i in range(len(org_data)):
        ax1.text(org_data[i, 0], org_data[i, 1], f"{i - 1}")
    ax1.set_xlabel('x'), ax1.set_ylabel('y')

    ax1.plot(ifd_topP_coords[:, 0], ifd_topP_coords[:, 1], c="orange", linewidth=2, marker="o", mfc="red")
    for i in range(len(ifd_topP_coords)):
        ax1.text(ifd_topP_coords[i, 0], ifd_topP_coords[i, 1], f"{i - 1}")

    plt.suptitle(title)
    plt.tight_layout()

    if savepath=="":
        plt.show()
        plt.close()
    else:
        plt.savefig(savepath, dpi=300)
        plt.close()
