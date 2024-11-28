"""
@File           : mdl_visual.py
@Author         : Gefei Kong (modified by Assistant)
@Time           : Current Date
------------------------------------------------------------------------------------------------------------------------
@Description    : Visualization functions for the project
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import shapely
import rasterio

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

# New plot functions
def plot_buildings(raster_data, buildings, output_path, transform, tile_name):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(raster_data, cmap='gray', alpha=0.5, extent=[0, raster_data.shape[1], raster_data.shape[0], 0])

    for building in buildings:
        coords = np.array(building.exterior.coords)
        rows, cols = rasterio.transform.rowcol(transform, coords[:, 0], coords[:, 1])
        ax.plot(cols, rows, color='red', linewidth=2)

    ax.set_title(f'{tile_name} Detected Buildings (Total: {len(buildings)})')
    ax.set_xlim(0, raster_data.shape[1])
    ax.set_ylim(raster_data.shape[0], 0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_simplified_buildings(raster_data, original_buildings, simplified_buildings, output_path, transform, tile_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot original buildings
    ax1.imshow(raster_data, cmap='gray', alpha=0.5, extent=[0, raster_data.shape[1], raster_data.shape[0], 0])
    for building in original_buildings:
        coords = np.array(building.exterior.coords)
        rows, cols = rasterio.transform.rowcol(transform, coords[:, 0], coords[:, 1])
        ax1.plot(cols, rows, color='red', linewidth=2)
    ax1.set_title(f'{tile_name} Original Buildings (Total: {len(original_buildings)})')
    ax1.set_xlim(0, raster_data.shape[1])
    ax1.set_ylim(raster_data.shape[0], 0)

    # Plot simplified buildings
    ax2.imshow(raster_data, cmap='gray', alpha=0.5, extent=[0, raster_data.shape[1], raster_data.shape[0], 0])
    for building in simplified_buildings:
        coords = np.array(building.exterior.coords)
        rows, cols = rasterio.transform.rowcol(transform, coords[:, 0], coords[:, 1])
        ax2.plot(cols, rows, color='blue', linewidth=2)
    ax2.set_title(f'{tile_name} Simplified Buildings (Total: {len(simplified_buildings)})')
    ax2.set_xlim(0, raster_data.shape[1])
    ax2.set_ylim(raster_data.shape[0], 0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_gt_vs_simplified(gt_gdf, simplified_gdf, eval_results, output_path, tile_name):
    fig, ax = plt.subplots(figsize=(12, 12))
    
    gt_gdf.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2, label='Ground Truth')
    simplified_gdf.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=2, label='Simplified')
    
    ax.set_title(f'{tile_name} Ground Truth vs Simplified Buildings')
    ax.legend()
    ax.axis('off')
    
    stats_text = f"Mean IoU: {eval_results['IOU'].mean():.4f}\n"
    stats_text += f"Mean Hausdorff Distance: {eval_results['HD'].mean():.4f}"
    plt.text(0.05, 0.05, stats_text, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

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
    print(f"Initial separation plot saved to {output_path}")