
# Author: Laura Gunsalus
# University of California, San Francisco
# 2023

# Plotting functionality 

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import TwoSlopeNorm  
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

def plot_distance_matrix(distance_matrix, title=None, figure_size=10, font_size=15,
                         alt_cmap=None, map_color="yellow", cmap_order="rev",
                         ax_label='Genomic position (Mb)', vmax=None, vmin=None,
                         tick_positions=None, midpoint_normalize=False, midpoint_value=0,
                         colorbar_labels=None, hide_all=False):
    """
    Plot a distance matrix with various customization options.
    
    Parameters:
    - distance_matrix: 2D array representing the distances.
    - title: Title of the plot.
    - figure_size: Size of the figure.
    - font_size: Font size for labels and title.
    - alt_cmap: Alternative colormap if provided.
    - map_color: Color to use for bad values in colormap.
    - cmap_order: If 'rev', reverse the colormap.
    - ax_label: Label for the x and y axis.
    - vmax, vmin: Colorbar value range.
    - tick_positions: Positions for the ticks.
    - midpoint_normalize: If True, normalize colorbar with a midpoint.
    - midpoint_value: Value at which the midpoint is set for normalization.
    - colorbar_labels: Labels for the colorbar.
    - hide_all: If True, hide the axis labels and ticks.
    """
    # Set the default colormap and reverse it if needed
    if alt_cmap is not None:
        color_map = alt_cmap
    else:
        color_map = plt.cm.RdBu_r if cmap_order == "rev" else plt.cm.RdBu
    color_map.set_bad(color=map_color)

    # Initialize figure and axis
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))

    # Determine the min and max if not provided
    vmin = vmin or np.nanmin(distance_matrix)
    vmax = vmax or np.nanmax(distance_matrix)

    # Apply normalization if required
    if midpoint_normalize:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=midpoint_value, vmax=vmax)
    else:
        norm = None

    # Create the matrix plot
    im = ax.matshow(distance_matrix, cmap=color_map, norm=norm, vmin=vmin, vmax=vmax)

    # Configure axis ticks and labels
    if tick_positions:
        start_loc, end_loc = tick_positions
        tick_labels = [round(pos / 1e6, 1) for pos in (start_loc, end_loc)]
        tick_locs = [-0.5, len(distance_matrix) - 0.5]
        ax.set_xticks(tick_locs)
        ax.set_yticks(tick_locs)
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)

    # Set axis labels
    if ax_label:
        ax.set_xlabel(ax_label, labelpad=20, fontsize=font_size)
        ax.set_ylabel(ax_label, labelpad=-10, fontsize=font_size)

    # Adjust tick parameters
    ax.tick_params('both', labelsize=font_size, width=0.5, length=2, pad=1)

    # Add colorbar with labels if provided
    if colorbar_labels:
        cax = make_axes_locatable(ax).append_axes('right', size='6%', pad='2%')
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(colorbar_labels, fontsize=font_size, labelpad=20, rotation=270)
        cb.ax.tick_params(labelsize=font_size)

    # Set plot title
    if title and not hide_all:
        ax.set_title(title, fontsize=font_size)

    # Hide all axis labels and ticks if requested
    if hide_all:
        ax.axis('off')

    # Display the plot
    plt.show()


    
def plot_in_3d(cellF, point_size=250):
    """
    Plots 3D coordinates of cells from a DataFrame.

    Parameters:
    - cellF: A pandas DataFrame with columns 'x', 'y', 'z' representing the coordinates,
             and the index representing the cell identifiers.
    - point_size: The size of the points to be plotted.

    The function creates a 3D scatter plot with customized aesthetics, where grid lines are
    removed, tick labels are hidden, and the spines are made transparent.
    """
    
    # Create a new figure with a white background
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100, facecolor='w', edgecolor='k', subplot_kw={'projection': '3d'})

    ax = fig.add_subplot(111, projection='3d')

    # Plot the line connecting the points in dark grey, semi-transparent
    ax.plot(xs=cellF['x'].values,
            ys=cellF['y'].values,
            zs=cellF['z'].values,
            color="darkgrey", linewidth=2, alpha=0.6)

    # Scatter plot the points, colored by their index values
    scatter = ax.scatter(xs=cellF['x'].values,
                         ys=cellF['y'].values,
                         zs=cellF['z'].values,
                         s=point_size,
                         c=cellF.index.values,
                         alpha=0.6,
                         cmap='PRGn')

    # Customize the aesthetics by removing grid lines and tick labels
    ax.grid(False)  
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Make the spines (the edges of the figure) transparent
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Remove the ticks as they are unnecessary in this context
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    ax.set_zticks([])

    # Show the plot
    plt.show()


