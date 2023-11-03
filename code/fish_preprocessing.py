import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
import scipy.spatial.distance
from tqdm import tqdm_notebook
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from scipy.stats import linregress

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import figure
import seaborn as sns
from scipy.spatial import distance_matrix
import scipy.spatial.distance
from matplotlib import cm
import re
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors

def get_distance_matrix(cellF):
    """
    Return distance matrix from 3d coordinates
    """
    x = cellF["X(nm)"].values
    y = cellF["Y(nm)"].values
    z = cellF["Z(nm)"].values
    coords = [(i,j,k) for i,j,k in zip(z,y,z)]
    distances = scipy.spatial.distance.pdist(coords)
    distance_matrix = scipy.spatial.distance.squareform(distances)
    return distance_matrix

def pull_chrom(chrom_number, chromF):
    """
    Select chromosome subset from full dataset. Split columns. 
    """ 
    ex_copy = chromF[chromF['Chromosome copy number'] == chrom_number] # number of unique contacts per cell
    cellF = ex_copy.copy()
    
    cellF[['chr','pos']] = cellF["Genomic coordinate"].str.split(":", n = 1, expand = True) 
    cellF[['start','end']] = cellF["pos"].str.split("-", n = 1, expand = True) 
    
    cellF['start'] = cellF['start'].astype('int')
    cellF['end'] = cellF['end'].astype('int')
    cellF['mid'] = ((cellF['start'] + cellF['end'])/2).astype(np.int)
        
    return cellF

def pull_cell(cell_number, chromF):
    """
    Select cell subset from full dataset. Split columns. 
    Equivalent to pull_chrom but different formatting. 
    """ 
    ex_copy = chromF[chromF['cell number'] == cell_number] # number of unique contacts per cell
    cellF = ex_copy.copy()
    
    cellF[['chr','pos']] = cellF["genomic coordinate"].str.split(":", n = 1, expand = True) 
    cellF[['start','end']] = cellF["pos"].str.split("-", n = 1, expand = True) 
    
    cellF['start'] = cellF['start'].astype('int')
    cellF['end'] = cellF['end'].astype('int')
    cellF['mid'] = ((cellF['start'] + cellF['end'])/2).astype(np.int)
        
    return cellF

# old version- changed column names to lower case
# def get_distance_matrix(cellF):
#     """
#     Return distance matrix from 3d coordinates
#     """
#     x = cellF["X(nm)"].values
#     y = cellF["Y(nm)"].values
#     z = cellF["Z(nm)"].values
#     coords = [(i,j,k) for i,j,k in zip(z,y,z)]
#     distances = scipy.spatial.distance.pdist(coords)
#     distance_matrix = scipy.spatial.distance.squareform(distances)
#     return distance_matrix

def get_distance_matrix(cellF):
    """
    Return distance matrix from 3d coordinates
    """
    cellF.columns= cellF.columns.str.strip().str.lower()
    x = cellF["x(nm)"].values
    y = cellF["y(nm)"].values
    z = cellF["z(nm)"].values
    coords = [(i,j,k) for i,j,k in zip(z,y,z)]
    distances = scipy.spatial.distance.pdist(coords)
    distance_matrix = scipy.spatial.distance.squareform(distances)
    return distance_matrix

def interpolate_missing_coords(distance_matrix, missing_ratio = 0.25):
    """
    Fill nan values in distance matrix with piecewise linear interpolant.
    """
    distance_matrix = distance_matrix.copy()
    missing_coords = np.isnan(distance_matrix)
    present_coords = ~missing_coords
    
    # get missing ratio
    missing = (missing_coords.sum()) / (present_coords.sum() + missing_coords.sum())
    if missing >= missing_ratio:
        raise Exception("Warning: %.0f%% of matrix is missing." % (100 * missing))
        
    # interpolate
    else:
        present_idx = np.where(present_coords)[0]
        present_vals = distance_matrix[present_coords]
        missing_idx  = np.where(missing_coords)[0]
        distance_matrix[missing_coords] = np.interp(missing_idx, present_idx, present_vals)
        return distance_matrix


def build_transcription_labels(cellF):
    """
    Pull transcription state to use as graph label.
    """
    gene_values = cellF[~pd.isnull(cellF['Transcription'])]
    gene_values = gene_values.sort_values(by=['Gene names'])

    unique_genes = gene_values['Gene names']
    unique_states = gene_values['Transcription']

    split_genes = [gene.split(";") for gene in unique_genes]              # split double genes
    split_genes = [item for sublist in split_genes for item in sublist]   # flatten list
    split_genes = [i for i in split_genes if i]                           # remove empty strings

    split_states = [gene.split(";") for gene in unique_states]            # split doubles
    split_states = [item for sublist in split_states for item in sublist] # flatten list
    split_states = [i for i in split_states if i]                         # remove empty strings
    split_states = [1 if state == 'on' else 0 for state in split_states]  # map to 0/1
    return split_states

def return_gene_mapping(chromF):
    """
    Create mapping from idx to gene, from gene to idx
    """
    cellF = pull_chrom(1, chromF)

    gene_values = cellF[~pd.isnull(cellF['Transcription'])]
    gene_values = gene_values.sort_values(by=['Gene names'])

    unique_genes = gene_values['Gene names']
    unique_states = gene_values['Transcription']

    split_genes = [gene.split(";") for gene in unique_genes]              # split double genes
    split_genes = [item for sublist in split_genes for item in sublist]   # flatten list
    split_genes = [i for i in split_genes if i]                           # remove empty strings

    indices = list(range(len(split_genes)))
    gene_to_idx = dict(zip(split_genes, indices)) 
    return split_genes, gene_to_idx

def store_all_fish(min_graph, max_graph, chromF, resolution = 500000):
    """
    Store all data in dictionary where key is cell_id and value is [x,y]
    """
    cell_counter = 0
    failed_counter = 0
    storage_dict = dict()
    for cell_no in tqdm_notebook(range(min_graph, max_graph+1)):
        try:
            #resolution = 500000
            cellF = pull_chrom(cell_no, chromF)
            cellF['reduced'] =cellF.start//resolution
            sub = cellF[['X(nm)', 'Y(nm)', 'Z(nm)', 'reduced', 'Chromosome copy number']]
            cell_mean = sub.groupby('reduced').mean()
            distance_matrix = get_distance_matrix(cell_mean)
            distance_matrix = interpolate_missing_coords(distance_matrix)
            labels = build_transcription_labels(cellF)
            storage_dict[cell_counter] = [distance_matrix, labels]
            cell_counter += 1
        except:
            failed_counter += 1        
    return storage_dict, failed_counter

def build_normalized_map(cellF, plot_maps=False):
    """
    Normalize maps to distance using method described in paper.
    """
    distance_matrix = get_distance_matrix(cellF)
    distance_matrix = interpolate_missing_coords(distance_matrix)
    contact_thresh = 500
    contact_map_combined = distance_matrix
    # thresholded distance matrix
        # ratio of rows above contact threshold
    #contact_map_combined = np.sum(distance_matrix < contact_thresh, axis=0) / np.sum(np.isnan(distance_matrix)==False, axis=0)

    # normalize genomic distance effects

    #for contact_gaussian_sigma in np.arange(0,3,0.03):
    # Generate correlation map
    contact_gaussian_sigma = 1.9
    hic_gaussian_sigma = 1.9

    # collapsed spatial distance matrix
    contact_entries = contact_map_combined[np.triu_indices(len(contact_map_combined),1)]

    # distance matrix from genomic (not spatial) distance
    indices = np.expand_dims(cellF.mid.values, axis=1)
    genomic_distance_map = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(indices))
    small_val = 0.0000001
    genomic_distance_map[genomic_distance_map == 0] = small_val

    # collapsed genomic distance matrix
    genomic_distance_entries = genomic_distance_map[np.triu_indices(len(genomic_distance_map),1)]

    # keep non-zero distance entries 
    kept = (genomic_distance_entries > 0) * (contact_entries > 0)

    # correlation between spatial and genomic distance
    contact_lr = linregress(np.log(genomic_distance_entries[kept]), np.log(contact_entries[kept]))

    # normalize genomic distance contact map
    contact_norm_map = np.exp(np.log(genomic_distance_map) * contact_lr.slope + contact_lr.intercept)


    # set 0 distances along the axis to 1
    for _i in range(contact_norm_map.shape[0]):
        contact_norm_map[_i,_i] = 1

    # normalize contact map by distance
    contact_normed_map = contact_map_combined / contact_norm_map
    contact_corr_map_combined = np.corrcoef(gaussian_filter(contact_normed_map, contact_gaussian_sigma))

    if plot_maps:
        color_map = plt.cm.RdBu
        color_map_r = plt.cm.RdBu_r
        font_size = 20
        figure_size = 20

        fig, axs = plt.subplots(2,2,figsize=(figure_size,figure_size))

        axs[0, 0].matshow(distance_matrix, cmap = color_map)
        title = "spatial distance contact matrix"
        axs[0, 0].set_title(title, fontsize=font_size)
        axs[0, 1].matshow(genomic_distance_map, cmap = color_map)
        title = "genomic distance matrix"
        axs[0, 1].set_title(title, fontsize=font_size)
        axs[1, 0].matshow(contact_normed_map, cmap = color_map)
        title = "normalized spatial contact matrix"
        axs[1, 0].set_title(title, fontsize=font_size)
        axs[1, 1].matshow(contact_corr_map_combined, cmap = color_map_r)
        title = "gaussian filter- contact matrix"
        axs[1, 1].set_title(title, fontsize=font_size)

        #for ax in axs.flat:
        #    ax.set(xlabel='x-label', ylabel='y-label')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            #ax.label_outer()
            ax.axis('off');


    return contact_corr_map_combined

def normalize_map(contact_map_combined, cellF, plot_maps=False):
    """
    Normalize maps to distance using method described in paper.
    """
    # thresholded distance matrix
        # ratio of rows above contact threshold
    #contact_map_combined = np.sum(distance_matrix < contact_thresh, axis=0) / np.sum(np.isnan(distance_matrix)==False, axis=0)

    # normalize genomic distance effects

    #for contact_gaussian_sigma in np.arange(0,3,0.03):
    # Generate correlation map
    contact_gaussian_sigma = 1.9
    hic_gaussian_sigma = 1.9

    # collapsed spatial distance matrix
    contact_entries = contact_map_combined[np.triu_indices(len(contact_map_combined),1)]

    # distance matrix from genomic (not spatial) distance
    indices = np.expand_dims(cellF.mid.values, axis=1)
    genomic_distance_map = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(indices))
    small_val = 0.0000001
    genomic_distance_map[genomic_distance_map == 0] = small_val

    # collapsed genomic distance matrix
    genomic_distance_entries = genomic_distance_map[np.triu_indices(len(genomic_distance_map),1)]

    # keep non-zero distance entries 
    kept = (genomic_distance_entries > 0) * (contact_entries > 0)

    # correlation between spatial and genomic distance
    contact_lr = linregress(np.log(genomic_distance_entries[kept]), np.log(contact_entries[kept]))

    # normalize genomic distance contact map
    contact_norm_map = np.exp(np.log(genomic_distance_map) * contact_lr.slope + contact_lr.intercept)


    # set 0 distances along the axis to 1
    for _i in range(contact_norm_map.shape[0]):
        contact_norm_map[_i,_i] = 1

    # normalize contact map by distance
    contact_normed_map = contact_map_combined / contact_norm_map
    contact_corr_map_combined = np.corrcoef(gaussian_filter(contact_normed_map, contact_gaussian_sigma))

    if plot_maps:
        color_map = plt.cm.RdBu
        color_map_r = plt.cm.RdBu_r
        font_size = 20
        figure_size = 20

        fig, axs = plt.subplots(2,2,figsize=(figure_size,figure_size))

        axs[0, 0].matshow(distance_matrix, cmap = color_map)
        title = "spatial distance contact matrix"
        axs[0, 0].set_title(title, fontsize=font_size)
        axs[0, 1].matshow(genomic_distance_map, cmap = color_map)
        title = "genomic distance matrix"
        axs[0, 1].set_title(title, fontsize=font_size)
        axs[1, 0].matshow(contact_normed_map, cmap = color_map)
        title = "normalized spatial contact matrix"
        axs[1, 0].set_title(title, fontsize=font_size)
        axs[1, 1].matshow(contact_corr_map_combined, cmap = color_map_r)
        title = "gaussian filter- contact matrix"
        axs[1, 1].set_title(title, fontsize=font_size)

        #for ax in axs.flat:
        #    ax.set(xlabel='x-label', ylabel='y-label')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            #ax.label_outer()
            ax.axis('off');


    return contact_corr_map_combined

def distance_normalize_map(contact_map_combined, cellF):
    contact_entries = contact_map_combined[np.triu_indices(len(contact_map_combined),1)]

    # distance matrix from genomic (not spatial) distance
    indices = np.expand_dims(cellF.mid.values, axis=1)
    genomic_distance_map = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(indices))
    small_val = 0.0000001
    genomic_distance_map[genomic_distance_map == 0] = small_val

    # collapsed genomic distance matrix
    genomic_distance_entries = genomic_distance_map[np.triu_indices(len(genomic_distance_map),1)]

    # keep non-zero distance entries 
    kept = (genomic_distance_entries > 0) * (contact_entries > 0)

    # correlation between spatial and genomic distance
    contact_lr = linregress(np.log(genomic_distance_entries[kept]), np.log(contact_entries[kept]))

    # normalize genomic distance contact map
    contact_norm_map = np.exp(np.log(genomic_distance_map) * contact_lr.slope + contact_lr.intercept)


    # set 0 distances along the axis to 1
    for _i in range(contact_norm_map.shape[0]):
        contact_norm_map[_i,_i] = 1

    # normalize contact map by distance
    contact_normed_map = contact_map_combined / contact_norm_map
    #contact_corr_map_combined = np.corrcoef(gaussian_filter(contact_normed_map, contact_gaussian_sigma))
    return contact_normed_map

def contact_pca(contact_map):
    """
    Resturn first principal compenent of map.
    """
    contact_model = PCA(1)
    contact_model.fit(contact_map)
    contact_pc1_combined = np.reshape(contact_model.fit_transform(contact_map), -1)
    return contact_pc1_combined
    
def subset_by_window(cellF, start_loc, end_loc):
    subF = cellF[cellF.mid >= start_loc]
    subF = subF[subF.mid <= end_loc]
    return subF

#############################################################################################
# Transcription
#############################################################################################

def get_gene_pos(cellF, cur_gene):
    """
    Return gene mid position of gene of interest,
    """
    cur_gene = cur_gene + ";"
    gene_indices = cellF['Gene names'].str.contains(cur_gene, regex=False) 
    gene_indices = gene_indices.fillna(False) 
    loc = cellF[gene_indices]
    if loc.empty:
        raise NameError('Gene provided not present in dataframe.')
    else:
        return loc.mid.item()

def get_boundary_pos(cellF, window_location):
    """
    Return index of matrix containing gene of interest,
    """
    pos_counter = 0
    boundary_pos = None
    last_pos = None
    error = 100000
    for val in cellF['start'].values:
        last_pos = pos_counter
        if abs(val-window_location) < error:
            boundary_pos = pos_counter
        pos_counter += 1
    if boundary_pos == None: # position out of range
        boundary_pos = last_pos 
        print("Boundary position out of range")
    assert(boundary_pos != None)
    return boundary_pos

def paint_boundary_box(in_matrix, start_boundary, end_boundary):
    """
    Paints boundary box at start and end locations.
    """
    if start_boundary < 0:
        print("Start boundary out of range")
        start_boundary = 0
    if end_boundary > in_matrix.shape[0]:
        print("End boundary out of range")
        end_boundary = in_matrix.shape[0] - 1
    adj_matrix = in_matrix.copy()
    adj_matrix[start_boundary,start_boundary:end_boundary] = np.nan
    adj_matrix[end_boundary,start_boundary:end_boundary] = np.nan
    adj_matrix[start_boundary:end_boundary,start_boundary] = np.nan
    adj_matrix[start_boundary:end_boundary,end_boundary] = np.nan
    return adj_matrix

def paint_map(in_matrix, position):
    """
    Paints row and column of position with nans'
    """
    adj_matrix = in_matrix.copy()
    adj_matrix[position] = np.nan
    adj_matrix[:, position]  = np.nan
    return adj_matrix

#############################################################################################
# Plotting
#############################################################################################

class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
    
def plot_distance_matrix(distance_matrix, 
                         title=None, 
                         figure_size = 10, 
                         font_size= 15,
                         map_color="yellow",
                         cmap_order = "rev",
                         ax_label='Genomic position (Mb)', 
                         vmax = None,
                         vmin = None,
                         tick_positions = None,
                         center_color = True,
                         midpoint_normalize=False,
                         midpoint_value = 0,
                         colorbar_labels=None):
    """
    Plot a distance matrix.
    """
    tick_label_length=2
    tick_label_width=0.5
    tick_font_size = 20

    def convert_to_mb(in_pos):
        in_pos = in_pos/1000000
        return round(in_pos,1) 

    fig, ax = plt.subplots(figsize=(figure_size,figure_size))
    
    # Map Color
    if vmax==None:
        vmin = np.nanmin(distance_matrix)
        vmax = np.nanmax(distance_matrix)

    if cmap_order == "rev":
        color_map = plt.cm.RdBu_r
    else:
        color_map = plt.cm.RdBu
    color_map.set_bad(color=map_color)

    #norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax = vmax, vcenter=0)
    if midpoint_normalize:
        norm = MidpointNormalize(midpoint=midpoint_value,vmin=vmin, vmax=vmax)
        im = ax.matshow(distance_matrix, cmap = color_map, norm=norm)
    else:
        im = ax.matshow(distance_matrix, cmap = color_map, vmin=vmin, vmax=vmax)
    #plt.colorbar(cur)
    #colorbar = clippedcolorbar(cur)

    # Draw Map
    #im = ax.matshow(distance_matrix, cmap = color_map) 
    
    # Ticks and axis labels
    if tick_positions:
        [start_loc,end_loc] = tick_positions
        tick_labels = [convert_to_mb(start_loc),convert_to_mb(end_loc)]
        tick_locs = [-0.5, len(distance_matrix)-0.5]
        ax.set_xticks(tick_locs, minor=False)
        ax.set_yticks(tick_locs, minor=False)
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
    if ax_label is not None:
        ax.set_xlabel(ax_label, labelpad=20, fontsize=font_size)
        ax.set_ylabel(ax_label, labelpad=-10, fontsize=font_size)
    ax.tick_params('both', labelsize=tick_font_size, 
                   width=tick_label_width, length = tick_label_length,
                   pad=1)
    
    # Color bar
    if colorbar_labels:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='6%', pad="2%")
        cb = plt.colorbar(im, cax=cax, orientation='vertical', 
                          extend='neither')
        cb.ax.minorticks_off()
        cb.ax.tick_params(labelsize=font_size, width=tick_label_width, length=tick_label_length-1,pad=1)
        [i[1].set_linewidth(tick_label_width) for i in cb.ax.spines.items()]
        # border
        cb.outline.set_linewidth(tick_label_width)
        if colorbar_labels is not None:
            cb.set_label(colorbar_labels, fontsize=font_size, labelpad=20, rotation=270)
            
    # Title
    ax.set_title(title, fontsize=font_size)
    #ax.axis('off');

def plot_pc_combined(pca_data, map_data):
    """
    Plot contact map with first principal component.
    """
    fig, ax = plt.subplots(figsize=(2, 4), dpi=600)
    grid = plt.GridSpec(2, 1, height_ratios=[1,4], hspace=0, wspace=0.)

    # PC1 Plot
    contact_ax = plt.subplot(grid[0])
    contact_ax.bar(np.where(pca_data>=0)[0],
                   pca_data[pca_data>=0],
                   width=1, color='r', label='A')
    contact_ax.bar(np.where(pca_data<0)[0],
                   pca_data[pca_data<0],
                   width=1, color='b', label='B')
    contact_ax.set_xticklabels([])
    contact_ax.set_yticks([-10,0,10,20])
    contact_ax.set_ylabel("Imaging PC1", fontsize=5, labelpad=0)
    contact_ax.yaxis.set_tick_params(labelsize=5)

    # Contact Map
    hic_ax = plt.subplot(grid[1], sharex=contact_ax)
    color_map_r = plt.cm.RdBu_r
    hic_ax.matshow(map_data, cmap = color_map_r)
    hic_ax.set_aspect('equal')
    hic_ax.axis('off');
    plt.gcf().subplots_adjust(bottom=0.4)
    plt.show()