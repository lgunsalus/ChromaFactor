# Author: Laura Gunsalus
# University of California, San Francisco
# 2023

# Preprocessing and handling of Su et al. dataset

def pull_bulk_cells(cell_list, chromF):
    """
    Select chromosome subset from full dataset. Split columns. 
    """ 
    ex_copy = chrom21[chrom21['Chromosome copy number'].isin(cell_list)] # number of unique contacts per cell
    cellF = ex_copy.copy()
    
    cellF[['chr','pos']] = cellF["Genomic coordinate"].str.split(":", n = 1, expand = True) 
    cellF[['start','end']] = cellF["pos"].str.split("-", n = 1, expand = True) 
    
    cellF['start'] = cellF['start'].astype('int')
    cellF['end'] = cellF['end'].astype('int')
    cellF['mid'] = ((cellF['start'] + cellF['end'])/2).astype(np.int)
        
    return cellF

def reduce_map(cellF, resolution = 1, method="mean"):
    
    # collapse by resolution
    cellF['reduced'] =cellF.start//resolution
    
    # collapse by cell number
    sub = cellF[['X(nm)', 'Y(nm)', 'Z(nm)', 'reduced', 'mid', 'Chromosome copy number']]
    if method=="mean":
        fish_mean = sub.groupby('reduced').mean()
    else:
        fish_mean = sub.groupby("reduced").median()
    return fish_mean

def get_cellids_by_transcription(max_graph,cur_idx,chromF):
    """
    get cell ids with positive label for gene.
    """
    positive_cells = []
    negative_cells = []
    for cell_no in tqdm_notebook(range(1, max_graph)):
        cellF = pull_chrom(cell_no, chromF)
        labels= build_transcription_labels(cellF)
        cur_label = labels[cur_idx]
        if cur_label: # positive cell
            positive_cells.append(cell_no)
        else:
            negative_cells.append(cell_no)
    return positive_cells, negative_cells