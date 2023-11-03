
# See: Comparing chromatin contact maps at scale: methods and insights

# https://www.biorxiv.org/content/10.1101/2023.04.04.535480v1 
# Gunsalus and McArthur et al., 2023
# doi: https://doi.org/10.1101/2023.04.04.535480

def insulation_track(map, window_size=10, plot=False, ax=None):
    '''
    get the insulation profile using a diamond-shaped window-based method, specially it scans along 
    the diagonal of a matrix using a W by W diamond-shaped window, calculating the average contact 
    frequency within each window. The locations at which the average contact frequency reaches a 
    local minimum are identified as candidate TAD boundaries.
    Input:
        map: n x n numpy array
        window_size: size of the dimond-shaped window
        plot: True or False
        ax: if you want to plot on an already specified axis
    Returns:
        array: n x 1 insulation track of map
    '''

    insulation_track = []

    # Select the diagonal
    for loc in range(0,len(map)):
        # Ignore if it's an edge pixel
        if loc <= window_size or loc >= len(map)-window_size:
            insulation_track.append(np.nan)
            continue
            
        # Define focal region
        focal_start = loc-window_size-1
        focal_end = loc+window_size+1
        
        window = map[focal_start:loc-1, loc+1:focal_end]
        window_mean = np.nanmean([np.exp(i) for i in list(chain(*window))])

        insulation_track.append(math.log2(window_mean))
    
    if plot:
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(insulation_track)
        ax.set_xlim(0, BINS)
        ax.set_ylabel('insulation score')

    return(np.array(insulation_track))

def downres(input_map, new_resolution = 40000, input_map_size = 2**20):
    """
    Change the resolution of the contact map.
    input:
        input_map: n x n numpy array
        new_resolution (in bp)
        input_map_size: original map size (in bp)
    output: 
        m x m numpy array, where m is the no of bp per pixel
    """
    pixel_size = round(input_map_size / new_resolution)
    resized = resize(input_map, (pixel_size,pixel_size), anti_aliasing=False)
    return resized


def DI_track(input_mat, input_map_size = 2**20,
                 new_resolution = 2000,
                 window_resolution = 10000,
                replace_ends = True,
                buffer = 50):
    """
    Calculate directionality index track 
    Source: https://zhonglab.gitbook.io/3dgenome/chapter2-computational-analysis/3.2-higer-order-data-analysis/tad-calling-algorithms
    
    Input:
        input_mat: n x n numpy array
        input_map_size: size of original map in bp
        new_resolution: resolution of intended map in bp
        window_resolution: resolution of sliding window in bp
        replace_ends: replaces ends of DI track with 0s
        buffer: how far to replace with 0
    Output:
        DI track
    """
    downres_map = downres(input_mat, new_resolution)
    no_pixels = downres_map.shape[0]
    bp_per_pixels = np.around(input_map_size / new_resolution)
    pixels_per_window = round(window_resolution / bp_per_pixels) 


    summed_map = np.nansum(downres_map, axis=0) # contact summed across one axis
    extended_map = np.concatenate([np.repeat(summed_map[0], pixels_per_window),
               summed_map,
              np.repeat(summed_map[0], pixels_per_window)]) # extend in window size each direction

    DI  = []
    for i in range(pixels_per_window, summed_map.shape[0] + pixels_per_window):
        A = extended_map[i - pixels_per_window:i].sum()
        B = extended_map[i:i+pixels_per_window].sum()
        E = (A + B) / 2

        sign = (B - A)/ abs(B - A)
        upstream = ((A - E)**2) / E
        downstream = ((B - E)**2) / E
        score = sign* (upstream + downstream)
        DI.append(score)
    if replace_ends:
        return np.array([0] * buffer   + DI[buffer:len(DI) - buffer] + [0] * buffer)
    return np.array(DI)