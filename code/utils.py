
# Author: Laura Gunsalus
# University of California, San Francisco
# 2023

# Distance matrix utils


import scipy.spatial

def get_distance_matrix(single_map):
    """
    Computes the pairwise Euclidean distance matrix from 3D coordinates.

    This function takes a [x x 3] matrix representing the coordinates (x, y, z)
    of points in a 3D space and returns a [x x x] distance matrix where each 
    element (i, j) represents the Euclidean distance between points i and j.
    
    Parameters:
    - single_map: A numpy array of shape [x x 3], where each row corresponds to
                  the (x, y, z) coordinates of a point in 3D space.

    Returns:
    - A numpy array of shape [x x x], which is the pairwise distance matrix of the
      input coordinates.
    """
    
    # Extract the individual x, y, z coordinates from the input matrix
    x = single_map[:, 0]
    y = single_map[:, 1]
    z = single_map[:, 2]
    
    # Create a list of tuples, where each tuple represents a point in 3D space
    coords = [(i, j, k) for i, j, k in zip(x, y, z)]
    
    # Compute the pairwise distances between all points in the coords list
    distances = scipy.spatial.distance.pdist(coords)
    
    # Convert the condensed distance vector to a square distance matrix
    distance_matrix = scipy.spatial.distance.squareform(distances)
    
    return distance_matrix

def balanced_subsample(x,y,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys

def standard_scale(x):
    mean = np.mean(x)
    std = np.std(x)
    out = (x - mean) / std
    return out 


def map_probe_number(start_loc, stop_loc, feature, writeF):
    """
    Maps bin numbers onto feature dataframe based on overlap with given locations.

    Args:
    - start_loc: The start location of the bin to map.
    - stop_loc: The stop location of the bin to map.
    - feature: The feature number to map onto the bin.
    - writeF: The dataframe containing bin start and stop information.

    Returns:
    - None: The function directly modifies the 'writeF' dataframe.
    """
    # Iterate over each row in the dataframe to check for overlap
    for index, row in writeF.iterrows():
        track_start, track_stop = row['start'], row['stop']
        # Calculate the overlap amount between the bins
        overlap_amount = overlap(track_start, track_stop, start_loc, stop_loc)
        # If there is an overlap, assign the feature number to the 'bin_no' column
        if overlap_amount > 0:
            writeF.at[index, 'bin_no'] = feature
    # The function returns nothing as it modifies 'writeF' in place

def overlap(min1, max1, min2, max2):
    """
    Calculates the overlap amount between two ranges.

    Args:
    - min1: The start of the first range.
    - max1: The end of the first range.
    - min2: The start of the second range.
    - max2: The end of the second range.

    Returns:
    - An integer representing the amount of overlap between the two ranges.
    """
    # Return the amount of overlap, or zero if there is no overlap
    return max(0, min(max1, max2) - max(min1, min2))

def map_probe_locations(start_loc, stop_loc, feature):
    """
    Maps annotations to probes in a dataframe based on their location overlap.

    Args:
    - start_loc: The start location of the probe.
    - stop_loc: The stop location of the probe.
    - feature: The annotation feature to map to the probe.

    Returns:
    - None: The function directly modifies the 'trackF' dataframe which is assumed to be globally accessible.
    """
    # Iterate over the dataframe to map annotations to probes
    for index, row in trackF.iterrows():
        track_start, track_stop = row['start'], row['stop']
        # Calculate the overlap amount between the probe and track locations
        overlap_amount = overlap(track_start, track_stop, start_loc, stop_loc)
        # If there is an overlap, assign the feature to the 'annotation' column
        if overlap_amount > 0:
            trackF.at[index, 'annotation'] = feature
    # The function returns nothing as it modifies 'trackF' in place

def color_map(x, cmap="Spectral"):
    """
    Generates a color based on a value and a specified colormap.

    Args:
    - x: The value to map onto a color.
    - cmap: The name of the colormap to use.

    Returns:
    - A tuple representing the RGB color.
    """
    # Normalize the value 'x' by the maximum of 'probe_list'
    val = x / max(probe_list)
    # Retrieve the specified colormap
    cmap = plt.cm.get_cmap(cmap)
    # Get the RGBA values for the normalized value
    r, g, b, a = cmap(val)
    # Return the RGB values as a tuple, discarding the alpha channel
    return (r, g, b)


