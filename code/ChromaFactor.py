# Author: Laura Gunsalus
# University of California, San Francisco
# 2023

import numpy as np
from lucid.misc.channel_reducer import ChannelReducer
from plot import *

class ChromaFactor:
    def __init__(self, normalized, no_components=20):
        """
        Initializes the ChromaFactor object with normalized data and the number of components for NMF.
        
        Parameters:
        - normalized: The normalized data to be factored by NMF.
        - no_components: The number of components to reduce to using NMF.
        """
        self.normalized = normalized
        self.no_components = no_components
        self.nmf_maps = None
        self.components = None
        self.reconstruction_err = None

        self._reduce_channels()
    
    def _reduce_channels(self):
        """
        Applies Non-negative Matrix Factorization (NMF) to the normalized maps.
        Stores the results in instance variables for further use.
        """
        reshaped_maps = np.asarray(self.normalized).transpose(1, 2, 0)
        self.nmf = ChannelReducer(self.no_components, "NMF", l1_ratio=0)
        self.nmf_maps = self.nmf.fit_transform(reshaped_maps)
        self.components = self.nmf._reducer.components_
        self.reconstruction_err = self.nmf._reducer.reconstruction_err_
    
    def reconstruct_maps(self):
        """
        Reconstructs the maps by dot product of the NMF maps and components.
        Returns the new reshaped maps.
        """
        if self.components is None or self.nmf_maps is None:
            raise ValueError("The NMF components or maps have not been computed.")
        new_reshaped_maps = np.dot(self.nmf_maps, self.components)
        return new_reshaped_maps
    
    def plot_nmf_maps(self):
        """
        Plots all NMF maps using the `plot_distance_matrix` function.
        """
        if self.nmf_maps is None:
            raise ValueError("NMF maps have not been computed.")
        
        for i in range(self.nmf_maps.shape[-1]):
            plot_distance_matrix(self.nmf_maps[..., i], cmap_order="forward",
                                 hide_all=True, ax_label=None)

# Example usage:
# normalized_data = ... (some preprocessed data)
# chroma_factor = ChromaFactor(normalized_data)
# new_maps = chroma_factor.reconstruct_maps()
# print(f"Reconstruction Error: {chroma_factor.reconstruction_err}")
# chroma_factor.plot_nmf_maps()
