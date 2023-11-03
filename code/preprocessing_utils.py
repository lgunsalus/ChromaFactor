
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
import pickle 
import os

import pybedtools
from pybedtools import BedTool
from Bio import SeqIO
from Bio.Seq import Seq

import pickle
import pandas as pd
import numpy as np
import os 
from tqdm import tqdm_notebook
import csv
import networkx as nx
import sys
import time
from scipy import stats
import matplotlib.pyplot as plt

###############################################################################
# Raw Single Cell Dataset Creation
###############################################################################

def read_chrom_sizes(chrom_sizes_file):
    """
    create a dictionary mapping from chromosomes to chromosome sizes
    """ 
    chrom_sizes = pd.read_csv(chrom_sizes_file, header=None,sep='\t')
    chrom_sizes.columns = ['chr', 'sizes']
    chrom_sizes["chr"] = chrom_sizes['chr'].map(lambda x: x.strip('chr'))

    return dict(zip(chrom_sizes.chr.values, chrom_sizes.sizes.values))

def get_gc_content(sequence):
    """Return GC content of sequence."""
    return ((sequence.count('G') + sequence.count('C'))/len(sequence))

class scRawDataset(object):
    """
    Produces a mapping table from chromosome location to features: sequence, label, etc.
    """
    def __init__(self,process_dir, run_name, cell_type, verbose=True, build="hg38"):
        
        self.verbose= verbose
        self.process_dir = process_dir
        self.run_name = run_name
        self.cell_type = cell_type
        
        if build == "hg38":
            self.chrom_sizes_file = "/home/lgunsalus/Documents/data/reference_genomes/hg38.chrom.sizes"
            self.reference_path = "/home/lgunsalus/Documents/data/reference_genomes/hg38.fa"
        else:
            self.chrom_sizes_file = "/home/lgunsalus/Documents/data/genome/reference/hg19.chrom.sizes"
            self.reference_path = "/home/lgunsalus/Documents/data/reference_genomes/hg19.ml.fa"
        
        # Create Lookup Table
    
    def preprocess_contact_dataset(self, input_file):
        """
        Preprocess contacts and write unique ranges to bed file
        """
        if self.verbose:
            print("Reading in single cell contact matrix...")
        # read in csv of unique reads
        scDF = pd.read_csv(input_file, dtype={'chr_a': str, 'chr_b': str})

        if self.verbose:
            print("Preprocessing Contacts...")
    
        # convert all chromosome names to strings
        scDF['chr_a'] = scDF['chr_a'].astype(str)
        scDF['chr_a'] = scDF['chr_b'].astype(str)

        # center contact location
        scDF['center_a'] = scDF['start_a'] + (scDF['end_a'] - scDF['start_a'])//2
        scDF['center_b'] = scDF['start_b'] + (scDF['end_b'] - scDF['start_b'])//2

        # filter out intra-chromosomal contacts
        scDF = scDF[scDF['chr_a']==scDF['chr_b']]

        # add cell identifier
        scDF["unique_id"] = scDF["chr_a"] + "-" + scDF["cell_id"]

        # unravel
        loc_a = scDF[['chr_a', 'center_a']]
        loc_b = scDF[['chr_b', 'center_b']]
        all_locs = pd.concat([loc_a,loc_b.rename(columns={'chr_b':'chr_a', 'center_b':'center_a'})], ignore_index=True)
        resolution = 1000

        # get bin number 
        all_locs = all_locs.assign(bin_no = all_locs['center_a']//resolution)
        all_locs = all_locs.assign(start = all_locs['bin_no'] * resolution)
        all_locs = all_locs.assign(end = all_locs['start'] + resolution)

        # filter out of range bins
        size_dict = read_chrom_sizes(self.chrom_sizes_file)
        all_locs = all_locs.assign(chrom_size = all_locs['chr_a'].map(size_dict))
        all_locs = all_locs.assign(too_large = all_locs['chrom_size'] < all_locs['end'])
        all_locs = all_locs[all_locs['too_large'] == False]

        # filter out chromosomes
        all_locs = all_locs[all_locs['chr_a'] != "Y"]
        all_locs = all_locs[all_locs['chr_a'] != "M"]

        # remove duplicate bins
        bin_calls = all_locs[['chr_a', 'start', 'end']]
        unique_bin_calls = bin_calls.drop_duplicates() 

        # sort values 
        unique_bin_calls = unique_bin_calls.sort_values(by = ["chr_a", "start"]) 
        unique_bin_calls = unique_bin_calls.rename(columns={"chr_a": "chr"})

        # add chr number for fasta
        unique_bin_calls['chr'] = 'chr' + unique_bin_calls['chr'].astype(str)

        # Write to bed file
        bed_out = self.process_dir + self.run_name + "_ranges.bed"
        if self.verbose:
            print(f'Writing ranges to {bed_out}...')
        if not os.path.exists(self.process_dir): # make output directory 
            os.makedirs(self.process_dir)
        unique_bin_calls.to_csv(bed_out, sep='\t', index=False, header=False) # write to bed file
        return
    
    def get_sequence_fragments(self):
        """
        Extract DNA sequence from genomic ranges bed file
        """
        if self.verbose:
            print("Extracting DNA sequence from genomic ranges bed file...")
            
        # Define output files
        bed_out = self.process_dir + self.run_name + "_ranges.bed"
        genomic_fragments = self.process_dir + self.run_name + "_fragments.fa.out"

        # Read in ranges, extract sequence, save
        genomic_ranges_bed = BedTool(bed_out)                              # read in genomic ranges bed file
        reference_fasta = pybedtools.example_filename(self.reference_path) # read in reference sequence
        ranges_sequence = genomic_ranges_bed.sequence(fi=reference_fasta)  # sequence under genomic ranges
        saved_seqs = ranges_sequence.save_seqs(genomic_fragments)          # save genomic range sequences as fasta file
        return
    
    def get_tf_coverage(self, tf_peaks_bed_file, frac=0.8):
        """
        Get TF coverage across genomic ranges (labels)
        """
        
        if self.verbose:
            print("Calling coverage with TF peaks...")
            
        # Define output files
        ranges_bed_file = self.process_dir + self.run_name + "_ranges.bed"
        coverage_bed_file = self.process_dir + self.run_name + "_" + self.cell_type +"_binary.bed"
        binary_coverage_file = self.process_dir + self.run_name + "_" + self.cell_type + "_binary.bed"

        # Read in ranges, call coverage, save
        ranges_bed = pybedtools.example_bedtool(ranges_bed_file)      # read in genomic ranges bed file
        tf_peaks_bed = pybedtools.example_bedtool(tf_peaks_bed_file)  # read in chip-seq peaks bed file
        coveraged_bed = ranges_bed.coverage(tf_peaks_bed, F=frac)     # determine coverage
        saved_coverage = coveraged_bed.saveas(binary_coverage_file)   # save to bed file
        return 
    
    def read_in_fragments(self):
        """
        Read in generated sequences and coverage, save to CSV
        """
        
        if self.verbose:
            print("Saving sequence fragments and labels")
            
        # Define input and output files
        genomic_fragments = self.process_dir + self.run_name + "_fragments.fa.out"
        binary_coverage_file = self.process_dir + self.run_name + "_" + self.cell_type + "_binary.bed"
        save_pickle = self.process_dir + self.run_name + "_" + self.cell_type + "_sequence_labels.pkl"
            
        # Read in sequence fragments
        with open(genomic_fragments) as fasta_file: 
            identifiers = []
            seqs = []
            for seq_record in SeqIO.parse(fasta_file, 'fasta'): 
                identifiers.append(seq_record.id)
                seqs.append(str(seq_record.seq))
        seqF = pd.DataFrame(list(zip(identifiers, seqs)), 
                   columns =['identifiers', 'seqs'])

        # Read in coverage
        ctcf_coverage = pd.read_csv(binary_coverage_file,sep='\t', header=None)
        assert(seqF.shape[0] == ctcf_coverage.shape[0])
        totalF = pd.concat([seqF, ctcf_coverage], axis=1)

        # Add labels
        totalF['label'] = totalF[3] > 0
        totalF['label'] = totalF['label'].astype(int)

        # Get GC Content
        totalF['gc_content'] = totalF['seqs'].apply(lambda x: get_gc_content(x))

        # Save to pickle
        save_table = totalF[['identifiers', 'seqs', 'label', 'gc_content']]
        save_dict = save_table.set_index('identifiers').to_dict('index')
        if (self.verbose):
            print(f'Writing lookup dictionary to {save_pickle}...')
        with open(save_pickle, 'wb') as handle:
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return 
        
###############################################################################
# Single Cell Graph Creation
###############################################################################''



class scGraphPreprocess(object):
    """
    Preprocess individual cells: Create edge_index, pull label and sequence, save to file
    """    
    def __init__(self,
                 contactF, 
                 lookup_dict, 
                 save_dir, 
                 save = True, 
                 resolution = 1000):
        """
        contactF: contact data frame from single cell high data set
        """            
        
        # Contact matrix
        self.contactF = contactF
        
        # Graph information
        self.graph_id = str(contactF.graph_id.values[0])
        self.chrom = contactF.chr_a.values[0]
        self.cell_id = contactF.unique_id.values[0]
        self.cell_type = self.contactF.celltype.values[0]
        self.resolution = resolution
        
        # Node information
        self.lookup_dict = lookup_dict
        self.filteredF, self.bin_to_node = self.call_sequence(self.contactF)
        
        # Create edge index
        self.edge_index = self.create_edge_index()
        
        # Metadata to save
        self.positive_ratio = self.get_positive_ratio()
        self.num_nodes = self.filteredF.label.values.shape[0]
        self.num_edges = self.edge_index.shape[1]
        
        # Save info
        self.save_dir = save_dir
        self.save_pickle_file = self.save_dir + self.graph_id + ".pkl"
        
        if save:
            self.save_pickle()
    
    def call_sequence(self, 
                      contactF):
        """
        Filter and unravel contact matrix, collect sequence and label
        """
        
        # unravel
        loc_a = contactF[['chr_a', 'bin_a']]
        loc_b = contactF[['chr_b', 'bin_b']]
        all_locs = pd.concat([loc_a,loc_b.rename(columns={'chr_b':'chr_a', 'bin_b':'bin_a'})], ignore_index=True)

        # remove duplicate bins
        all_locs = all_locs.drop_duplicates() 

        # get position
        all_locs['start'] = all_locs['bin_a'] * self.resolution
        all_locs['end'] = all_locs['start'] + self.resolution

        # add chromosome ID for dictionary key
        all_locs = all_locs.assign(lookup_key = "chr" + all_locs['chr_a'].astype(str) + ":" + 
                                                        all_locs['start'].astype(str) + "-" + 
                                                        all_locs['end'].astype(str))

        # call sequence
        def rowFunc(row):
            call = self.lookup_dict[row['lookup_key']]
            return [call['seqs'], call['label'], call['gc_content']]
        all_locs['call'] = all_locs.apply(rowFunc, axis=1)
        all_locs[['seq','label', 'gc_content']] = pd.DataFrame(all_locs.call.tolist(), index= all_locs.index)

        # filter out rows without sequence
        all_locs = all_locs.dropna(thresh=1)

        # create bin to node dictionary
        bin_to_node = dict(zip(all_locs.bin_a.values, np.arange(all_locs.shape[0])))

        # reset index
        all_locs = all_locs.reset_index()
            
        return all_locs, bin_to_node

    def create_edge_index(self):
        """
        Create edge_index from contacts
        """

        # ensure contacts are valid bins with matching sequence and labels
        chromDF = self.contactF[self.contactF['bin_a'].isin(self.filteredF.bin_a.values)]
        chromDF = chromDF[chromDF['bin_b'].isin(self.filteredF.bin_a.values)]

        # add edge_index
        node_a = chromDF.bin_a.map(self.bin_to_node)
        node_b = chromDF.bin_b.map(self.bin_to_node)
        edge_index = np.stack([node_a.values, node_b.values])
        return edge_index   
    
    def save_pickle(self):
        edge_index = self.edge_index
        x_features = self.filteredF.seq.values
        y_labels = self.filteredF.label.values
        position = self.filteredF.start.values

        graph_dict = {'edge_index': edge_index,
                      'x_features': x_features,
                      'y_labels': y_labels,
                      'position': position}

        with open(self.save_pickle_file, 'wb') as handle:
            pickle.dump(graph_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return
    
    def get_write_status(self):
        """
        Return metadata associated with single cell graph
        """
        return [self.graph_id, 
                self.chrom, 
                self.cell_id, 
                self.save_pickle_file, 
                self.cell_type,
                self.positive_ratio, 
                self.num_nodes, 
                self.num_edges]
        
    def get_positive_ratio(self):
        """
        Return positive ratio
        """
        labels = self.filteredF.label.values
        return (labels == 1).sum()/labels.shape[0]

class BuildGraphs(object):
    """
    Preprocess single cell HiC dataset to save graphs for training. 
    """    
    def __init__(self, resolution = 1000):   
        #def __init__(self, contactF, lookup_dict, save_dir, save = True, resolution = 1000):
        
        self.resolution = resolution
    
    def read_chrom_sizes(self, 
                         chrom_sizes_file):
        """
        create a dictionary mapping from chromosomes to chromosome sizes
        input: path to chromosome size file
        output: dictoinary mapping chromosome to chromosome size
        """ 
        chrom_sizes = pd.read_csv(chrom_sizes_file,header=None,sep='\t')
        chrom_sizes.columns = ['chr', 'sizes']
        chrom_sizes["chr"] = chrom_sizes['chr'].map(lambda x: x.strip('chr'))

        return dict(zip(chrom_sizes.chr.values, chrom_sizes.sizes.values))

    def preprocess_sc_dataframe(self, 
                                input_file, 
                                chrom_sizes_file):
        """
        read in and filter single cell Hi-C dataset
        input: pre-processed single cell graph dataset with contacts
        output: single cell contact dataframe
        """    

        print("Importing file...")
        # read in csv of unique reads
        scDF = pd.read_csv(input_file, dtype={'chr_a': str, 'chr_b': str})

        # convert all chromosome names to strings
        scDF['chr_a'] = scDF['chr_a'].astype(str)
        scDF['chr_a'] = scDF['chr_b'].astype(str)

        # center contact location
        scDF['center_a'] = scDF['start_a'] + (scDF['end_a'] - scDF['start_a'])//2
        scDF['center_b'] = scDF['start_b'] + (scDF['end_b'] - scDF['start_b'])//2
        
        # filter out intra-chromosomal contacts
        scDF_filtered = scDF[scDF['chr_a']==scDF['chr_b']]

        # filter out chromosomes
        scDF_filtered = scDF_filtered[scDF_filtered['chr_a'] != "Y"]
        scDF_filtered = scDF_filtered[scDF_filtered['chr_a'] != "M"]

        # remove out of range contacts
        size_dict = self.read_chrom_sizes(chrom_sizes_file)
        scDF_filtered = scDF_filtered.assign(chrom_size = scDF_filtered['chr_a'].map(size_dict))
        scDF_filtered = scDF_filtered.assign(too_large = scDF_filtered['chrom_size'] < scDF_filtered[['end_a', 'end_b']].values.max(1))
        scDF_filtered = scDF_filtered[scDF_filtered['too_large'] == False]

        # add cell identifier
        scDF_filtered["unique_id"] = scDF_filtered["chr_a"] + "-" + scDF_filtered["cell_id"]

        # define bins
        scDF_filtered = scDF_filtered.assign(bin_a = scDF_filtered.center_a.values//self.resolution,
                                             bin_b = scDF_filtered.center_b.values//self.resolution)

        # filter out within-bin contacts
        scDF_filtered = scDF_filtered[scDF_filtered['bin_a'] != scDF_filtered['bin_b']]

        scDF_filtered["unique_bin_a"] = scDF_filtered["bin_a"].astype(str) + "-" + scDF_filtered["unique_id"]
        scDF_filtered["unique_bin_b"] = scDF_filtered["bin_b"].astype(str) + "-" + scDF_filtered["unique_id"]
        scDF_filtered = scDF_filtered.set_index(['unique_bin_a', 'unique_bin_b'])

        return scDF_filtered
    
    def split_disconnected_components(self, df):
        """
        Label each disconnected compontent with a unique identifier
        """
        print("Splitting disconnected components...")
        graph  = nx.Graph()
        graph.add_edges_from(df.index)
        graph_dict = { node_id : graph_id for graph_id, graph in enumerate(nx.connected_components(graph)) for node_id in graph}
        df['graph_id'] = df.index.get_level_values('unique_bin_a').map(graph_dict)
        return df
    
    def preprocess_single_cells(self,
                                labeledDF, 
                                lookup_pickle, 
                                process_dir,
                                run_name, 
                                graph_start=None,
                                graph_end=None):

        # Load lookup dictionary
        with open(lookup_pickle, 'rb') as handle:
            lookup_dict = pickle.load(handle)

        # Group by unique graph ID
        unique_ids = labeledDF['graph_id'].unique()

        # Process subset of graphs
        if graph_start != None:
            unique_ids = unique_ids[graph_start:graph_end]

        scDF_grouped = labeledDF.groupby(['graph_id'])

        cell_write_statuses = [] 
        failed_ids = []  
        save_dir = process_dir + "cell_graphs/"
        if not os.path.exists(save_dir): # make output directory 
            os.makedirs(save_dir)

        print("Preprocessing cells...")
        for unique_id in tqdm_notebook(unique_ids):
            try:
                # Extract contacts per chromosome
                cellDF = scDF_grouped.get_group(unique_id)

                # Preprocess Graph
                cell_graph = scGraphPreprocess(cellDF, lookup_dict, save_dir)
                status = cell_graph.get_write_status() # save metadata
                cell_write_statuses.append(status)

            except:
                # In the case of a failed graph build, write id to list
                failed_ids.append(unique_id)

        print("Writing metadata...")
        # Write metadata to csv
        metadata_csv = process_dir + run_name + "_graph_metadata.csv"
        with open(metadata_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(cell_write_statuses)

        # write failed ids to text file
        failed_id_file = process_dir + run_name + "_failed_ids.txt"
        f=open(failed_id_file,'w')
        for ele in failed_ids:
            f.write(ele+'\n')
        f.close()

        return

###############################################################################
# CNN Dataset utils
###############################################################################
    
def get_normalized_distribution(data, bin_number=100):
    """Fit dataset to a normalized distribution."""
    bin_size = 1/bin_number
    density = stats.kde.gaussian_kde(data)
    x = np.arange(0., 1, bin_size)
    normalize_count = density(x)
    normalize_ratio = [float(i)/sum(normalize_count) for i in normalize_count]
    return x, normalize_ratio

def get_balanced_dataset(totalF, bin_number=100, verbose=True):
    """
    Balance positive dataset against negative dataset.
    Class balance and fit the negative to the positive example distribution.
    """
    # Subset into positive and negative classes
    positive_labels = totalF[totalF['label'] == 1]
    negative_labels = totalF[totalF['label'] == 0]

    # Get normalized distribution of positives
    data = positive_labels['gc_content']
    x, normalize_ratio = get_normalized_distribution(data, bin_number=100)
    no_positives = positive_labels.shape[0]
    negative_sample_list = []
    total_missing_negatives = 0
    bin_size = 1/bin_number
    
    # Sample from negative distribution to match positive
    for i in tqdm_notebook(range(bin_number)):
        cur_ratio = normalize_ratio[i]
        gc_min = x[i]
        gc_max = gc_min + bin_size
        sample_number = int(round(cur_ratio * no_positives))
        negative_subset = negative_labels[(negative_labels['gc_content'] >= gc_min) & (negative_labels['gc_content'] <= gc_max)]
        
        # Too few negative examples to sample
        if (negative_subset.shape[0] < sample_number):
            missing_negatives = sample_number - negative_subset.shape[0]
            negative_sample_list.append(negative_subset)
            total_missing_negatives += missing_negatives
        else: 
            negative_sample = negative_subset.sample(n=sample_number, random_state=1)
            negative_sample_list.append(negative_sample)

    gc_matched_negatives = pd.concat(negative_sample_list)
    balancedF = pd.concat([positive_labels, gc_matched_negatives])
    
    if verbose:
        print("Total missing ratio: " + str(total_missing_negatives/no_positives))
        print(str(total_missing_negatives) + " of " + str(no_positives) + " total negatives missing, " + "total missing ratio: " + str(total_missing_negatives/no_positives))
    return balancedF

def get_class_balance(totalF):
    positive_labels = totalF[totalF['label'] == 1]
    negative_labels = totalF[totalF['label'] == 0]
    no_positives = positive_labels.shape[0]
    no_negatives = negative_labels.shape[0]
    pos_ratio = no_positives /(no_positives + no_negatives)
    print(str(no_positives) + " positive examples, " + str(no_negatives) + " negative examples")
    print("Positive/Negative Ratio: " + str(pos_ratio))
    return

def plot_gc_distribution(exampleF):
    """Plot the GC content distribution between positives and negatives"""
    labels = exampleF.label.unique()
    plt.hist([exampleF.loc[exampleF.label == x, 'gc_content'] for x in labels], label=labels, bins = 50)

    plt.title("Positive and Negative GC Content Distribution")
    plt.show()
    return

def write_labels_to_csv(forward_and_reverse, ctcf_labels):
    """write to a csv."""
    forward_and_reverse[['label', 'seqs', 'chr']].to_csv(ctcf_labels, index=False)
    print("Wrote labels to " + ctcf_labels)
    return 

def pickle_to_csv(lookup_pickle, lookup_csv):
    """
    Save pickle to csv file
    """
    # Load lookup dictionary
    with open(lookup_pickle, 'rb') as handle:
        lookup_dict = pickle.load(handle)
    lookupF = pd.DataFrame.from_dict(lookup_dict)
    lookupF = lookupF.T
    lookupF.to_csv(lookup_csv)
    return 






