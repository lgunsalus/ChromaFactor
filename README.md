# ChromaFactor: deconvolution of single-molecule chromatin organization with non-negative matrix factorization


<img width="698" alt="ChromaFactor_fig" src="https://github.com/lgunsalus/ChromaFactor/assets/11052222/6cca0b8c-d111-4397-8168-031b9732dd86">

## Introduction
The investigation of chromatin organization in single cells is pivotal for understanding the causal relationships between genome structure and function. However, the inherent heterogeneity in single-molecule data poses significant analytical challenges. ChromaFactor is our novel computational approach to deconvolve single-molecule chromatin organization datasets into distinguishable primary components and identify key cell subpopulations driving transcriptional signal. Our method's efficacy is demonstrated through its application to single-molecule imaging datasets across various genomic scales, where it has identified strong correlations between templates and key functional phenotypes, including active transcription, enhancer-promoter proximity, and genomic compartmentalization.

## Tutorial Overview

**Figure 1- Visualize molecules and apply ChromaFactor.ipynb**  
Walkthrough for visualizing single molecule chromatin examples and implementing ChromaFactor analysis.

**Figure 2- Single molecule examples.ipynb**  
Examples of single-molecule analyses showcasing individual chromatin fiber contributions.

**Figure 3- Random Forest and Transcription.ipynb**  
Predicting transcription with a Random Forest from ChromaFactor templates. Investigating enhancer-promoter distances in relation to templates.

**Figure 4- ChromaFactor on Su et al..ipynb**  
Application of ChromaFactor to the dataset from Su et al., demonstrating its utility across larger genomic contexts.

## Code Overview

**ChromaFactor.py**  
The core module containing the implementation of the ChromaFactor approach.

**DeepLearningChromatinStructure_code.py**  
Supporting code for ORCA dataset preprocessing as described in [Nature Communications](doi.org/10.1038/s41467-021-23831-4).

**file_utils.py**  
Utility functions for file handling within the ChromaFactor analysis pipeline.

**fish_preprocessing.py**  
Preprocessing routines for FISH (Fluorescence In Situ Hybridization) data prior to ChromaFactor analysis.

**genomic_annotation.py**  
Scripts for genomic annotation, as detailed in the preprint available at [bioRxiv](https://doi.org/10.1101/2023.04.04.535480).

**plot.py**  
Functions dedicated to plotting and visualizing data within the ChromaFactor framework.

**preprocessing_utils.py**  
General preprocessing utilities.

**utils.py**  
Other assorted functions.

