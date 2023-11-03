# Code to process ORCA dataset.
# This code was taken directly from Rajpurkar et al.
# Original code can be found here: https://github.com/aparna-arr/DeepLearningChromatinStructure/blob/master/DataPreprocessing/prepare_data_one_gene_at_a_time.py

# Author: Aparna R. Rajpurkar, 2021

import numpy as np
from scipy import interpolate

def read_data(xyzfile, rnafile):
    '''Function to read in raw experimental data, clean up NANs, and output useful data matricies'''
    xyzfp = open(xyzfile, "r")
    header = list()
    datDict = dict()

    # loop over the file
    for line in xyzfp:
        if not line[0].isdigit():
            header = line.rstrip().split(',')
            continue        
        else:
            # grab the useful information
            # "cell" being example number
            #barcode, x, y, z, cell = line.rstrip().split(',')    
            cell, embNumber, segNumber, x, y, z, brightness, barcode = line.rstrip().split(',')
            barcode = int(barcode)
            cell = int(cell)

            # convert NaN strings to numpy nans
            if x == 'NaN':
                x = np.nan
            else:
                x = float(x)
            
            if y == 'NaN':
                y = np.nan
            else:
                y = float(y)
        
            if z == 'NaN':
                z = np.nan
            else:
                z = float(z)

            # add xyz coordinates by barcodes to data dictionary by cell/example number
            if cell not in datDict:            
                datDict[cell] = dict()
                datDict[cell] = dict()
                datDict[cell]['dat'] = [[x,y,z]]
                datDict[cell]['info'] = [barcode]
            else:
                datDict[cell]['dat'].append([x,y,z])
                datDict[cell]['info'].append(barcode)

    xyzfp.close()    

    # read in the RNA data, or Y matrix
    rnafp = open(rnafile, "r")

    rnaheader = list()
    rnaDict = dict()

    for line in rnafp:
        if not line[0].isdigit():
            rnaheader = line.rstrip().split(',')
            continue        
        else:
            # get the 3 RNA values per example
            #rna1, rna2, rna3, cell = line.rstrip().split(',')
            #cell, embNumber, segNumber, AbdB_Main_Exon, AbdB_Short_Exon, AbdA_Exon, Ubx_Exon, Antp_Exon, Scr_Exon, Dfd_Exon, pb_Exon, lab_Exon, Iab4_ncGene, bxd_ncGene,inv_Exon, en_Whole_Gene, ftz_Whole_Gene, zen_Whole_Gene, Ama_Whole_Gene, sna_Whole_Gene, elav_Exon, rna1, AbdB_Short_Intron, rna2, rna3, Antp_Intron, Scr_intron, Dfd_Intron, pb_Intron, lab_Intron, inv_Intron, elav_Intron = line.rstrip().split(',')
            cell, embNumber, segNumber, AbdB_Main_Exon, AbdB_Short_Exon, AbdA_Exon, Ubx_Exon, Antp_Exon, Scr_Exon, Dfd_Exon, pb_Exon, lab_Exon, Iab4_ncGene, bxd_ncGene,inv_Exon, en_Whole_Gene, ftz_Whole_Gene, zen_Whole_Gene, Ama_Whole_Gene, sna_Whole_Gene, elav_Exon, AbdB_Main_Intron, AbdB_Short_Intron, AbdA_Intron, Ubx_Intron, Antp_Intron, Scr_intron, Dfd_Intron, pb_Intron, lab_Intron, inv_Intron, elav_Intron = line.rstrip().split(',')
            
            rna_list = [AbdB_Main_Exon, AbdB_Short_Exon, AbdA_Exon, Ubx_Exon, Antp_Exon, Scr_Exon, Dfd_Exon, pb_Exon, lab_Exon, Iab4_ncGene, bxd_ncGene,inv_Exon, en_Whole_Gene, ftz_Whole_Gene, zen_Whole_Gene, Ama_Whole_Gene, sna_Whole_Gene, elav_Exon, AbdB_Main_Intron, AbdB_Short_Intron, AbdA_Intron, Ubx_Intron, Antp_Intron, Scr_intron, Dfd_Intron, pb_Intron, lab_Intron, inv_Intron, elav_Intron]
            rna_list = [int(rna) for rna in rna_list]
            #rna1 = int(rna1)
            #rna2 = int(rna2)
            #rna3 = int(rna3)
            cell = int(cell)

            rnaDict[cell] = dict()
            
            # add to the data dictionary
            #rnaDict[cell]['dat'] = [rna1, rna2, rna3]
            rnaDict[cell]['dat'] = rna_list
            rnaDict[cell]['info'] = [cell, embNumber, segNumber]

    rnafp.close()

    return datDict, rnaDict

def fill_in_missing(datDict):
    num_barcodes = 52

    for cell in datDict:
        partial_map = datDict[cell]['dat']
        barcodes = datDict[cell]['info']

        full_map = np.empty((num_barcodes, 3))
        full_map[:] = np.NaN

        for i,barcode_i in enumerate(barcodes):
            full_map[barcode_i-1] = partial_map[i]

        datDict[cell]['dat'] = full_map
    return datDict

def interpolate_coords(coords):
    '''function to interpolate NaN values as a minimal imputation strategy'''
    newcoords = coords
    nanidx = np.argwhere(np.isnan(coords))
    goodidx = np.argwhere(~np.isnan(coords))

    # identify "good" and "nan" values and indexes
    x = np.arange(len(coords))
    goodx = x[goodidx]
    nanx = x[nanidx]
    goody = coords[goodidx]
    goodx = goodx.squeeze()
    goody = goody.squeeze()

    # perform the interpolation
    f = interpolate.interp1d(goodx, goody, fill_value="extrapolate")

    ynew = f(nanx)
    
    # replace nan values with interpolated coordinates
    newcoords[nanx] = ynew

    return newcoords

def normalize(xyzdat):
    '''function to center XYZ coordinates on the center of mass'''
    normdat = xyzdat - np.mean(xyzdat, axis=0)
    return normdat

def filter_dat(xyzdat, rnadat, req_xyz_perc = 0.5, interpol=True, norm=True, bool_thresh=-1):
    '''given the high rate of NaNs in the data, need to filter out cells with less real data than a useful theshold'''
    filtXYZDat = list()
    filtRNADat = list()
    
    # iterate over cells
    for c in xyzdat:
        currDat = xyzdat[c]['dat']
        currDat = np.array(currDat)
        sums = np.sum(currDat, axis=1)
        goodRows = np.sum(~np.isnan(sums))

        # keep only those cells/examples where we have more than the required real data percentage
        if goodRows / currDat.shape[0] >= req_xyz_perc:
            if interpol == True:
                # interpolate the missing values
                currDat[:,0] = interpolate_coords(currDat[:,0])
                currDat[:,1] = interpolate_coords(currDat[:,1])
                currDat[:,2] = interpolate_coords(currDat[:,2])

                if norm == True:
                    # center the coordinates to the center of mass                 
                    filtXYZDat.append(normalize(currDat))   
                else:
                    filtXYZDat.append(currDat)                        
            else:
                if norm == True:
                    filtXYZDat.append(normalize(currDat))
                else:
                    filtXYZDat.append(currDat)        

            # make the output value binary
            currRNA = rnadat[c]['dat']
            currRNA = np.array(currRNA)

            if bool_thresh == -1:
                filtRNADat.append(currRNA)
            else:
                boolRNA = np.where(currRNA >= bool_thresh, 1, 0)
                filtRNADat.append(boolRNA)
    
    filtXYZDat = np.array(filtXYZDat)
    filtRNADat = np.array(filtRNADat)
    return filtXYZDat, filtRNADat

    