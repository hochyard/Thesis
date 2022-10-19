import pandas as pd
import pickle
from pandas_plink import read_plink, read_plink1_bin
import numpy as np
import pickle
import os

def split_X_to_chunks(gene_end_file_name, gene_output_end_file_name, chunk_size):
    from_ = 0
    to_ = chunk_size
    gene_output_file = '/home/hochyard/UKBB/results/data_for_model/chunks_for_each_chr/chr_' + str(chr) + gene_output_end_file_name
    with open(gene_output_file, 'wb') as gene_file_handle:
        file_name = '/home/hochyard/UKBB/results/data_for_model/chr' + str(chr) + gene_end_file_name
        G = read_plink1_bin(file_name, verbose=False)  # xarray.core.dataarray.DataArray
        while True:  # while there are still samples
            G_to_pandas = G[from_:to_, :].to_pandas()  # pandas.core.frame.DataFrame
            if G_to_pandas.empty == False:  # there's more samples
                (bim, fam, bed) = read_plink(file_name, verbose=False)  # bed is dask.array.core.Array
                snp_columns = bim['snp'].to_numpy()
                G_to_pandas.set_axis(snp_columns, axis=1, inplace=True)
                G_to_pandas = G_to_pandas.astype(np.int8)
                G_to_pandas.reset_index(level=0, inplace=True)
                G_to_pandas.rename(columns={'sample': 'FID'}, inplace=True)
                pickle.dump(G_to_pandas, gene_file_handle, protocol=4)
                from_ = to_
                to_ = to_ + chunk_size
            else:
                return


chunk_size=1000
for chr in range(os.environ['from'],os.environ['to']):
    #train
    split_X_to_chunks(gene_end_file_name="_X_train_no_cov_no_missing.bed",
                     gene_output_end_file_name="_X_train_1k_chunks_no_missing.pkl",
                     chunk_size=chunk_size)
    #test
    split_X_to_chunks(gene_end_file_name="_X_test_no_cov_no_missing.bed",
                     gene_output_end_file_name="_X_test_1k_chunks_no_missing.pkl",
                     chunk_size=chunk_size)
