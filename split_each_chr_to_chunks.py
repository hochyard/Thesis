import pandas as pd
import pickle
from pandas_plink import read_plink, read_plink1_bin
import numpy as np
import pickle
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler

def split_X_to_chunks(cov_file, gene_end_file_name, gene_output_end_file_name, chunk_size, chr, gene_scaler):
    #cov_df = pd.read_pickle(cov_file)
    from_ = 0
    to_ = chunk_size
    gene_output_file = '/home/hochyard/UKBB/results/data_for_model/chunks_for_each_chr/chr_' + str(chr) + gene_output_end_file_name
    with open(gene_output_file, 'wb') as gene_file_handle:
        file_name = '/home/hochyard/UKBB/results/data_for_model/chr' + str(chr) + gene_end_file_name
        G = read_plink1_bin(file_name, verbose=False)  # xarray.core.dataarray.DataArray
        while True:  # while there are still samples
            G_to_pandas = G[from_:to_, :].to_pandas()  # pandas.core.frame.DataFrame
            if G_to_pandas.empty == False:  # their samples
                (bim, fam, bed) = read_plink(file_name, verbose=False)  # bed is dask.array.core.Array
                snp_columns = bim['snp'].to_numpy()
                G_to_pandas.set_axis(snp_columns, axis=1, inplace=True)
                G_to_pandas = G_to_pandas.astype(np.int8)
                G_to_pandas.reset_index(level=0, inplace=True)
                G_to_pandas.rename(columns={'sample': 'FID'}, inplace=True)
                #df = pd.merge(cov_df, G_to_pandas, how='inner', on='FID')
                not_to_scale = G_to_pandas['FID']
                scaled_values = gene_scaler.transform(G_to_pandas.drop('FID', axis = 1))
                G_to_pandas = pd.DataFrame(scaled_values, index = G_to_pandas.index, columns = G_to_pandas.drop('FID', axis = 1).columns)
                G_to_pandas['FID'] = not_to_scale
                pickle.dump(G_to_pandas, gene_file_handle, protocol=4)
                #pickle.dump(df, gene_file_handle, protocol=4)
                from_ = to_
                to_ = to_ + chunk_size
            else:
                return

def split_y_to_chunks(output_file, input_file, x_chunk_file, chunk_size):
     y = pd.read_pickle(input_file)
     with open(output_file, 'wb') as y_file_handle:
         with open(x_chunk_file, 'rb') as x_file_handle:
             from_ = 0
             to_ = chunk_size
             while True:
                 try:
                     X_batch = pickle.load(x_file_handle)
                     y_batch = y[y['FID'].isin(X_batch['FID'])]
                     pickle.dump(y_batch, y_file_handle, protocol=4) 
                     from_ = to_
                     to_ = to_ + chunk_size
                 except EOFError:
                        break

def StandardScaler_incremental(end_file_name_to_standard, chr, chunk_size):
    from_ = 0
    to_ = chunk_size
    file_name = '/home/hochyard/UKBB/results/data_for_model/chr' + str(chr) + end_file_name_to_standard
    G = read_plink1_bin(file_name, verbose=False)  # xarray.core.dataarray.DataArray
    chr_scaler = StandardScaler()
    while True:  # while there are still samples
        G_to_pandas = G[from_:to_, :].to_pandas()  # pandas.core.frame.DataFrame
        if G_to_pandas.empty == False:  # their samples
            chr_scaler.partial_fit(G_to_pandas)
            from_ = to_
            to_ = to_ + chunk_size
        else:
            print(chr_scaler.mean_)
            print(chr_scaler.var_)
            return chr_scaler
                

# --------------------train-------------------------------
#check()
# handle X + y

for chr in range(19,20):
    scaler = StandardScaler_incremental("_height_X_train_no_cov_no_missing.bed", chr, chunk_size=1000)
    #train
    split_X_to_chunks(cov_file="/home/hochyard/UKBB/results/data_for_model/height_cov_matrix_train.pkl",
                     gene_end_file_name="_height_X_train_no_cov_no_missing.bed",
                     gene_output_end_file_name="_height_X_train_1k_chunks_no_missing_scaled.pkl",
                     chunk_size=1000,
                     chr = chr,
                     gene_scaler = scaler)
    #test
    split_X_to_chunks(cov_file="/home/hochyard/UKBB/results/data_for_model/height_cov_matrix_test.pkl",
                     gene_end_file_name="_height_X_test_no_cov_no_missing.bed",
                     gene_output_end_file_name="_height_X_test_1k_chunks_no_missing_scaled.pkl",
                     chunk_size=1000,
                     chr = chr,
                     gene_scaler = scaler)

#split_y_to_chunks(output_file = "/home/hochyard/UKBB/results/data_for_model/chunks_for_each_chr/height_y_train_1k_chunks_no_missing.pkl",
#                  input_file = "/home/hochyard/UKBB/results/data_for_model/height_y_train.pkl",
#                  x_chunk_file = "chr_1_height_X_train_1k_chunks_with_cov_no_missing.pkl",
#                  chunk_size = 1000)

    






