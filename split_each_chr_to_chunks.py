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



# def split_X_to_chunks(cov_file, end_file_name, output_file):
#     cov_df = pd.read_pickle(cov_file)
#     from_ = 0
#     to_ = 10
#     number_sample_chunks = 0
#     with open(output_file, 'wb') as file_handle:
#         while True:  # while there are still samples
#             df = pd.DataFrame()
#             for chr in range(1, 23):
#                 print(chr)
#                 file_name = '/home/hochyard/UKBB/results/data_for_model/chr' + str(chr) + end_file_name
#                 (bim, fam, bed) = read_plink(file_name, verbose=False)  # bed is dask.array.core.Array
#                 snp_columns = bim['snp'].to_numpy()
#                 snp_columns_df = pd.DataFrame(snp_columns)
#                 snp_columns_df.to_csv('snps_name')
#                 G = read_plink1_bin(file_name, verbose=False)  # xarray.core.dataarray.DataArray
#                 from_snps_ = 0
#                 to_snps_ = 1000
#                 while True:
#                     G_to_pandas = G[:, from_snps_:to_snps_].to_pandas()  # pandas.core.frame.DataFrame
#                     if G_to_pandas.empty == False:  # if remain snps
#                         G_to_pandas.set_axis(snp_columns[from_snps_:to_snps_], axis=1, inplace=True)
#                         print(G_to_pandas)
#                         print(G_to_pandas.isna().sum())
#                         G_to_pandas.dropna(axis=1, inplace=True)  # remove snps with na values ???????????????????????????????
#                         G_to_pandas = G_to_pandas.astype(np.int8)
#                         # G_to_pandas = G_to_pandas.astype(np.float16)
#                         G_to_pandas.reset_index(level=0, inplace=True)
#                         G_to_pandas.rename(columns={'sample': 'FID'}, inplace=True)
#                         G_to_pandas = G_to_pandas.sort_values('FID').reset_index(drop=True)
#                         if G_to_pandas.iloc[from_:to_, :].empty == False:  # their are no samples any more
#                             if chr == 1 and from_snps_ == 0:
#                                 df = pd.merge(cov_df.iloc[from_:to_, :], G_to_pandas.iloc[from_:to_, :], on='FID')
#                             else:
#                                 df = pd.merge(df, G_to_pandas.iloc[from_:to_, :], on='FID')
#                             print(df)
#                         else:
#                             return number_sample_chunks
#                         from_snps_ = to_
#                         to_snps_ = to_ + 1000
#                     else:
#                         break
#             number_sample_chunks = number_sample_chunks + 1
#             print(df)
#             pickle.dump(df, file_handle, protocol=4)
#             from_ = to_
#             to_ = to_ + 10


# def split_X_to_chunks(cov_file, end_file_name, output_file):
#     cov_df = pd.read_pickle(cov_file)
#     df = pd.DataFrame()
#     from_ = 0
#     to_ = 1000
#     number_chunks = 0
#     with open(output_file, 'wb') as file_handle:
#         while True:
#             for chr in range(1, 23):
#                 content_open = "/home/hochyard/UKBB/results/data_for_model/chr" + str(chr) + end_file_name
#                 chr_df = pd.read_pickle(content_open)
#                 if chr_df.iloc[from_:to_, :].empty == False:
#                     if chr == 1:
#                         df = pd.merge(cov_df.iloc[from_:to_, :], chr_df.iloc[from_:to_, :], on='FID')
#                     else:
#                         df = pd.merge(df,chr_df.iloc[from_:to_, :], on='FID')
#                 else:
#                     return number_chunks
#             number_chunks = number_chunks + 1
#             pickle.dump(df, file_handle, protocol=4)
#             from_ = to_
#             to_ = to_ + 1000

# def split_X_to_chunks(cov_file,end_file_name):
#     cov_df = pd.read_pickle(cov_file)
#     df = {}
#     for chr in range(1, 23):
#         content_open = "/home/hochyard/UKBB/results/data_for_model/chr" + str(chr) + end_file_name
#         chr_df = pd.read_pickle(content_open)
#         from_ = 0
#         to_ = 1000
#         number_chunks = 0
#         while True:
#             if chr_df.iloc[from_:to_, :].empty == False:
#                 number_chunks = number_chunks + 1
#                 if chr == 1:
#                     df[number_chunks] = pd.merge(cov_df.iloc[from_:to_, :],
#                                                              chr_df.iloc[from_:to_, :], on='FID')
#                 else:
#                     df[number_chunks] = pd.merge(df[number_chunks],
#                                                              chr_df.iloc[from_:to_, :], on='FID')
#                 from_ = to_
#                 to_ = to_ + 1000
#             else:
#                 break
#         return number_chunks, df

# def save_x_chunks_as_pickle(output_file, df):
# with open(output_file, 'wb') as file_handle:
# pickle.dump(df, file_handle, protocol=4)

# def save_x_chunks_as_pickle(output_file, number_chunks , df):
#     with open(output_file, 'wb') as file_handle:
#         for chunk in range(1,number_chunks+1):
#             pickle.dump(df[chunk], file_handle, protocol=4)

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

    






