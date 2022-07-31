# Thesis

Files order:
1. Split samples ID to train/test: split_train_test_IDs_for_autoencoder.py 
2. Due to data size, split the data into samples chuncks: Split_each_chr_to_chunks.py
3. Train dimensionality reduction:
  3.1. For PCA: PCA_dimension_reduction.py
  3.2. For Autoencoder: Autoencoder_for_each_chr.py
4. Preprocessing on the covariate matrix: covariate_arrangement.py
5. Predict the variables after dimensionality reduction (just for autoencoder): snp_prediction_dimension_reduction.py
6. Scale the variables after dimensionality reduction: scale_genes.py
7. Join variables from all chromosomes after dimensionality reduction + covarivate matrix: join_all_chr_after_dr.py
8. Match the dataframe (the result of section 7) to specific phenotype: match_pheno_IDs.py
9. Train DNN:
  9.1. For continuous phenotype: NN.py
  9.2. For binary phenotype: NN_for_binary_pheno.py

Notes:
1. The input data sets are after preprocessing using PLINK2 software.
