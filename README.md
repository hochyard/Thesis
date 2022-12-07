# Thesis

I used a two stages approach. The first stage is a trait-agnostic (phenotype-agnostic), an unsupervised approach to dimensionality reduction. The second stage is training a prediction model using a supervised machine learning algorithm. Moreover, the first stage, which is computational resource intense, is independent of phenotype, i.e., its output can be used as an input for a prediction model for any chosen trait or disease, and therefore can be trained only once. 

I evaluated the approach using two dimensionality reduction models, Deep Autoencoder and Principal Component Analysis, and two phenotype prediction models, Deep Neural Network and Extreme Gradient Boosting. The models were trained using the UK Biobank dataset with over 340K subjects and 460K features (genes). Moreover, I evaluated the approach on two phenotypes, height (continuous), and hypertension (binary).

In this github folder, there are the central code files in my thesis. Files order:
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
9. Train perdiction model:
  9.1. For height phenotype: Height_prediction.py
  9.2. For hypertension phenotype: hypertension_prediction.py

Notes:
1. The input data sets are after preprocessing using PLINK2 software.
2. The trained dimensionality reduction models are now available in: https://github.com/nadavlab/genotyping_dimensionality_reduction 
