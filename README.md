# GWAS Weighted Kernel Bayesian Regression (GWKBR)
GWKBR is a novel genomic prediction method that integrates machine learning techniques (such as weighted Gaussian kernel regression and Bayesian optimization), Bayesian inference, genome-wide association study (GWAS) and cross-validation process.

## Tutorial and Examples
We implemented GWKBR in Python. Dependencies: python > 3.6.

We provided example code and toy datasets to illustrate how to use GWKBR for hyperparameter optimization and genomic prediction. Please check GWKBR.py to see how to run GWKBR on the toy example we provided in the example_data directory.

## Prepare files
The prepare files need to be placed in the example_data folder and include the following five files：

all_genotypes.bed(bim/fam): Binary genotype files for all individuals (training and test sets). In the .fam file, the last column records the phenotype, with test-set individuals coded as −9.

train_id.txt: Individual IDs of the training set. A single column where each row represents one individual, corresponding to the second column of the .fam file. 

test_id.txt: Individual IDs of the test set. A single column where each row represents one individual, corresponding to the second column of the .fam file.

## Running command
Before running the program, the users needs to install the required packages (os, numpy, skopt, scikit-learn, scipy, etc.), as well as the PLINK and GEMMA software. Then, place the software and the example_data folder in the same directory. Enter the current directory and run the program by typing the command python GWKBR.py. For example:

cd path/to/your/directory

python GWKBR.py

## output files
The output files will be stored in the current folder and include best_params.txt, y_test_pred.txt, gwas_snps and snp_weights.txt.

1. best_params.txt: The optimal hyperparameters determined by cross-validation and the Bayesian optimization algorithm (including kernel type, gamma, and lambda).

2. y_test_pred.txt: The predicted values for the test set individuals obtained by fitting the model using the optimal hyperparameters. The file contains two columns: individual IDs and their corresponding predicted values.

3. gwas_snps: SNPs with estimated effects identified from GWAS analysis based on the entire training set.

4. snp_weights.txt: SNP weights calculated based on the entire training set (corresponding to gwas_snps).

## Other Notes on the Software
The GWKBR.py script includes the complete process of kernel model selection (Kw or K model), hyperparameter optimization, and prediction. If the kernel model and optimal hyperparameters have already been determined and only prediction is required, one can directly run Prediction_weighted-K.py (Kw model) or Prediction_unweighted-K.py (K model), provided that the corresponding values of gamma and lambda are specified in the script.

## QUESTIONS AND FEEDBACK
For questions or concerns with GWKBR software, please contact xwangchnm@163.com.

We welcome and appreciate any feedback you may have with our software and/or instructions.
