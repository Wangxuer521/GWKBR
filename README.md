# GWAS Weighted Kernel Bayesian Regression (GWKBR)
GWKBR is a novel genomic prediction method that integrates machine learning techniques (such as weighted Gaussian kernel regression and Bayesian optimization), Bayesian inference, genome-wide association study (GWAS) and cross-validation process.

## Tutorial and Examples
We implemented GWKBR in Python. Dependencies: python > 3.6.

We provided example code and toy datasets to illustrate how to use GWKBR for hyperparameter optimization and genomic prediction. Please check GWKBR.py to see how to run GWKBR on the toy example we provided in the example_data directory.

## Prepare files
The prepare files include the following five files：

all_genotypes.bed(bim/fam): Binary genotype files for all individuals (training and test sets). In the .fam file, the last column records the phenotype, with test-set individuals coded as −9.

train_id.txt: Individual IDs of the training set. A single column where each row represents one individual, corresponding to the second column of the .fam file. 

test_id.txt: Individual IDs of the test set. A single column where each row represents one individual, corresponding to the second column of the .fam file.

## Running command
Before running the program, the users needs to install the required packages (os, numpy, skopt, scikit-learn, scipy, etc.), as well as the PLINK and GEMMA software. Then, navigate to the directory containing the GWKBR.py script and execute the program with different commands depending on the specific task. 

### Hyperparameter optimization, model selection, model training, and prediction
Example 1:
```
GWKBR_CV_NJOBS=5 python GWKBR.py --train_id ./example_data/train_id.txt \
                                 --test_id ./example_data/test_id.txt \
                                 --geno ./example_data/all_genotypes
```
GWKBR_CV_NJOBS: Number of threads (1-5), with a default value of 3;

--train_id: Path to the training set individual ID file;

--test_id: Path to the test set individual ID file;

--geno: Path to the genotype file (prefix).

By default, the GWKBR program performs the full procedure of hyperparameter optimization, model selection, model training, and prediction. In addition, GWKBR can also directly fit the model and perform prediction using predefined hyperparameters, but the additional command-line arguments --predict, --model, --gamma, and --lambda need to be specified, as illustrated below.

### Model fitting and prediction based on Kw (or K) model with known hyperparemeters
Example 2:
```
python GWKBR.py --train_id ./example_data/train_id.txt \
                --test_id ./example_data/test_id.txt \
                --geno ./example_data/all_genotypes \
                --predict \
                --model Kw (or K) \
                --gamma 3.19e-6 \
                --lambda 0.01 
```
The --train_id, --test_id, and --geno parameters are described above;

--predict: indicates that the model will perform prediction based on the specified hyperparameters;

--model: Model type (K or Kw);

--gamma: Bandwidth parameter of the Gaussian kernel (floating-poing value, e.g., 3.19e-6)

--lambda: Regularization parameter (floating-point value, e.g., 0.01)

## output files
The output files will be stored in the results folder and include best_params.txt, variance_component.log.txt, y_test_pred.txt, gwas_snps and snp_weights.txt.

1. variance_component.log.txt: Variance component file estimated using the REML AI algorithm.

2. y_test_pred.txt: The predicted values for the test set individuals obtained by fitting the model using the optimal hyperparameters. The file contains two columns: individual IDs and their corresponding predicted values.

3. best_params.txt: The optimal hyperparameters (including kernel type, gamma, and lambda) determined by cross-validation and the Bayesian optimization algorithm (generated when executing the full GWKBR pipeline, as in Example 1 above) .

4. gwas_snps: SNPs with estimated effects identified from GWAS analysis based on the entire training set.

5. snp_weights.txt: SNP weights calculated based on the entire training set (corresponding to gwas_snps).

## QUESTIONS AND FEEDBACK
For questions or concerns with GWKBR software, please contact xwangchnm@163.com.

We welcome and appreciate any feedback you may have with our software and/or instructions.
