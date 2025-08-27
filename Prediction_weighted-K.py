# -*- coding: utf-8 -*-
import glob
import datetime
import os, shutil
import subprocess
import numpy as np
import pandas as pd
from numba import njit
from scipy.linalg import eigh
from scipy.sparse import diags
from skopt.space import Real
from scipy.stats import pearsonr
from skopt import BayesSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, RegressorMixin

@njit
def numba_invgamma(alpha, beta):
    return 1.0 / np.random.gamma(alpha, 1.0 / beta)

@njit
def gibbs_sampling_core(n_iter, burn_in, alpha, beta, y, K, G, initial_sigma_e2, initial_sigma_a2, lambda_, thinning_interval):
    n = len(y)
    sigma_e2_samples = np.zeros(n_iter, dtype=np.float64)
    sigma_a2_samples = np.zeros(n_iter, dtype=np.float64)
    sigma_e2_samples[0] = initial_sigma_e2
    sigma_a2_samples[0] = initial_sigma_a2

    K_T = K.T
    K_sq = np.dot(K_T, K)
    Ky = np.dot(K_T, y)

    for i in range(1, n_iter):
        # Update sigma_a2
        A = (1.0 / sigma_e2_samples[i-1]) * K_sq + lambda_ * (1.0 / sigma_a2_samples[i-1]) * G
        rhs = (1.0 / sigma_e2_samples[i-1]) * Ky
        mu_post = np.linalg.solve(A, rhs)
        y_pred = np.dot(K, mu_post)
        residual = y - y_pred
        alpha_post = alpha + n * 0.5
        beta_post = beta + 0.5 * np.dot(residual, residual)
        sigma_a2_samples[i] = numba_invgamma(alpha_post, beta_post)

        # Update sigma_e2
        A = (1.0 / sigma_e2_samples[i-1]) * K_sq + lambda_ * (1.0 / sigma_a2_samples[i]) * G
        rhs = (1.0 / sigma_e2_samples[i-1]) * Ky
        mu_post = np.linalg.solve(A, rhs)
        y_pred = np.dot(K, mu_post)
        residual = y - y_pred
        beta_post = beta + 0.5 * np.dot(residual, residual)
        sigma_e2_samples[i] = numba_invgamma(alpha_post, beta_post)

        if i % 10000 == 0:
            print(f"Iteration {i}/{n_iter} completed")

    thinned_sigma_a2_samples = sigma_a2_samples[burn_in::thinning_interval]
    thinned_sigma_e2_samples = sigma_e2_samples[burn_in::thinning_interval]

    return thinned_sigma_a2_samples, thinned_sigma_e2_samples, sigma_a2_samples, sigma_e2_samples

# Define the weighted GWKBR model (GWKBR)
class BayesianCombinedKernelRidge(BaseEstimator, RegressorMixin):
    def __init__(self, gamma=0.00006, lambda_=1.0, sigma_a2=2000, sigma_e2=700, G=None, weights=None):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.sigma_a2 = sigma_a2
        self.sigma_e2 = sigma_e2
        self.G = G
        self.weights = weights

    def _weighted_gaussian_kernel(self, X1, X2, gamma, weights):
        # Construct the weight matrix as a sparse diagonal matrix
        W = diags(weights, format='csr')
        
        # Calculate the weighted squared Euclidean distance using sparse matrix multiplication
        pairwise_sq_dists = np.sum((X1 @ W) * X1, axis=1)[:, np.newaxis] + \
                            np.sum((X2 @ W) * X2, axis=1) - \
                            2 * np.dot(X1 @ W, X2.T)
        
        # Calculate Gaussian kernel matrix
        return np.exp(-gamma * pairwise_sq_dists)

    def _generate_G_matrix(self, X):
        dosageA = X
        MA = dosageA - 1
        p1A = np.round((np.sum(dosageA, axis=0)) / (dosageA.shape[0] * 2), 3)
        pA = 2 * (p1A - 0.5)
        PA = np.tile(pA, (dosageA.shape[0], 1))
        ZA = MA - PA
        nani_A = dosageA.shape[0]
        Sum_2pq_A = 2 * np.sum(p1A * (1 - p1A))
        G = np.dot(ZA, ZA.T) / Sum_2pq_A + np.eye(ZA.shape[0]) * 1e-6

        # Make the matrix positive definite
        eigvals, eigvecs = np.linalg.eigh(G)
        rtol = 1e-6 * eigvals[0]
        eigvals[eigvals < rtol] = rtol
        G = np.dot(eigvecs, np.dot(np.diag(eigvals), eigvecs.T))

        return G

    def fit(self, X, y):
        if self.G is None:
            self.G = self._generate_G_matrix(X)
        self.n_features = X.shape[1]
        K = self._weighted_gaussian_kernel(X, X, self.gamma, self.weights)
        self.K = K
        self.X_fit_ = X
        self.y = y

        if self.G is not None and self.G.shape != K.shape:
            raise ValueError("G matrix shape does not match K matrix shape.")

        self._compute_posterior()
        
        return self

    def _compute_posterior(self):      
        # Solve ((1/se2)K^T K + lambda*(1/sa2)G) * mu = (1/se2)K^T y
        KtK = self.K.T @ self.K
        A = (1.0 / self.sigma_e2) * KtK + self.lambda_ * (1.0 / self.sigma_a2) * self.G
        rhs = (1.0 / self.sigma_e2) * (self.K.T @ self.y)
        self.mu_post = np.linalg.solve(A, rhs)
        self.Sigma_post = None

    def predict(self, X):
        K_test = self._weighted_gaussian_kernel(
            X, 
            self.X_fit_[:, :self.n_features], 
            self.gamma, 
            self.weights
        )
        return np.dot(K_test, self.mu_post)
        
# Define Gibbs sampling
def gibbs_sampling(n_iter, burn_in, alpha, beta, model_instance, thinning_interval=10):
    y = model_instance.y
    K = model_instance.K
    G = model_instance.G
    initial_sigma_e2 = 1
    initial_sigma_a2 = 1
    lambda_ = model_instance.lambda_

    (thinned_sigma_a2_samples, thinned_sigma_e2_samples, 
     sigma_a2_samples, sigma_e2_samples) = gibbs_sampling_core(
        n_iter, burn_in, alpha, beta, y, K, G, 
        initial_sigma_e2, initial_sigma_a2, lambda_, thinning_interval
    )

    print("Number of Gibbs sampling thinned_sigma_a2_samples:", len(thinned_sigma_a2_samples))
    print("Number of Gibbs sampling thinned_sigma_e2_samples:", len(thinned_sigma_e2_samples))

    return thinned_sigma_a2_samples, thinned_sigma_e2_samples, sigma_a2_samples, sigma_e2_samples

# Calculate the SNP weights
def calculate_snp_weights(id_train):    
    # Generate reference population individual list
    try:
        with open('./example_data/all_genotypes.fam', 'r') as r, \
            open('ref_id.list', 'w') as w:

            dic = {}
            for i in r:
                f = i.split()
                dic[f[1]] = f[0]
                
            for i in id_train:
                if i in dic:
                    w.write(str(dic[i]) + '\t' + i + '\n')
    except Exception as e:
        print(f"Error when generating ref_id.list: {e}")
        return None
        
    # GWAS
    try:
        plink_exe = "plink"
        
        subprocess.run(
            [plink_exe,
             "--bfile", "./example_data/all_genotypes",
             "--keep", "ref_id.list",
             "--make-bed",
             "--out", "ref_genotypes"
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )        

        if os.path.exists("ref_genotypes.log"):
            os.remove("ref_genotypes.log")
        for fn in glob.glob("*.nosex"):
            os.remove(fn)

        #change -9 to 0 in fam file
        try:
            df = pd.read_csv('ref_genotypes.fam', delim_whitespace=True, header=None)
            unique_values = df.iloc[:, -1].unique()
            
            if set(unique_values) == {1, -9} or set(unique_values) == {-9, 1}:
                df.iloc[:, -1] = df.iloc[:, -1].replace(-9, 0)    # replace -9 with 0         
                df.to_csv('ref_genotypes1.fam', sep=' ', header=False, index=False)
                
            if os.path.exists("ref_genotypes1.fam"):
                os.replace("ref_genotypes1.fam", "ref_genotypes.fam")
        except Exception as e:
            print(f"Error changing fam file: {e}")
            return None

        #run GEMMA
        subprocess.run(
            [gemma_path,
             "-bfile", "ref_genotypes",
             "-miss", "1.0",
             "-gk", "2",
             "-o", "kinship"
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        kinship_file = os.path.join("output", "kinship.sXX.txt")
        subprocess.run(
            [gemma_path,
             "-bfile", "ref_genotypes",
             "-k", kinship_file,
             "-miss", "1.0",
             "-lmm", "1",
             "-o", "gemma_gwas"
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running embedded GWAS steps: {e}")
        return None
        
    # Generate GWAS results
    try:
        assoc_path = os.path.join("output", "gemma_gwas.assoc.txt")
        bim_path   = "./example_data/all_genotypes.bim"
        
        assoc_df = pd.read_csv(assoc_path, delim_whitespace=True)
        bim_df   = pd.read_csv(bim_path, delim_whitespace=True, header=None,
                               names=['CHR','SNP','CM','BP','A1','A2'])

        assoc_df = assoc_df.rename(columns={'rs':'SNP'})
        if 'beta' not in assoc_df.columns or 'se' not in assoc_df.columns:
            print('Please check the format of gemma_gwas.assoc.txt!')
            raise SystemExit(1)

        merged = pd.merge(bim_df[['SNP']], assoc_df[['SNP','beta','se']], on='SNP', how='left').dropna()

        merged['SNP'].to_csv('gwas_snps', index=False, header=False)
        merged[['beta','se']].to_csv('gwas.result', sep='\t', index=False, header=False)
    
    except Exception as e:
        print(f"Error when generating gwas.result and gwas_snps: {e}")
        return None 
    
    #Calculate SNP weights
    try:
        input_file = "gwas.result"
        PI = 0.001
        Navg = 5

        data = np.loadtxt(input_file)

        SNPeffect = data[:, 0]
        stand_error = data[:, 1]
        bf = np.exp(0.5 * (SNPeffect / stand_error) ** 2)

        Nleft = round((Navg - 1) / 2)
        if Nleft < 0:
            Nleft = 0

        print(f"Number of SNPs: {len(bf)}")
        print(f"PI: {PI}")
        print(f"Number of SNPs to left (and right) included in smoothed average: {Nleft}")

        avg = np.zeros((len(bf), 3))

        for i in range(len(bf)):
            left_index = max(0, i - Nleft)
            right_index = min(i + Nleft, len(bf)-1)
            log_bf = np.log(bf[left_index:right_index+1])
            avg_log_bf = np.mean(log_bf)
            avg_bf = np.exp(avg_log_bf)
            
            avg_pp = PI * avg_bf / (PI * avg_bf + (1 - PI))
            
            avg[i, 2] = avg_bf
            avg[i, 1] = avg_pp

        avg[:, 0] = avg[:, 1] * (len(bf) / np.sum(avg[:, 1]))

        np.savetxt("snp_weight.out", avg)
      
    except Exception as e:
        print(f"Error when calculating SNP weights: {e}")
        return None    
        
    try:
        subprocess.run(
            ['plink', '--bfile', 'ref_genotypes', '--extract', 'gwas_snps', '--recodeA'], 
            check=True, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
        
        num_snps_in_plink = int(
            subprocess.check_output("awk '{print NF}' plink.raw | head -n 1", shell=True)
                .decode().strip()
        ) - 6
        
        num_snps_in_snp_weights = int(
            subprocess.check_output("wc -l < snp_weight.out", shell=True)
                .decode().strip()
        )
        
        if num_snps_in_plink != num_snps_in_snp_weights:
            print("Error: the number of SNPs in plink.raw and snp_weight.out is different.")
            return None
            
        snp_weights = np.loadtxt('snp_weight.out')[:, 0]
        np.savetxt("snp_weights.txt", snp_weights)
        
        # Delete the temporary files
        files_to_remove = ['ref_genotypes.bed', 'ref_genotypes.bim', 'ref_genotypes.fam',  
                           'plink.raw', 'plink.log', 'plink.nosex', 
                           'gwas.result', 'ref_id.list', 'snp_weight.out']
        for file in files_to_remove:
            try:
                os.remove(file)
            except FileNotFoundError:
                print(f"Warning: {file} not found.")
                
        # Delete the output folder
        output_folder = 'output'
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        else:
            print(f"{output_folder} folder not found.")       

    except subprocess.CalledProcessError as e:
        print(f"Error when generating snp_weights.txt: {e}")
        snp_weights = None
    
    return snp_weights

# Generate genotypes for training and test sets
def _generate_train_test_geno(id_train, id_test):
    subprocess.run(
        ["plink",
         "--bfile", "./example_data/all_genotypes",
         "--extract", "gwas_snps",
         "--recodeA",
         "--out", "plink"
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )

    df = pd.read_csv("plink.raw", sep=r"\s+", engine="c", dtype=str)
    iids = df.iloc[:, 1].to_numpy()
    X = df.iloc[:, 6:].astype(np.int32).to_numpy(copy=False)
    idx_map = {iid: idx for idx, iid in enumerate(iids)}

    X_train = X[[idx_map[i] for i in id_train], :]
    X_test  = X[[idx_map[i] for i in id_test], :]

    for fn in glob.glob("plink.*"):
        os.remove(fn)

    return X_train, X_test

# Generate phenotypes for training set
def _generate_train_phe(id_train):
    fam = pd.read_csv('./example_data/all_genotypes.fam', sep=r'\s+', header=None, engine='c', dtype={1:str})
    fam.columns = ['FID','IID','Father','Mother','Sex','Pheno']
    pheno_map = dict(zip(fam['IID'].astype(str), fam['Pheno']))

    y_train = np.array([float(pheno_map[i]) for i in id_train], dtype=float)
    
    return y_train


print('Start time:', datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"), flush=True)

# Input data
gemma_path = "/public/home/06025/WORK/Software/mambaforge-pypy3/envs/gemma/bin/gemma"
all_snps = np.loadtxt('./example_data/all_genotypes.bim', dtype=str, usecols=1)
id_train = np.loadtxt('./example_data/train_id.txt', dtype=str)
id_test = np.loadtxt('./example_data/test_id.txt', dtype=str)

# Calculate snp weights
snp_weights = calculate_snp_weights(id_train)

# Generate genotypes for training and test sets
X_train, X_test = _generate_train_test_geno(id_train, id_test)
y_train = _generate_train_phe(id_train)

model_instance = BayesianCombinedKernelRidge(weights=snp_weights)
model_instance.fit(X_train, y_train)

# Gibbs sampling parameters
n_iter = 60000
burn_in = 20000
alpha = 3 / 2
beta = 3 * 1.0 / 2
thinning_interval = 10

# Output variance components
print("Start Gibbs Sampling!")
(thinned_sigma_a2_samples, thinned_sigma_e2_samples, 
 raw_sigma_a2_samples, raw_sigma_e2_samples) = gibbs_sampling(
    n_iter, burn_in, alpha, beta, 
    model_instance, thinning_interval
)
sigma_a2_estimated = np.mean(thinned_sigma_a2_samples)
sigma_e2_estimated = np.mean(thinned_sigma_e2_samples)
with open('variance_components.txt', 'w') as w1:
    w1.write('sigma_a2' + '\t' + str(sigma_a2_estimated) + '\n' +
             'sigma_e2' + '\t' + str(sigma_e2_estimated) + '\n')

# Train and predict using the best model
best_model = BayesianCombinedKernelRidge(
    gamma=2.74e-6, 
    lambda_=0.1, 
    sigma_a2=sigma_a2_estimated, 
    sigma_e2=sigma_e2_estimated, 
    weights=snp_weights
)
best_model.fit(X_train, y_train)
y_test_pred = best_model.predict(X_test)

# Save the prediction results
pred_str = y_test_pred.astype(str)
id_pred = np.column_stack((id_test, pred_str))
np.savetxt('y_test_pred.txt', id_pred, fmt='%s %s', header='ID\tPrediction', comments='')
print('End time:', datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"), flush=True)
