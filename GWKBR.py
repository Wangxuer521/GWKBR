# -*- coding: utf-8 -*-
import glob
import datetime
import os, shutil
import subprocess
import numpy as np
import pandas as pd
from numba import njit
from skopt.space import Real
from scipy.linalg import eigh
from scipy.sparse import diags
from skopt import BayesSearchCV
from sklearn.metrics import make_scorer
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedKFold
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

# Define the unweighted GWKBR model (KBR)
class KBR_Model(BaseEstimator, RegressorMixin):
    def __init__(self, gamma=0.00006, lambda_=1.0, sigma_a2=2000, sigma_e2=700, G=None):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.sigma_a2 = sigma_a2
        self.sigma_e2 = sigma_e2
        self.G = G
    
    def _gaussian_kernel(self, X1, X2, gamma):
        x1_sq = np.sum(X1 * X1, axis=1)[:, None]
        x2_sq = np.sum(X2 * X2, axis=1)[None, :]
        cross = X1 @ X2.T
        d2 = x1_sq + x2_sq - 2.0 * cross
        return np.exp(-gamma * d2)

    def _generate_G_matrix(self, X):
        dosageA = X
        MA = dosageA - 1
        p1A = np.round((np.sum(dosageA, axis=0)) / (dosageA.shape[0] * 2), 3)
        pA = 2 * (p1A - 0.5)
        PA = np.tile(pA, (dosageA.shape[0], 1))
        ZA = MA - PA
        Sum_2pq_A = 2 * np.sum(p1A * (1 - p1A))
        G = np.dot(ZA, ZA.T) / Sum_2pq_A

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
        K = self._gaussian_kernel(X, X, self.gamma)
        self.K = K
        self.X_fit_ = X
        self.y = y

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
        K_test = self._gaussian_kernel(X, self.X_fit_[:, :self.n_features], self.gamma)
        return np.dot(K_test, self.mu_post)

# Define the weighted GWKBR model (GWKBR)
class GWKBR_Model(BaseEstimator, RegressorMixin):
    def __init__(self, gamma=0.00006, lambda_=1.0, sigma_a2=2000, sigma_e2=700, G=None, weights=None):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.sigma_a2 = sigma_a2
        self.sigma_e2 = sigma_e2
        self.G = G
        self.weights = weights
        
    def _weighted_gaussian_kernel(self, X1, X2, gamma, weights):
        X1w = X1 * weights
        X2w = X2 * weights
        x1_sq = np.sum(X1 * X1w, axis=1)[:, None]
        x2_sq = np.sum(X2 * X2w, axis=1)[None, :]
        cross = X1w @ X2.T
        d2 = x1_sq + x2_sq - 2.0 * cross
        return np.exp(-gamma * d2)

    def _generate_G_matrix(self, X):
        dosageA = X
        MA = dosageA - 1
        p1A = np.round((np.sum(dosageA, axis=0)) / (dosageA.shape[0] * 2), 3)
        pA = 2 * (p1A - 0.5)
        PA = np.tile(pA, (dosageA.shape[0], 1))
        ZA = MA - PA
        Sum_2pq_A = 2 * np.sum(p1A * (1 - p1A))
        G = np.dot(ZA, ZA.T) / Sum_2pq_A + np.eye(ZA.shape[0]) * 1e-6

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
        bim_df   = pd.read_csv(bim_path,   delim_whitespace=True, header=None,
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

# Recalculate the SNP weights
def recalculate_snp_weights(id_train_1):
    np.savetxt('id_train_temp.txt', id_train_1, fmt='%s')
    
    # Generate reference population individual list
    try:        
        with open('id_train_temp.txt', 'r') as r1, \
            open('all_genotypes_gwas.fam', 'r') as r2, \
            open('train_id.list', 'w') as w:

            dic = {}
            for i in r2:
                f = i.split()
                dic[f[1]] = f[0]
                
            for i in r1:
                f = i.split()
                if f[0] in dic:
                    w.write(str(dic[f[0]]) + '\t' + f[0] + '\n')
    except Exception as e:
        print(f"Error when generating train_id.list: {e}")
        return None
        
    # GWAS
    try:
        plink_exe = "plink"
        
        subprocess.run(
            [plink_exe,
             "--bfile", "all_genotypes_gwas",
             "--keep", "train_id.list",
             "--make-bed",
             "--out", "train_genotypes"
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )        

        if os.path.exists("train_genotypes.log"):
            os.remove("train_genotypes.log")
        for fn in glob.glob("*.nosex"):
            os.remove(fn)

        #change -9 to 0 in fam file
        try:
            df = pd.read_csv('train_genotypes.fam', delim_whitespace=True, header=None)
            unique_values = df.iloc[:, -1].unique()
            
            if set(unique_values) == {1, -9} or set(unique_values) == {-9, 1}:
                df.iloc[:, -1] = df.iloc[:, -1].replace(-9, 0)    # replace -9 with 0         
                df.to_csv('train_genotypes1.fam', sep=' ', header=False, index=False)
                
            if os.path.exists("train_genotypes1.fam"):
                os.replace("train_genotypes1.fam", "train_genotypes.fam")
        except Exception as e:
            print(f"Error changing fam file: {e}")
            return None

        #run GEMMA
        subprocess.run(
            [gemma_path,
             "-bfile", "train_genotypes",
             "-miss", "1.0",
             "-gk", "2",
             "-o", "kinship"
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        kinship_file = os.path.join("output", "kinship.sXX.txt")
        subprocess.run(
            [gemma_path,
             "-bfile", "train_genotypes",
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
        bim_path   = "all_genotypes_gwas.bim"
        
        assoc_df = pd.read_csv(assoc_path, delim_whitespace=True)
        bim_df   = pd.read_csv(bim_path,   delim_whitespace=True, header=None,
                               names=['CHR','SNP','CM','BP','A1','A2'])

        assoc_df = assoc_df.rename(columns={'rs':'SNP'})
        if 'beta' not in assoc_df.columns or 'se' not in assoc_df.columns:
            print('Please check the format of gemma_gwas.assoc.txt!')
            raise SystemExit(1)

        merged = pd.merge(bim_df[['SNP']], assoc_df[['SNP','beta','se']], on='SNP', how='left').dropna()

        merged['SNP'].to_csv('gwas_snps_temp', index=False, header=False)
        merged[['beta','se']].to_csv('gwas.result', sep='\t', index=False, header=False)
    
    except Exception as e:
        print(f"Error when generating gwas.result and gwas_snps_temp: {e}")
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
            ['plink', '--bfile', 'train_genotypes', '--extract', 'gwas_snps_temp', '--recodeA'], 
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
            
        snp_weights_cv = np.loadtxt('snp_weight.out')[:, 0]
        
    except subprocess.CalledProcessError as e:
        print(f"Error when generating snp_weight.out: {e}")
        snp_weights_cv = None
    
    return snp_weights_cv

# Define hyperparameter optimization (quantitative traits)
def cross_validate_and_optimize_continuous(model_class, param_space, X_train, y_train, snp_weights=None):
    if snp_weights is not None:
        model = BayesSearchCV(
            model_class(
                weights=snp_weights, 
                sigma_a2=sigma_a2_estimated_GWKBR, 
                sigma_e2=sigma_e2_estimated_GWKBR
                ),
            param_space, 
            n_iter=30, 
            cv=5, 
            n_jobs=-1, 
            random_state=42, 
            scoring=pearson_scorer
        )
    else:
        model = BayesSearchCV(
            model_class(
                sigma_a2=sigma_a2_estimated_KBR, 
                sigma_e2=sigma_e2_estimated_KBR
                ),
            param_space, 
            n_iter=30, 
            cv=5, 
            n_jobs=-1, 
            random_state=42, 
            scoring=pearson_scorer
        )
    
    model.fit(X_train, y_train)
    return model.best_params_

# Defining hyperparameter optimization (binary traits)
def cross_validate_and_optimize_binary(model_class, param_space, X_train, y_train, snp_weights=None):
    if snp_weights is not None:
        model = BayesSearchCV(
            model_class(
                weights=snp_weights, 
                sigma_a2=sigma_a2_estimated_GWKBR, 
                sigma_e2=sigma_e2_estimated_GWKBR
                ),
            param_space, 
            n_iter=50, 
            cv=5, 
            n_jobs=-1, 
            random_state=42, 
            scoring=auroc_scorer
        )
    else:
        model = BayesSearchCV(
            model_class(
                sigma_a2=sigma_a2_estimated_KBR, 
                sigma_e2=sigma_e2_estimated_KBR
                ),
            param_space,
            n_iter=50, 
            cv=5, 
            n_jobs=-1, 
            random_state=42, 
            scoring=auroc_scorer
        )
    
    model.fit(X_train, y_train)
    return model.best_params_

# Define the Pearson correlation coefficient score
def pearson_corr_coef(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

# Define the AUROC score
def auroc_score(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

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

# Compare the performance of KBR and GWKBR in cross-validation
def cross_validate_model_selection(model_class, best_params, X_train, y_train, id_train, model_name, snp_weights=None):
    rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)
    auroc_scores = []
    pearson_scores = []
    
    subprocess.run(
        ["plink", 
        "--bfile", "./example_data/all_genotypes", 
        "--extract", "gwas_snps", 
        "--make-bed",
        "--out", "all_genotypes_gwas"
        ],
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL,
        check=True
    )
    gwas_snps = np.loadtxt('gwas_snps', dtype=str)

    for fold, (train_idx, val_idx) in enumerate(rkf.split(X_train)):
        X_train_1, X_val_1 = X_train[train_idx], X_train[val_idx]
        y_train_1, y_val_1 = y_train[train_idx], y_train[val_idx]
        id_train_1, id_val_1 = id_train[train_idx], id_train[val_idx]
        
        # Recalculate the SNP weights
        if model_name == "GWKBR":   
            snp_weights_cv = recalculate_snp_weights(id_train_1)
            
            # Extract the positions and select the corresponding columns.
            gwas_snps_temp = np.loadtxt('gwas_snps_temp', dtype=str)
            loc = {snp: i for i, snp in enumerate(gwas_snps.tolist())}
            snp_indices = [loc[s] for s in gwas_snps_temp if s in loc]
            if len(snp_indices) != len(gwas_snps_temp):
                missing = set(gwas_snps_temp) - set(loc.keys())
                for ms in list(missing)[:5]:
                    print(f"Warning: SNP {ms} not found in gwas_snps.")
            X_train_1 = X_train_1[:, snp_indices]
            X_val_1   = X_val_1[:, snp_indices]
           
            # Delete the temporary files
            files_to_remove = ['id_train_temp.txt', 'train_id.list',
                               'train_genotypes.bed', 'train_genotypes.bim', 'train_genotypes.fam', 
                               'gwas.result', 'gwas_snps_temp', 'snp_weight.out', 
                               'plink.raw', 'plink.log', 'plink.nosex']
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
            
        else:
            snp_weights_cv = None
        
        # Train the model
        if model_name == "GWKBR":
            model = model_class(
                **best_params, 
                sigma_a2=sigma_a2_estimated_GWKBR, 
                sigma_e2=sigma_e2_estimated_GWKBR, 
                weights=snp_weights_cv
                )
        else:
            model = model_class(
                **best_params, 
                sigma_a2=sigma_a2_estimated_KBR, 
                sigma_e2=sigma_e2_estimated_KBR
                )        

        model.fit(X_train_1, y_train_1)
        y_val_pred = model.predict(X_val_1)
        
        if trait_type == 'binary':
            auroc_scores.append(auroc_score(y_val_1, y_val_pred))
        else:
            pearson_scores.append(pearson_corr_coef(y_val_1, y_val_pred))
    
    for fn in glob.glob("all_genotypes_gwas.*"):
        os.remove(fn)
        
    if trait_type == 'binary':
        print(f"{model_name} - All AUROC scores:", auroc_scores)
        return np.mean(auroc_scores)
    else:
        print(f"{model_name} - All Pearson scores:", pearson_scores)
        return np.mean(pearson_scores)


# Output start time
print('Start time:', datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"), flush=True)

# Input data
gemma_path = "/public/home/06025/WORK/Software/mambaforge-pypy3/envs/gemma/bin/gemma"
id_train = np.loadtxt('./example_data/train_id.txt', dtype=str)
id_test = np.loadtxt('./example_data/test_id.txt', dtype=str)

# Calculate snp weights
snp_weights = calculate_snp_weights(id_train)

# Generate genotypes for training and test sets
X_train, X_test = _generate_train_test_geno(id_train, id_test)
y_train = _generate_train_phe(id_train)

# Determine the trait type
if np.all(np.isin(y_train, [0, 1])):
    trait_type = 'binary'
    auroc_scorer = make_scorer(auroc_score, greater_is_better=True)
    print('AUROC Scorer selected.')
else:
    trait_type = 'continuous'
    pearson_scorer = make_scorer(pearson_corr_coef, greater_is_better=True)
    print('Pearson Scorer selected.')

# Gibbs sampling parameters
n_iter = 6000
burn_in = 2000
alpha = 3 / 2
beta = 3 * 1.0 / 2
thinning_interval = 10

# Perform Gibbs sampling for KBR
model_instance_KBR = KBR_Model()
model_instance_KBR.fit(X_train, y_train)
print("Start Gibbs Sampling for KBR!")
(thinned_sigma_a2_samples_KBR, thinned_sigma_e2_samples_KBR, 
 raw_sigma_a2_samples_KBR, raw_sigma_e2_samples_KBR) = gibbs_sampling(
    n_iter, burn_in, alpha, beta, 
    model_instance_KBR, thinning_interval
)
sigma_a2_estimated_KBR = np.mean(thinned_sigma_a2_samples_KBR)
sigma_e2_estimated_KBR = np.mean(thinned_sigma_e2_samples_KBR)

# Perform Gibbs sampling for GWKBR.
model_instance_GWKBR = GWKBR_Model(weights=snp_weights)
model_instance_GWKBR.fit(X_train, y_train)
print("Start Gibbs Sampling for GWKBR!")
(thinned_sigma_a2_samples_GWKBR, thinned_sigma_e2_samples_GWKBR,
 raw_sigma_a2_samples_GWKBR, raw_sigma_e2_samples_GWKBR) = gibbs_sampling(
    n_iter, burn_in, alpha, beta,
    model_instance_GWKBR, thinning_interval
)
sigma_a2_estimated_GWKBR = np.mean(thinned_sigma_a2_samples_GWKBR)
sigma_e2_estimated_GWKBR = np.mean(thinned_sigma_e2_samples_GWKBR)

# Hyperparameter optimization
if trait_type == 'binary':
    print("Optimizing KBR model...")
    param_space_KBR = {
        "gamma": Real(1e-8, 1e-3, prior="log-uniform"), 
        "lambda_": Real(0.1, 10, prior="log-uniform")
        }
    best_params_KBR = cross_validate_and_optimize_binary(
        KBR_Model, 
        param_space_KBR, 
        X_train, y_train
        )
           
    print("Optimizing GWKBR model...")
    param_space_GWKBR = {
        "gamma": Real(1e-8, 1e-3, prior="log-uniform"),
        "lambda_": Real(0.1, 10, prior="log-uniform")
        }
    best_params_GWKBR = cross_validate_and_optimize_binary(
        GWKBR_Model, 
        param_space_GWKBR, 
        X_train, y_train, 
        snp_weights=snp_weights
        )
else: 
    print("Optimizing KBR model...")
    param_space_KBR = {
        "gamma": Real(1e-8, 1e-3, prior="log-uniform"),
        "lambda_": Real(0.1, 10, prior="log-uniform")
        }
    best_params_KBR = cross_validate_and_optimize_continuous(
        KBR_Model, 
        param_space_KBR, 
        X_train, y_train
        )
           
    print("Optimizing GWKBR model...")
    param_space_GWKBR = {
        "gamma": Real(1e-8, 1e-3, prior="log-uniform"), 
        "lambda_": Real(0.1, 10, prior="log-uniform")
        }
    best_params_GWKBR = cross_validate_and_optimize_continuous(
        GWKBR_Model, 
        param_space_GWKBR,
        X_train, y_train, 
        snp_weights=snp_weights
        )

# Select the best model
mean_score_KBR = cross_validate_model_selection(
    KBR_Model, 
    best_params_KBR, 
    X_train, y_train,
    id_train,
    "KBR"
    )
mean_score_GWKBR = cross_validate_model_selection(
    GWKBR_Model, 
    best_params_GWKBR,
    X_train, y_train, 
    id_train,
    "GWKBR", 
    snp_weights=snp_weights
    )

if mean_score_KBR > mean_score_GWKBR:
    print(f"KBR is better with mean score: {mean_score_KBR}")
    best_model_class = KBR_Model
    best_params = best_params_KBR
    with open('best_params.txt', 'w') as w1:
        w1.write('kernel' + '\t' + 'unweighted-kernel' + '\n')
        for k, v in best_params_KBR.items():
            w1.write(str(k) + '\t' + str(v) + '\n')
    with open('variance_components.txt', 'w') as w2:
        w2.write('sigma_a2' + '\t' + str(sigma_a2_estimated_KBR) + '\n' + 
                 'sigma_e2' + '\t' + str(sigma_e2_estimated_KBR) + '\n')
else:
    print(f"GWKBR is better with mean score: {mean_score_GWKBR}")
    best_model_class = GWKBR_Model
    best_params = best_params_GWKBR
    with open('best_params.txt', 'w') as w1:
        w1.write('kernel' + '\t' + 'weighted-kernel' + '\n')
        for k, v in best_params_GWKBR.items():
            w1.write(str(k) + '\t' + str(v) + '\n')
    with open('variance_components.txt', 'w') as w2:
        w2.write('sigma_a2' + '\t' + str(sigma_a2_estimated_GWKBR) + '\n' + 
                 'sigma_e2' + '\t' + str(sigma_e2_estimated_GWKBR) + '\n')

# Train and predict using the best model
if best_model_class == GWKBR_Model:
    best_model = best_model_class(
        **best_params, 
        sigma_a2=sigma_a2_estimated_GWKBR, 
        sigma_e2=sigma_e2_estimated_GWKBR, 
        weights=snp_weights
        )
else:
    best_model = best_model_class(
        **best_params, 
        sigma_a2=sigma_a2_estimated_KBR, 
        sigma_e2=sigma_e2_estimated_KBR
        )
best_model.fit(X_train, y_train)
y_test_pred = best_model.predict(X_test)

# Save the prediction results
pred_str = y_test_pred.astype(str)
id_pred = np.column_stack((id_test, pred_str))
np.savetxt('y_test_pred.txt', id_pred, fmt='%s', delimiter='\t', 
           header='ID\tPrediction', comments='')
print('End time:', datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"), flush=True)
