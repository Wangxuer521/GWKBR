# -*- coding: utf-8 -*-
import argparse
import datetime
import glob
import os
import re
import shutil
import subprocess
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from numba import njit
from scipy.linalg import eigh
from scipy.sparse import diags
from scipy.stats import pearsonr
from skopt import BayesSearchCV
from skopt.space import Real
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import RepeatedKFold

# Define the unweighted GWKBR model (K model)
class K_Model(BaseEstimator, RegressorMixin):
    def __init__(self, gamma=0.00005, lambda_=1.0, sigma_a2=1.0, sigma_e2=1.0, G=None):
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
        K_test = self._gaussian_kernel(
            X,
            self.X_fit_[:, :self.n_features],
            self.gamma
        )
        return np.dot(K_test, self.mu_post)

# Define the weighted GWKBR model (Kw model)
class Kw_Model(BaseEstimator, RegressorMixin):
    def __init__(self, gamma=0.00005, lambda_=1.0, sigma_a2=1.0, sigma_e2=1.0, G=None, weights=None):
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

# Calculate the SNP weights
def calculate_snp_weights(id_train):
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate reference population individual list
    try:
        with open(f"{geno_prefix}.fam", 'r') as r, open('ref_id.list', 'w') as w:
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
             "--bfile", geno_prefix,
             "--keep", "ref_id.list",
             "--make-bed",
             "--out", "ref_genotypes"
            ],
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL, 
            check=True
        )
        
        if os.path.exists("ref_genotypes.log"):
            os.remove("ref_genotypes.log")
        for fn in glob.glob("*.nosex"):
            os.remove(fn)

        # change -9 to 0 in fam file
        try:
            df = pd.read_csv('ref_genotypes.fam', delim_whitespace=True, header=None)
            unique_values = df.iloc[:, -1].unique()
            
            if set(unique_values) == {1, -9} or set(unique_values) == {-9, 1}:
                df.iloc[:, -1] = df.iloc[:, -1].replace(-9, 0)  # replace -9 with 0
                df.to_csv('ref_genotypes1.fam', sep=' ', header=False, index=False)
                
            if os.path.exists("ref_genotypes1.fam"):
                os.replace("ref_genotypes1.fam", "ref_genotypes.fam")
        except Exception as e:
            print(f"Error changing fam file: {e}")
            return None

        # run GEMMA
        subprocess.run(
            [gemma_path,
             "-bfile", "ref_genotypes",
             "-miss", "1.0",
             "-gk", "2",
             "-o", "kinship"
            ],
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL, 
            check=True
        )
        kinship_file = os.path.join("output", "kinship.sXX.txt")
        subprocess.run(
            [gemma_path,
             "-p", "ref_genotypes.fam",
             "-k", kinship_file,
             "-n", "6",
             "-vc", "2",
             "-o", "variance_component"
            ],
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL, 
            check=True
        )
        subprocess.run(
            [gemma_path,
             "-bfile", "ref_genotypes",
             "-k", kinship_file,
             "-miss", "1.0",
             "-lmm", "1",
             "-o", "gemma_gwas"
            ],
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL, 
            check=True
        )
        
    except subprocess.CalledProcessError as e:
        print(f"Error running embedded GWAS steps: {e}")
        return None

    # Generate GWAS results
    try:
        assoc_path = os.path.join("output", "gemma_gwas.assoc.txt")
        bim_path = f"{geno_prefix}.bim"
        
        assoc_df = pd.read_csv(assoc_path, delim_whitespace=True)
        bim_df = pd.read_csv(bim_path, delim_whitespace=True, header=None,
                              names=['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2'])
                              
        assoc_df = assoc_df.rename(columns={'rs': 'SNP'})
        if 'beta' not in assoc_df.columns or 'se' not in assoc_df.columns:
            print('Please check the format of gemma_gwas.assoc.txt!')
            raise SystemExit(1)
            
        merged = pd.merge(
            bim_df[['SNP']],
            assoc_df[['SNP', 'beta', 'se']],
            on='SNP',
            how='left'
        ).dropna()
        
        merged['SNP'].to_csv(os.path.join(results_dir, 'gwas_snps'), index=False, header=False)
        merged[['beta', 'se']].to_csv('gwas.result', sep='\t', index=False, header=False)
        
    except Exception as e:
        print(f"Error when generating gwas.result and gwas_snps: {e}")
        return None

    # Calculate SNP weights
    try:
        input_file = "gwas.result"
        PI = 0.001
        Navg = 5
        
        data = np.loadtxt(input_file)
        
        SNPeffect = data[:, 0]
        stand_error = data[:, 1]
        lr = 0.5 * (SNPeffect / stand_error) ** 2
        
        Nleft = round((Navg - 1) / 2)
        if Nleft < 0:
            Nleft = 0
            
        print(f"Number of SNPs: {len(lr)}")
        print(f"PI: {PI}")
        print(f"Number of SNPs to left (and right) included in smoothed average: {Nleft}")
        
        avg = np.zeros((len(lr), 3))
        
        for i in range(len(lr)):
            left_index = max(0, i - Nleft)
            right_index = min(i + Nleft, len(lr) - 1)
            avg_lr = np.mean(lr[left_index:right_index+1])
            
            avg_pp = PI * np.exp(avg_lr) / (PI * np.exp(avg_lr) + (1 - PI))
            
            avg[i, 2] = avg_lr
            avg[i, 1] = avg_pp
            
        avg[:, 0] = avg[:, 1] * (len(lr) / np.sum(avg[:, 1]))
        
        np.savetxt("snp_weight.out", avg)
        
    except Exception as e:
        print(f"Error when calculating SNP weights: {e}")
        return None

    try:
        subprocess.run(
            ['plink',
             '--bfile', 'ref_genotypes',
             '--extract', os.path.join(results_dir, 'gwas_snps'),
             '--recodeA'
            ],
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
        np.savetxt(os.path.join(results_dir, "snp_weights.txt"), snp_weights)
        
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
        log_file = os.path.join(output_folder, 'variance_component.log.txt')
        if os.path.exists(output_folder):
            if os.path.exists(log_file):
                shutil.copy(log_file, os.path.join(results_dir, 'variance_component.log.txt'))
            else:
                print("variance_component.log.txt not found in output folder.")
            shutil.rmtree(output_folder)
        else:
            print(f"{output_folder} folder not found.")
            
    except subprocess.CalledProcessError as e:
        print(f"Error when generating snp_weights.txt: {e}")
        snp_weights = None

    return snp_weights

# Estimate variance components
def estimate_variance_components(id_train):
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate reference population individual list
    try:
        with open(f"{geno_prefix}.fam", 'r') as r, open('ref_id.list', 'w') as w:
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

    # Estimate variance components
    try:
        plink_exe = "plink"
        subprocess.run(
            [plink_exe,
             "--bfile", geno_prefix,
             "--keep", "ref_id.list",
             "--make-bed",
             "--out", "ref_genotypes"
            ],
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL, 
            check=True
        )
        
        if os.path.exists("ref_genotypes.log"):
            os.remove("ref_genotypes.log")
        for fn in glob.glob("*.nosex"):
            os.remove(fn)
            
        # change -9 to 0 in fam file
        try:
            df = pd.read_csv('ref_genotypes.fam', delim_whitespace=True, header=None)
            unique_values = df.iloc[:, -1].unique()
            if set(unique_values) == {1, -9} or set(unique_values) == {-9, 1}:
                df.iloc[:, -1] = df.iloc[:, -1].replace(-9, 0)  # replace -9 with 0
                df.to_csv('ref_genotypes1.fam', sep=' ', header=False, index=False)
            if os.path.exists("ref_genotypes1.fam"):
                os.replace("ref_genotypes1.fam", "ref_genotypes.fam")
        except Exception as e:
            print(f"Error changing fam file: {e}")
            return None

        # run GEMMA
        subprocess.run(
            [gemma_path,
             "-bfile", "ref_genotypes",
             "-miss", "1.0",
             "-gk", "2",
             "-o", "kinship"
            ],
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL, 
            check=True
        )       
        kinship_file = os.path.join("output", "kinship.sXX.txt")
        subprocess.run(
            [gemma_path,
             "-p", "ref_genotypes.fam",
             "-k", kinship_file,
             "-n", "6",
             "-vc", "2",
             "-o", "variance_component"
            ],
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL, 
            check=True
        )
        
        # Delete the temporary files
        files_to_remove = ['ref_genotypes.bed', 'ref_genotypes.bim', 'ref_genotypes.fam', 'ref_id.list']
        for file in files_to_remove:
            try:
                os.remove(file)
            except FileNotFoundError:
                print(f"Warning: {file} not found.")
                
        # Delete the output folder
        output_folder = 'output'
        log_file = os.path.join(output_folder, 'variance_component.log.txt')
        if os.path.exists(output_folder):
            if os.path.exists(log_file):
                shutil.copy(log_file, os.path.join(results_dir, 'variance_component.log.txt'))
            else:
                print("variance_component.log.txt not found in output folder.")
            shutil.rmtree(output_folder)
        else:
            print(f"{output_folder} folder not found.")
            
    except subprocess.CalledProcessError as e:
        print(f"Error estimating variance component steps: {e}")
        return None

# Recalculate the SNP weights
def recalculate_snp_weights(id_train_1, verbose=False):
    np.savetxt('id_train_temp.txt', id_train_1, fmt='%s')
    
    # Generate reference population individual list
    try:
        with (
            open("id_train_temp.txt", "r") as r1,
            open("all_genotypes_gwas.fam", "r") as r2,
            open("train_id.list", "w") as w,
        ):
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
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL, 
            check=True
        )
        
        if os.path.exists("train_genotypes.log"):
            os.remove("train_genotypes.log")
        for fn in glob.glob("*.nosex"):
            os.remove(fn)
            
        # change -9 to 0 in fam file
        try:
            df = pd.read_csv('train_genotypes.fam', delim_whitespace=True, header=None)
            unique_values = df.iloc[:, -1].unique()
            
            if set(unique_values) == {1, -9} or set(unique_values) == {-9, 1}:
                df.iloc[:, -1] = df.iloc[:, -1].replace(-9, 0)
                df.to_csv('train_genotypes1.fam', sep=' ', header=False, index=False)
            
            if os.path.exists("train_genotypes1.fam"):
                os.replace("train_genotypes1.fam", "train_genotypes.fam")
        except Exception as e:
            print(f"Error changing fam file: {e}")
            return None

        # run GEMMA
        subprocess.run(
            [gemma_path,
             "-bfile", "train_genotypes",
             "-miss", "1.0",
             "-gk", "2",
             "-o", "kinship"
            ],
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL, 
            check=True
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
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL, 
            check=True
        )
        
    except subprocess.CalledProcessError as e:
        print(f"Error running embedded GWAS steps: {e}")
        return None

    # Generate GWAS results
    try:
        assoc_path = os.path.join("output", "gemma_gwas.assoc.txt")
        bim_path = "all_genotypes_gwas.bim"
        
        assoc_df = pd.read_csv(assoc_path, delim_whitespace=True)
        bim_df = pd.read_csv(bim_path, delim_whitespace=True, header=None,
                             names=['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2'])
                               
        assoc_df = assoc_df.rename(columns={'rs': 'SNP'})
        if 'beta' not in assoc_df.columns or 'se' not in assoc_df.columns:
            print('Please check the format of gemma_gwas.assoc.txt!')
            raise SystemExit(1)
            
        merged = pd.merge(
            bim_df[['SNP']],
            assoc_df[['SNP', 'beta', 'se']],
            on='SNP',
            how='left'
        ).dropna()
        
        merged['SNP'].to_csv('gwas_snps_temp', index=False, header=False)
        merged[['beta', 'se']].to_csv('gwas.result', sep='\t', index=False, header=False)
        
    except Exception as e:
        print(f"Error when generating gwas.result and gwas_snps_temp: {e}")
        return None

    # Calculate SNP weights
    try:
        input_file = "gwas.result"
        PI = 0.001
        Navg = 5
        
        data = np.loadtxt(input_file)
        
        SNPeffect = data[:, 0]
        stand_error = data[:, 1]
        lr = 0.5 * (SNPeffect / stand_error) ** 2
        
        Nleft = round((Navg - 1) / 2)
        if Nleft < 0:
            Nleft = 0
            
        if verbose:
            print(f"Number of SNPs: {len(lr)}")
            print(f"PI: {PI}")
            print(f"Number of SNPs to left (and right) included in smoothed average: {Nleft}")
        
        avg = np.zeros((len(lr), 3))
        
        for i in range(len(lr)):
            left_index = max(0, i - Nleft)
            right_index = min(i + Nleft, len(lr) - 1)
            avg_lr = np.mean(lr[left_index:right_index+1])
            
            avg_pp = PI * np.exp(avg_lr) / (PI * np.exp(avg_lr) + (1 - PI))
            
            avg[i, 2] = avg_lr
            avg[i, 1] = avg_pp
            
        avg[:, 0] = avg[:, 1] * (len(lr) / np.sum(avg[:, 1]))
        
        np.savetxt("snp_weight.out", avg)   
        
    except Exception as e:
        print(f"Error when calculating SNP weights: {e}")
        return None

    try:
        subprocess.run(
            ['plink',
             '--bfile', 'train_genotypes',
             '--extract', 'gwas_snps_temp',
             '--recodeA'
            ],
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
                sigma_a2=sigma_a2_estimated,
                sigma_e2=sigma_e2_estimated
            ),
            param_space,
            n_iter=50,
            cv=5,
            n_jobs=-1,
            random_state=42,
            scoring=pearson_scorer
        )
    else:
        model = BayesSearchCV(
            model_class(
                sigma_a2=sigma_a2_estimated,
                sigma_e2=sigma_e2_estimated
            ),
            param_space,
            n_iter=50,
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
                sigma_a2=sigma_a2_estimated,
                sigma_e2=sigma_e2_estimated
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
                sigma_a2=sigma_a2_estimated,
                sigma_e2=sigma_e2_estimated
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
def _generate_train_test_geno(id_train, id_test, filter_snps=False):
    # Run PLINK to generate raw genotype data (optionally filter SNPs)
    plink_cmd = ["plink", "--bfile", geno_prefix]
    
    if filter_snps:
        plink_cmd += ["--extract", os.path.join("results", "gwas_snps")]
    plink_cmd += ["--recodeA", "--out", "plink"]
    
    subprocess.run(
        plink_cmd, 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL, 
        check=True
    )
    
    df = pd.read_csv("plink.raw", sep=r"\s+", engine="c", dtype=str)
    iids = df.iloc[:, 1].to_numpy()
    X = df.iloc[:, 6:].astype(np.int32).to_numpy(copy=False)
    idx_map = {iid: idx for idx, iid in enumerate(iids)}
    X_train = X[[idx_map[i] for i in id_train], :]
    X_test = X[[idx_map[i] for i in id_test], :]
    
    for fn in glob.glob("plink.*"):
        os.remove(fn)
        
    return X_train, X_test

# Generate phenotypes for training set
def _generate_train_phe(id_train):
    fam = pd.read_csv(f"{geno_prefix}.fam", sep=r'\s+', header=None, engine='c', dtype={1: str})
    fam.columns = ['FID', 'IID', 'Father', 'Mother', 'Sex', 'Pheno']
    pheno_map = dict(zip(fam['IID'].astype(str), fam['Pheno']))
    y_train = np.array([float(pheno_map[i]) for i in id_train], dtype=float)
    
    return y_train

# Obtain variance component estimates
def read_sigma2_from_log(log_path: str):
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if 'sigma2 estimates' in line:
                floats = [float(x) for x in re.findall(r'[-+]?\d+\.\d+|\d+\.\d*', line)]
                if len(floats) >= 2:
                    return floats[0], floats[1]
                raise ValueError(f"Found 'sigma2 estimates' but not two floats: {line.strip()}")
    raise RuntimeError("No line containing 'sigma2 estimates' found in log.")

# Compare the performance of K model and Kw model in cross-validation
def cross_validate_model_selection(model_class, best_params, X_train, y_train, id_train, model_name, snp_weights=None):
    rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)

    subprocess.run(
        ["plink",
         "--bfile", geno_prefix,
         "--extract", os.path.join("results", "gwas_snps"),
         "--make-bed",
         "--out", "all_genotypes_gwas"],
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL, 
        check=True
    )
    
    gwas_snps = np.loadtxt(os.path.join('results', 'gwas_snps'), dtype=str)
    
    # Generate index for each fold
    fold_tasks = [(fold, train_idx, val_idx) for fold, (train_idx, val_idx) in enumerate(rkf.split(X_train))]
    
    # Set the level of parallelism (default: 3, can be overridden by environment variables)
    JOBLIB_TEMP = f"joblib_tmp_{model_name}"
    os.makedirs(JOBLIB_TEMP, exist_ok=True)
    _default_n = max(1, min(3, len(fold_tasks)))
    n_jobs_cv = int(os.environ.get("GWKBR_CV_NJOBS", str(_default_n)))
    
    # Run each fold
    def _run_one_fold(fold_no, train_idx, val_idx, X_train_arg, y_train_arg, id_train_arg, gwas_snps_arg):
        X_tr = X_train_arg[train_idx]; X_va = X_train_arg[val_idx]
        y_tr = y_train_arg[train_idx]; y_va = y_train_arg[val_idx]
        id_tr = id_train_arg[train_idx]
        
        if model_name == "Kw":
            workdir = f"cv_fold_{fold_no}"
            os.makedirs(workdir, exist_ok=True)
            
            for ext in (".bed", ".bim", ".fam"):
                src = f"all_genotypes_gwas{ext}"
                dst = os.path.join(workdir, f"all_genotypes_gwas{ext}")
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
                    
            cwd0 = os.getcwd()
            os.chdir(workdir)
            
            try:
                snp_weights_cv = recalculate_snp_weights(id_tr, verbose=False)
                
                # Perform column selection on X matrix
                gwas_snps_temp = np.loadtxt('gwas_snps_temp', dtype=str)
                loc = {snp: i for i, snp in enumerate(gwas_snps_arg.tolist())}
                snp_indices = [loc[s] for s in gwas_snps_temp if s in loc]
                if len(snp_indices) != len(gwas_snps_temp):
                    missing = set(gwas_snps_temp) - set(loc.keys())
                    for ms in list(missing)[:5]:
                        print(f"Warning: SNP {ms} not found in gwas_snps.")
                X_tr_use = X_tr[:, snp_indices]
                X_va_use = X_va[:, snp_indices]
                
                # Training & prediction
                model = model_class(
                    **best_params,
                    sigma_a2=sigma_a2_estimated,
                    sigma_e2=sigma_e2_estimated,
                    weights=snp_weights_cv
                )
                model.fit(X_tr_use, y_tr)
                y_va_pred = model.predict(X_va_use)
                
                # Calculate scores by trait type
                if trait_type == 'binary':
                    score = auroc_score(y_va, y_va_pred)
                else:
                    score = pearson_corr_coef(y_va, y_va_pred)
                    
            finally:
                os.chdir(cwd0)
                shutil.rmtree(workdir, ignore_errors=True)
                
        else:
            model = model_class(
                **best_params,
                sigma_a2=sigma_a2_estimated,
                sigma_e2=sigma_e2_estimated
            )
            
            model.fit(X_tr, y_tr)
            y_va_pred = model.predict(X_va)
            
            if trait_type == 'binary':
                score = auroc_score(y_va, y_va_pred)
            else:
                score = pearson_corr_coef(y_va, y_va_pred)
                
        return score

    # Run all folds in parallel
    with parallel_backend('loky', inner_max_num_threads=1):
        results = Parallel(
            n_jobs=n_jobs_cv,
            prefer="processes",
            batch_size=1,
            verbose=10,
            max_nbytes="1M",
            temp_folder=JOBLIB_TEMP
        )(
            delayed(_run_one_fold)(
                fold_no, tr_idx, va_idx,
                X_train, y_train, id_train, gwas_snps
            )
            for (fold_no, tr_idx, va_idx) in fold_tasks
        )
        
    # Delete the temporary files
    for fn in glob.glob("all_genotypes_gwas.*"):
        os.remove(fn)
        
    # Delete the joblib temporary directory
    shutil.rmtree(JOBLIB_TEMP, ignore_errors=True)
    
    # Print and return the mean
    if trait_type == 'binary':
        print(f"{model_name} - All AUROC scores:", results)
        return float(np.mean(results))
    else:
        print(f"{model_name} - All Pearson scores:", results)
        return float(np.mean(results))


if __name__ == "__main__":
    # Set command line argument parsing
    parser = argparse.ArgumentParser(description="GWKBR model applied to genomic prediction")
    parser.add_argument(
        "--train_id", 
        required=True, 
        help="Path to training set ID list file"
    )
    parser.add_argument(
        "--test_id", 
        required=True, 
        help="Path to test set ID list file"
    )
    parser.add_argument(
        "--geno", 
        required=True, 
        help="Prefix of genotype data files (without extension)"
    )
    parser.add_argument(
        "--predict", 
        action="store_true", 
        help="Run prediction mode (requires --model, --gamma, --lambda)"
    )
    parser.add_argument(
        "--model", 
        choices=["K", "Kw"], 
        help="Model type for prediction (K model, Kw model)"
    )
    parser.add_argument(
        "--gamma", 
        type=float, 
        dest="gamma", 
        help="Gamma hyperparameter (required with --predict)"
    )
    parser.add_argument(
        "--lambda", 
        type=float, 
        dest="lambda_", 
        help="Lambda hyperparameter (required with --predict)"
    )
    args = parser.parse_args()
    
    os.makedirs('results', exist_ok=True)

    # Output start time
    print('Start time:', datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"), flush=True)   

    # Input data
    gemma_path = "/public/home/06025/WORK/Software/mambaforge-pypy3/envs/gemma/bin/gemma"
    geno_prefix = args.geno
    id_train = np.loadtxt(args.train_id, dtype=str)
    id_test = np.loadtxt(args.test_id, dtype=str)

    # Prediction mode: use specified model and hyperparameters
    if args.predict:
        # Require model, gamma, and lambda
        if args.model is None or args.gamma is None or args.lambda_ is None:
            print("Error: --predict requires --model, --gamma, and --lambda to be specified.")
            sys.exit(1)
            
        if args.model == "K":
            # Estimate variance components on training set
            estimate_variance_components(id_train)
            
            # K model prediction
            X_train, X_test = _generate_train_test_geno(id_train, id_test, filter_snps=False)
            y_train = _generate_train_phe(id_train)
            
            # Initial model fit
            model_instance = K_Model()
            model_instance.fit(X_train, y_train)
            
            # Variance component acquisition
            sigma_a2_estimated = None
            sigma_e2_estimated = None
            
            for _p in [os.path.join('results', 'variance_component.log.txt')]:
                if os.path.exists(_p):
                    try:
                        sigma_a2_estimated, sigma_e2_estimated = read_sigma2_from_log(_p)
                        print(f"[Variance components] Read from: {_p}")
                        print(f"[Variance components] sigma_a2 = {sigma_a2_estimated}, sigma_e2 = {sigma_e2_estimated}")
                        break
                    except Exception as e:
                        print(f"Warning: failed to parse '{_p}': {e}")
                        
            if sigma_a2_estimated is None or sigma_e2_estimated is None:
                raise FileNotFoundError(
                    "Unable to locate or parse 'variance_component.log.txt'. "
                    "Checked: " + os.path.join('results', 'variance_component.log.txt')
                )

            # Train and predict using the best model (K model with fixed hyperparameters)
            best_model = K_Model(
                gamma=args.gamma,
                lambda_=args.lambda_,
                sigma_a2=sigma_a2_estimated,
                sigma_e2=sigma_e2_estimated
            )
            best_model.fit(X_train, y_train)
            y_test_pred = best_model.predict(X_test)
            
            # Save the prediction results
            pred_str = y_test_pred.astype(str)
            id_pred = np.column_stack((id_test, pred_str))
            np.savetxt(os.path.join('results', 'y_test_pred.txt'), id_pred, fmt='%s', 
                       delimiter='\t',header='ID\tPrediction', comments='')

            print('End time:', datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"), flush=True)
                
        elif args.model == "Kw":
            # Kw model prediction
            
            # Calculate SNP weights using training set
            snp_weights = calculate_snp_weights(id_train)
            
            # Generate genotypes for training and test sets (filter to GWAS SNPs)
            X_train, X_test = _generate_train_test_geno(id_train, id_test, filter_snps=True)
            y_train = _generate_train_phe(id_train)
            
            # Initial model fit
            model_instance = Kw_Model(weights=snp_weights)
            model_instance.fit(X_train, y_train)
            
            # Variance component acquisition
            sigma_a2_estimated = None
            sigma_e2_estimated = None
            
            for _p in [os.path.join('results', 'variance_component.log.txt')]:
                if os.path.exists(_p):
                    try:
                        sigma_a2_estimated, sigma_e2_estimated = read_sigma2_from_log(_p)
                        print(f"[Variance components] Read from: {_p}")
                        print(f"[Variance components] sigma_a2 = {sigma_a2_estimated}, sigma_e2 = {sigma_e2_estimated}")
                        break
                    except Exception as e:
                        print(f"Warning: failed to parse '{_p}': {e}")
                        
            if sigma_a2_estimated is None or sigma_e2_estimated is None:
                raise FileNotFoundError(
                    "Unable to locate or parse 'variance_component.log.txt'. "
                    "Checked: " + os.path.join('results', 'variance_component.log.txt')
                )
                
            # Train and predict using the Kw model
            best_model = Kw_Model(
                gamma=args.gamma,
                lambda_=args.lambda_,
                sigma_a2=sigma_a2_estimated,
                sigma_e2=sigma_e2_estimated,
                weights=snp_weights
            )
            best_model.fit(X_train, y_train)
            y_test_pred = best_model.predict(X_test)
            
            # Save the prediction results
            pred_str = y_test_pred.astype(str)
            id_pred = np.column_stack((id_test, pred_str))
            np.savetxt(os.path.join('results', 'y_test_pred.txt'), id_pred, fmt='%s', 
                       delimiter='\t',header='ID\tPrediction', comments='')

            print('End time:', datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"), flush=True)
              
    else:
        # Full pipeline: hyperparameter optimization, model selection, training and prediction (GWKBR pipeline)
        
        # Calculate SNP weights using training set
        print('Start SNP weight calculation:', datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"), flush=True)   
        snp_weights = calculate_snp_weights(id_train)
        
        # Generate genotypes for training and test sets (use GWAS SNPs)
        X_train, X_test = _generate_train_test_geno(id_train, id_test, filter_snps=True)
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

        # Variance component acquisition
        print('Start variance component estimation:', datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"), flush=True)   
        log_candidates = [os.path.join('results', 'variance_component.log.txt')]
        sigma_a2_estimated = None
        sigma_e2_estimated = None
        
        for _p in log_candidates:
            if os.path.exists(_p):
                try:
                    sigma_a2_estimated, sigma_e2_estimated = read_sigma2_from_log(_p)
                    print(f"[Variance components] Read from: {_p}")
                    print(f"[Variance components] sigma_a2 = {sigma_a2_estimated}, sigma_e2 = {sigma_e2_estimated}")
                    break
                except Exception as e:
                    print(f"Warning: failed to parse '{_p}': {e}")
                    
        if sigma_a2_estimated is None or sigma_e2_estimated is None:
            raise FileNotFoundError(
                "Unable to locate or parse 'variance_component.log.txt'. "
                "Checked: " + ", ".join(log_candidates)
            )

        # Hyperparameter optimization
        print('Start hyperparemeter optimization:', datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"), flush=True)   
        if trait_type == 'binary':
            print("Optimizing K model...")
            param_space_K = {
                "gamma": Real(1e-8, 1e-3, prior="log-uniform"),
                "lambda_": Real(0.01, 10, prior="log-uniform")
            }
            best_params_K = cross_validate_and_optimize_binary(
                K_Model,
                param_space_K,
                X_train,
                y_train
            )
            
            print("Optimizing Kw model...")
            param_space_Kw = {
                "gamma": Real(1e-8, 1e-3, prior="log-uniform"),
                "lambda_": Real(0.01, 10, prior="log-uniform")
            }
            best_params_Kw = cross_validate_and_optimize_binary(
                Kw_Model,
                param_space_Kw,
                X_train,
                y_train,
                snp_weights=snp_weights
            )
            
        else:
            print("Optimizing K model...")
            param_space_K = {
                "gamma": Real(1e-8, 1e-3, prior="log-uniform"),
                "lambda_": Real(0.01, 10, prior="log-uniform")
            }
            best_params_K = cross_validate_and_optimize_continuous(
                K_Model,
                param_space_K,
                X_train,
                y_train
            )
            
            print("Optimizing Kw model...")
            param_space_Kw = {
                "gamma": Real(1e-8, 1e-3, prior="log-uniform"),
                "lambda_": Real(0.01, 10, prior="log-uniform")
            }
            best_params_Kw = cross_validate_and_optimize_continuous(
                Kw_Model,
                param_space_Kw,
                X_train,
                y_train,
                snp_weights=snp_weights
            )

        # Select the best model
        print('Start model selection:', datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"), flush=True)   
        mean_score_K = cross_validate_model_selection(
            K_Model,
            best_params_K,
            X_train,
            y_train,
            id_train,
            "K"
        )
        mean_score_Kw = cross_validate_model_selection(
            Kw_Model,
            best_params_Kw,
            X_train,
            y_train,
            id_train,
            "Kw",
            snp_weights=snp_weights
        )

        if mean_score_K > mean_score_Kw:
            print(f"K model is better with mean score: {mean_score_K}")
            best_model_class = K_Model
            best_params = best_params_K
            
            with open(os.path.join('results', 'best_params.txt'), 'w') as w1:
                w1.write('kernel' + '\t' + 'unweighted-kernel' + '\n')
                for k, v in best_params_K.items():
                    w1.write(str(k) + '\t' + str(v) + '\n')
        else:
            print(f"Kw model is better with mean score: {mean_score_Kw}")
            best_model_class = Kw_Model
            best_params = best_params_Kw
            
            with open(os.path.join('results', 'best_params.txt'), 'w') as w1:
                w1.write('kernel' + '\t' + 'weighted-kernel' + '\n')
                for k, v in best_params_Kw.items():
                    w1.write(str(k) + '\t' + str(v) + '\n')

        # Train and predict using the best model
        print('Start model training and prediction:', datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"), flush=True)           
        if best_model_class == Kw_Model:
            best_model = best_model_class(
                **best_params,
                sigma_a2=sigma_a2_estimated,
                sigma_e2=sigma_e2_estimated,
                weights=snp_weights
            )
        else:
            best_model = best_model_class(
                **best_params,
                sigma_a2=sigma_a2_estimated,
                sigma_e2=sigma_e2_estimated
            )

        best_model.fit(X_train, y_train)
        y_test_pred = best_model.predict(X_test)

        # Save the prediction results
        pred_str = y_test_pred.astype(str)
        id_pred = np.column_stack((id_test, pred_str))
        np.savetxt(
            os.path.join('results', 'y_test_pred.txt'), 
            id_pred, 
            fmt='%s', 
            delimiter='\t',
            header='ID\tPrediction', 
            comments=''
        )

        # Output end time
        print('End time:', datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()