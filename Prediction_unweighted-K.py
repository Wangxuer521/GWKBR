# -*- coding: utf-8 -*-
import os
import glob
import datetime
import subprocess
import numpy as np
import pandas as pd
from numba import njit
from skopt.space import Real
from scipy.linalg import eigh
from skopt import BayesSearchCV
from scipy.stats import pearsonr
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

# Define the unweighted GWKBR model (KBR)
class BayesianCombinedKernelRidge(BaseEstimator, RegressorMixin):
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
        nani_A = dosageA.shape[0]
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
        K_test = self._gaussian_kernel(
            X, 
            self.X_fit_[:, :self.n_features], 
            self.gamma
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

# Generate genotypes for training and test sets
def _generate_train_test_geno(id_train, id_test):
    subprocess.run(
        ["plink",
         "--bfile", "./example_data/all_genotypes",
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
id_train = np.loadtxt('./example_data/train_id.txt', dtype=str)
id_test = np.loadtxt('./example_data/test_id.txt', dtype=str)

# Generate genotypes for training and test sets
X_train, X_test = _generate_train_test_geno(id_train, id_test)
y_train = _generate_train_phe(id_train)

model_instance = BayesianCombinedKernelRidge()
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

print(
    'sigma_a2\t' + str(sigma_a2_estimated) + '\n' + 
    'sigma_e2\t' + str(sigma_e2_estimated)
)

# Train and predict using the best model
best_model = BayesianCombinedKernelRidge(
    gamma=5.87e-5, 
    lambda_=0.1, 
    sigma_a2=sigma_a2_estimated, 
    sigma_e2=sigma_e2_estimated
)
best_model.fit(X_train, y_train)
y_test_pred = best_model.predict(X_test)

# Save the prediction results
pred_str = y_test_pred.astype(str)
id_pred = np.column_stack((id_test, pred_str))
np.savetxt('y_test_pred.txt', id_pred, fmt='%s %s', header='ID\tPrediction', comments='')
print('End time:', datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"), flush=True)