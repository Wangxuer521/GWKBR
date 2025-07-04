# -*- coding: utf-8 -*-
import glob
import datetime
import os, shutil
import subprocess
import numpy as np
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

# 自定义逆Gamma采样函数
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

    for i in range(1, n_iter):
        # 更新 sigma_a2
        Sigma_post = np.linalg.inv((1 / sigma_e2_samples[i-1]) * K_sq + lambda_ * (1 / sigma_a2_samples[i-1]) * G)
        mu_post = np.dot(Sigma_post, (1 / sigma_e2_samples[i-1]) * np.dot(K_T, y))
        y_pred = np.dot(K, mu_post)
        residual = y - y_pred
        alpha_post = alpha + n / 2
        beta_post = beta + 0.5 * np.dot(residual, residual)
        sigma_a2_samples[i] = numba_invgamma(alpha_post, beta_post)

        # 更新 sigma_e2
        Sigma_post = np.linalg.inv((1 / sigma_e2_samples[i-1]) * K_sq + lambda_ * (1 / sigma_a2_samples[i]) * G)
        mu_post = np.dot(Sigma_post, (1 / sigma_e2_samples[i-1]) * np.dot(K_T, y))
        y_pred = np.dot(K, mu_post)
        residual = y - y_pred
        beta_post = beta + 0.5 * np.dot(residual, residual)
        sigma_e2_samples[i] = numba_invgamma(alpha_post, beta_post)

        if i % 10000 == 0:  # 每10000次迭代打印一次进度
            print(f"Iteration {i}/{n_iter} completed")

    thinned_sigma_a2_samples = sigma_a2_samples[burn_in::thinning_interval]
    thinned_sigma_e2_samples = sigma_e2_samples[burn_in::thinning_interval]

    return thinned_sigma_a2_samples, thinned_sigma_e2_samples, sigma_a2_samples, sigma_e2_samples

# 定义KBR模型（无加权）
class KBR_Model(BaseEstimator, RegressorMixin):
    def __init__(self, gamma=0.00006, lambda_=1.0, sigma_a2=2000, sigma_e2=700, G=None):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.sigma_a2 = sigma_a2
        self.sigma_e2 = sigma_e2
        self.G = G

    def _gaussian_kernel(self, X1, X2, gamma):
        pairwise_sq_dists = -2 * np.dot(X1, X2.T) + np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1)
        return np.exp(-gamma * pairwise_sq_dists)

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
        X = X[:, :self.n_features]
        K = self._gaussian_kernel(X, X, self.gamma)
        self.K = K
        self.X_fit_ = X
        self.y = y

        self._compute_posterior()
        return self

    def _compute_posterior(self):
        self.Sigma_post = np.linalg.inv((1 / self.sigma_e2) * np.dot(self.K.T, self.K) + self.lambda_ * (1 / self.sigma_a2) * self.G)
        self.mu_post = np.dot(self.Sigma_post, (1 / self.sigma_e2) * np.dot(self.K.T, self.y))

    def predict(self, X):
        X = X[:, :self.n_features]
        K_test = self._gaussian_kernel(X, self.X_fit_[:, :self.n_features], self.gamma)
        return np.dot(K_test, self.mu_post)

# 定义GWKBR模型（带权重）
class GWKBR_Model(BaseEstimator, RegressorMixin):
    def __init__(self, gamma=0.00006, lambda_=1.0, sigma_a2=2000, sigma_e2=700, G=None, weights=None):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.sigma_a2 = sigma_a2
        self.sigma_e2 = sigma_e2
        self.G = G
        self.weights = weights

    def _weighted_gaussian_kernel(self, X1, X2, gamma, weights):
        W = diags(weights, format='csr')
        pairwise_sq_dists = np.sum((X1 @ W) * X1, axis=1)[:, np.newaxis] + np.sum((X2 @ W) * X2, axis=1) - 2 * np.dot(X1 @ W, X2.T)
        return np.exp(-gamma * pairwise_sq_dists)

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
        X = X[:, :self.n_features]
        K = self._weighted_gaussian_kernel(X, X, self.gamma, self.weights)
        self.K = K
        self.X_fit_ = X
        self.y = y

        self._compute_posterior()
        return self

    def _compute_posterior(self):
        self.Sigma_post = np.linalg.inv((1 / self.sigma_e2) * np.dot(self.K.T, self.K) + self.lambda_ * (1 / self.sigma_a2) * self.G)
        self.mu_post = np.dot(self.Sigma_post, (1 / self.sigma_e2) * np.dot(self.K.T, self.y))

    def predict(self, X):
        X = X[:, :self.n_features]
        K_test = self._weighted_gaussian_kernel(X, self.X_fit_[:, :self.n_features], self.gamma, self.weights)
        return np.dot(K_test, self.mu_post)

# 定义Gibbs抽样函数
def gibbs_sampling(n_iter, burn_in, alpha, beta, model_instance, thinning_interval=10):
    y = model_instance.y
    K = model_instance.K
    G = model_instance.G
    initial_sigma_e2 = 1
    initial_sigma_a2 = 1
    lambda_ = model_instance.lambda_

    thinned_sigma_a2_samples, thinned_sigma_e2_samples, sigma_a2_samples, sigma_e2_samples = gibbs_sampling_core(
        n_iter, burn_in, alpha, beta, y, K, G, initial_sigma_e2, initial_sigma_a2, lambda_, thinning_interval)

    print("Number of Gibbs sampling thinned_sigma_a2_samples:", len(thinned_sigma_a2_samples))
    print("Number of Gibbs sampling thinned_sigma_e2_samples:", len(thinned_sigma_e2_samples))

    return thinned_sigma_a2_samples, thinned_sigma_e2_samples, sigma_a2_samples, sigma_e2_samples

# 重新计算SNP权重函数
def recalculate_snp_weights(X_train_1, id_train_1):
    np.savetxt('id_train_temp.txt', id_train_1, fmt='%d')
    
    # to_ref_id_list.py
    try:
        with open('id_train_temp.txt', 'r') as r1, \
            open('../G/Maize_ywk.fam', 'r') as r2, \
            open('xin_id', 'r') as r3, \
            open('ref_id.list', 'w') as w:

            number_id = {}
            for i in r3:
                f = i.split()
                number_id[f[1]] = f[0]      #{ID编号: ID号}

            dic = {}
            for i in r2:
                f = i.split()
                dic[f[1]] = f[0]        #{ID号: fam第一列}
                
            for i in r1:
                f = i.split()
                if f[0] in number_id:
                    w.write(str(dic[number_id[f[0]]]) + '\t' + number_id[f[0]] + '\n')
    except Exception as e:
        print(f"Error when generating ref_id.list: {e}")
        return None
        
    # gwas.sh
    try:
        plink_exe = "plink"
        
        subprocess.run(
            [plink_exe,
             "--bfile", "../G/Maize_ywk",
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

        subprocess.run(
            [python_path, "change_-9_to_0.py"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )

        if os.path.exists("ref_genotypes1.fam"):
            os.replace("ref_genotypes1.fam", "ref_genotypes.fam")

        #run GEMMA
        gemma_path = "/public/home/06025/WORK/Software/mambaforge-pypy3/envs/gemma/bin/gemma"
        subprocess.run(
            [gemma_path,
             "-bfile", "ref_genotypes",
             "-miss", "1.0",
             "-gk", "1",
             "-o", "kinship"
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        kinship_file = os.path.join("output", "kinship.cXX.txt")
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
        
    # to_gwas_result.py   
    try:
        assoc_path = os.path.join("output", "gemma_gwas.assoc.txt")
        bim_path   = "../G/Maize_ywk.bim"
        
        with open(assoc_path, "r") as r1, open(bim_path, "r") as r2, \
             open("gwas.result", "w") as w1, open("final_snps", "w") as w2:
            
            effect_snps = set()
            snp_beta_se = {}

            header = r1.readline().split()
            if header[7] == 'beta' and header[8] == 'se':
                for i in r1:
                    f = i.split()            
                    effect_snps.add(f[1])
                    snp_beta_se[f[1]] = f[7]+'_'+f[8]       #snp_effect={snp: beta_se}
            else:
                print('Please check the format of gemma_gwas.assoc.txt!')
                exit()
           
            final_snps = []
            for i in r2:
                f = i.split()
                if f[1] in effect_snps:
                    final_snps.append(f[1])
                    w2.write(f[1] + '\n')

            for snp in final_snps:
                if snp in snp_beta_se:            
                    beta = snp_beta_se[snp].split('_')[0]
                    se = snp_beta_se[snp].split('_')[1]
                    w1.write(beta + '\t' + se + '\n')
                else:
                    print(f'Error! {f[0]} is not in the result file of GWAS!')
    
    except Exception as e:
        print(f"Error when generating gwas.result and final_snps: {e}")
        return None    
    
    #cal_snp_weight.py
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
            avg_bf = np.exp(avg_log_bf)             # 几何平均
            
            avg_pp = PI * avg_bf / (PI * avg_bf + (1 - PI))
            
            avg[i, 2] = avg_bf
            avg[i, 1] = avg_pp

        avg[:, 0] = avg[:, 1] * (len(bf) / np.sum(avg[:, 1]))    # 放大后的权重

        np.savetxt("snp_weight.out", avg)
      
    except Exception as e:
        print(f"Error when calculating SNP weights: {e}")
        return None    
        
    try:
        subprocess.run(
            ['plink', '--bfile', 'ref_genotypes', '--extract', 'final_snps', '--recodeA'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
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

# 定义超参数优化函数（数量性状）
def cross_validate_and_optimize_continuous(model_class, param_space, X_train, y_train, snp_weights=None):
    if snp_weights is not None:
        model = BayesSearchCV(
            model_class(weights=snp_weights, sigma_a2=sigma_a2_estimated_GWKBR, sigma_e2=sigma_e2_estimated_GWKBR),
            param_space, n_iter=50, cv=5, n_jobs=-1, random_state=42, scoring=pearson_scorer
        )
    else:
        model = BayesSearchCV(
            model_class(sigma_a2=sigma_a2_estimated_KBR, sigma_e2=sigma_e2_estimated_KBR),
            param_space, n_iter=50, cv=5, n_jobs=-1, random_state=42, scoring=pearson_scorer
        )
    
    model.fit(X_train, y_train)
    return model.best_params_

# 定义超参数优化函数（分类性状）
def cross_validate_and_optimize_binary(model_class, param_space, X_train, y_train, snp_weights=None):
    if snp_weights is not None:
        model = BayesSearchCV(
            model_class(weights=snp_weights, sigma_a2=sigma_a2_estimated_GWKBR, sigma_e2=sigma_e2_estimated_GWKBR),
            param_space, n_iter=1, cv=5, n_jobs=-1, random_state=42, scoring=auroc_scorer
        )
    else:
        model = BayesSearchCV(
            model_class(sigma_a2=sigma_a2_estimated_KBR, sigma_e2=sigma_e2_estimated_KBR),
            param_space, n_iter=1, cv=5, n_jobs=-1, random_state=42, scoring=auroc_scorer
        )
    
    model.fit(X_train, y_train)
    return model.best_params_

# 定义Pearson相关系数得分
def pearson_corr_coef(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

# 定义AUROC得分
def auroc_score(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

X_train = np.loadtxt('X1.txt')
y_train = np.loadtxt('y.txt')
id_train = np.loadtxt('rel_id')
snp_weights = np.loadtxt('../G/snp_weight.out')[:, 0]
X_test = np.loadtxt('X1_test.txt')
all_snps = np.loadtxt('plink_raw_snps', dtype=str)
python_path = '/public/home/06025/WORK/Software/anaconda3_202205/bin/python'

# 判断 y.txt 为分类变量（0/1）还是连续变量
if np.all(np.isin(y_train, [0, 1])):
    trait_type = 'binary'
    auroc_scorer = make_scorer(auroc_score, greater_is_better=True)
    print('AUROC Scorer selected.')
else:
    trait_type = 'continuous'
    pearson_scorer = make_scorer(pearson_corr_coef, greater_is_better=True)
    print('Pearson Scorer selected.')

# 输出开始时间
print('Start time:', datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"), flush=True)

# Gibbs抽样参数
n_iter = 50000
burn_in = 20000
alpha = 3 / 2
beta = 3 * 1.0 / 2
thinning_interval = 10

# 对 KBR 进行Gibbs抽样
model_instance_KBR = KBR_Model()
model_instance_KBR.fit(X_train, y_train)
print("Start Gibbs Sampling for KBR!")
thinned_sigma_a2_samples_KBR, thinned_sigma_e2_samples_KBR, raw_sigma_a2_samples_KBR, raw_sigma_e2_samples_KBR = gibbs_sampling(n_iter, burn_in, alpha, beta, model_instance_KBR, thinning_interval)
sigma_a2_estimated_KBR = np.mean(thinned_sigma_a2_samples_KBR)
sigma_e2_estimated_KBR = np.mean(thinned_sigma_e2_samples_KBR)

# 对 GWKBR 进行Gibbs抽样
model_instance_GWKBR = GWKBR_Model(weights=snp_weights)
model_instance_GWKBR.fit(X_train, y_train)
print("Start Gibbs Sampling for GWKBR!")
thinned_sigma_a2_samples_GWKBR, thinned_sigma_e2_samples_GWKBR, raw_sigma_a2_samples_GWKBR, raw_sigma_e2_samples_GWKBR = gibbs_sampling(n_iter, burn_in, alpha, beta, model_instance_GWKBR, thinning_interval)
sigma_a2_estimated_GWKBR = np.mean(thinned_sigma_a2_samples_GWKBR)
sigma_e2_estimated_GWKBR = np.mean(thinned_sigma_e2_samples_GWKBR)

# 超参数优化
if trait_type == 'binary':
    print("Optimizing KBR model...")
    param_space_KBR = {"gamma": Real(1e-7, 1e-4, prior="log-uniform"), "lambda_": Real(0.1, 3, prior="log-uniform")}
    best_params_KBR = cross_validate_and_optimize_binary(KBR_Model, param_space_KBR, X_train, y_train)
           
    print("Optimizing GWKBR model...")
    param_space_GWKBR = {"gamma": Real(1e-7, 1e-4, prior="log-uniform"), "lambda_": Real(0.1, 3, prior="log-uniform")}
    best_params_GWKBR = cross_validate_and_optimize_binary(GWKBR_Model, param_space_GWKBR, X_train, y_train, snp_weights=snp_weights)
else: 
    print("Optimizing KBR model...")
    param_space_KBR = {"gamma": Real(1e-7, 1e-4, prior="log-uniform"), "lambda_": Real(0.1, 3, prior="log-uniform")}
    best_params_KBR = cross_validate_and_optimize_continuous(KBR_Model, param_space_KBR, X_train, y_train)
           
    print("Optimizing GWKBR model...")
    param_space_GWKBR = {"gamma": Real(1e-7, 1e-4, prior="log-uniform"), "lambda_": Real(0.1, 3, prior="log-uniform")}
    best_params_GWKBR = cross_validate_and_optimize_continuous(GWKBR_Model, param_space_GWKBR, X_train, y_train, snp_weights=snp_weights)

# 在交叉验证中比较 KBR 和 GWKBR 的表现
def cross_validate_model_selection(model_class, best_params, X_train, y_train, id_train, model_name, snp_weights=None):
    rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)
    auroc_scores = []
    pearson_scores = []

    for fold, (train_idx, val_idx) in enumerate(rkf.split(X_train)):
        X_train_1, X_val_1 = X_train[train_idx], X_train[val_idx]
        y_train_1, y_val_1 = y_train[train_idx], y_train[val_idx]
        id_train_1, id_val_1 = id_train[train_idx], id_train[val_idx]
        
        # 重新计算SNP权重
        if model_name == "GWKBR":
            snp_weights_cv = recalculate_snp_weights(X_train_1, id_train_1)
            
            # 获取位置并挑选相应位置的列
            final_snps = np.loadtxt('final_snps', dtype=str)
            snp_indices = []
            for snp in final_snps:
                matches = np.where(np.array(all_snps) == snp)[0]
                if matches.size > 0:
                    snp_indices.append(matches[0])
                else:
                    print(f"Warning: SNP {snp} not found in all_snps.")
            
            X_train_1 = X_train_1[:, snp_indices]
            X_val_1 = X_val_1[:, snp_indices]
            
            # 删除文件
            files_to_remove = ['id_train_temp.txt', 'ref_id.list', 'ref_genotypes.bed', 'ref_genotypes.bim', 'ref_genotypes.fam', 'gwas.result', 'final_snps', 'snp_weight.out', 'plink.raw', 'plink.log', 'plink.nosex']
            for file in files_to_remove:
                try:
                    os.remove(file)
                except FileNotFoundError:
                    print(f"Warning: {file} not found.")
                    
            # 删除output文件夹
            output_folder = 'output'
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            else:
                print(f"{output_folder} folder not found.")
            
        else:
            snp_weights_cv = None
        
        # 训练模型
        if model_name == "GWKBR":
            model = model_class(**best_params, sigma_a2=sigma_a2_estimated_GWKBR, sigma_e2=sigma_e2_estimated_GWKBR, weights=snp_weights_cv)
        else:
            model = model_class(**best_params, sigma_a2=sigma_a2_estimated_KBR, sigma_e2=sigma_e2_estimated_KBR)        

        model.fit(X_train_1, y_train_1)
        y_val_pred = model.predict(X_val_1)
        
        if trait_type == 'binary':
            auroc_scores.append(auroc_score(y_val_1, y_val_pred))
        else:
            pearson_scores.append(pearson_corr_coef(y_val_1, y_val_pred))
    
    if trait_type == 'binary':
        print(f"{model_name} - All AUROC scores:", auroc_scores)
        return np.mean(auroc_scores)
    else:
        print(f"{model_name} - All Pearson scores:", pearson_scores)
        return np.mean(pearson_scores)

mean_score_KBR = cross_validate_model_selection(KBR_Model, best_params_KBR, X_train, y_train, id_train, "KBR")
mean_score_GWKBR = cross_validate_model_selection(GWKBR_Model, best_params_GWKBR, X_train, y_train, id_train, "GWKBR", snp_weights=snp_weights)

# 选择最佳模型
if mean_score_KBR > mean_score_GWKBR:
    print(f"KBR is better with mean score: {mean_score_KBR}")
    best_model_class = KBR_Model
    best_params = best_params_KBR
    with open('best_params.txt', 'w') as w1:
        w1.write('kernel' + '\t' + 'unweighted-kernel' + '\n')
        for k, v in best_params_KBR.items():
            w1.write(str(k) + '\t' + str(v) + '\n')
    with open('variance_components.txt', 'w') as w2:
        w2.write('sigma_a2' + '\t' + str(sigma_a2_estimated_KBR) + '\n' + 'sigma_e2' + '\t' + str(sigma_e2_estimated_KBR) + '\n')
else:
    print(f"GWKBR is better with mean score: {mean_score_GWKBR}")
    best_model_class = GWKBR_Model
    best_params = best_params_GWKBR
    with open('best_params.txt', 'w') as w1:
        w1.write('kernel' + '\t' + 'weighted-kernel' + '\n')
        for k, v in best_params_GWKBR.items():
            w1.write(str(k) + '\t' + str(v) + '\n')
    with open('variance_components.txt', 'w') as w2:
        w2.write('sigma_a2' + '\t' + str(sigma_a2_estimated_GWKBR) + '\n' + 'sigma_e2' + '\t' + str(sigma_e2_estimated_GWKBR) + '\n')

# 使用最佳模型训练并预测
if best_model_class == GWKBR_Model:
    best_model = best_model_class(**best_params, sigma_a2=sigma_a2_estimated_GWKBR, sigma_e2=sigma_e2_estimated_GWKBR, weights=snp_weights)
else:
    best_model = best_model_class(**best_params, sigma_a2=sigma_a2_estimated_KBR, sigma_e2=sigma_e2_estimated_KBR)
best_model.fit(X_train, y_train)
y_test_pred = best_model.predict(X_test)

# 保存预测结果
np.savetxt('y_test_pred.txt', y_test_pred)
print('End time:', datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"), flush=True)
