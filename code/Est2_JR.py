import numpy as np
from parfor import parfor
from scipy.stats import wishart, invwishart, multivariate_normal
from itertools import combinations

# ---------- (HMC) Jeffreys ----------
def pi_Jeffreys(returns, mu, Sigma):
        """
        %pdf of W_1(3, 1) at Sigma
        Sigma can be a scalar or a 1x1 array
        """
        (n,k) = returns.shape
        mu_sample = np.mean(returns, axis=0)
        Sigma_sample = np.cov(returns, rowvar=False)
        return invwishart.pdf(Sigma, df=n, scale=Sigma_sample * (n-1) + np.outer(mu_sample-mu, mu_sample-mu) * n)

# ---------- (HMC) Right ----------
def minors_product(Sigma):
    k = Sigma.shape[0]
    minors_product = 1.0
    for i in range(1, k):
        minor = Sigma[:i, :i]
        minors_product *= np.linalg.det(minor)
        #print(np.linalg.det(minor))
    return minors_product
# Placeholder target; replace freely with your own:

def pi_Right(returns, mu, Sigma):
        """
        %pdf of W_1(3, 1) at Sigma
        Sigma can be a scalar or a 1x1 array
        """
        (n,k) = returns.shape
        mu_sample = np.mean(returns, axis=0)
        Sigma_sample = np.cov(returns, rowvar=False)
        #if np.isfinite(Sigma).all() == False:
        #print(Sigma)
        #print("Mean", (Sigma_sample * (n-1) + np.outer(mu_sample, mu_sample) * n)/(n-k-1))
        return invwishart.pdf(Sigma, df=n, scale=Sigma_sample * (n-1) + np.outer(mu_sample-mu, mu_sample-mu) * n) * (np.linalg.det(Sigma) ** ((k-1)/2) /minors_product(Sigma))

# ---------- (MHMC) Right ----------
def minors_product_radio(Sigma,Sigma_cand):
    k = Sigma.shape[0]
    minors_product = []
    for i in range(1, k+1):
        minor = Sigma[:i, :i]
        det = np.linalg.det(minor)
        minor_cand = Sigma_cand[:i, :i]
        det_cand = np.linalg.det(minor_cand)
        minors_product.append((det/det_cand))
    return np.prod(minors_product)

def minors_product_radio_log(Sigma, Sigma_cand):
    k = Sigma.shape[0]
    log_sum = 0.0

    for i in range(1, k+1):
        minor = Sigma[:i, :i]
        det = np.linalg.det(minor)
        minor_cand = Sigma_cand[:i, :i]
        det_cand = np.linalg.det(minor_cand)
        if det <= 0 or det_cand <= 0:
            return 0
        log_ratio = np.log(det) - np.log(det_cand)
        log_sum += log_ratio
    return np.exp(log_sum)

# ---------- (MHMC) Scale-Right ----------
def scale_minors_product_radio(Sigma,Sigma_cand):
    k = Sigma.shape[0]
    #print(k)
    denominator = 1.0
    for i in range(1, k):  # i = 1 to k-1
        #print(i)
        comb = list(combinations(range(k), i))
        if len(comb) > 3000:
            geom_mean = 1
        else:
            @parfor(range(len(comb)))
            def det_sub(j):
                I = list(comb[j])
                submatrix = Sigma[np.ix_(I, I)]
                det_sq = np.linalg.det(submatrix)
                submatrix_cand = Sigma_cand[np.ix_(I, I)]
                det_sq_cand = np.linalg.det(submatrix_cand)
                return det_sq/det_sq_cand
            subset_dets = det_sub
            geom_mean = np.prod(subset_dets)**(1.0 / len(subset_dets))
        denominator *= geom_mean
    return denominator

# ---------- (MHMC) Rotate-Right ----------
rng = np.random.default_rng()
def normalGL(d):
    while True:
        A = np.random.randn(d, d)
        if np.linalg.det(A) != 0:
            return A
def GramSchmidt(A):
    k = A.shape[1]
    Gmm = A.copy()
    for i in range(k):
        ip = np.sum(Gmm[:, i:i+1] * Gmm[:, :i], axis=0, keepdims=True)
        sgm = Gmm[:, i] - np.sum(ip * Gmm[:, :i], axis=1)
        Gmm[:, i] = sgm / np.linalg.norm(sgm)
    return Gmm
def sample_orthogonal_matrix(d):
    A = normalGL(d)
    return GramSchmidt(A)
def log_minor_sum(GSGt, k):
    total = 0
    for i in range(1, k):  # i = 1 to k-1
        submatrix = GSGt[:i, :i]
        total += np.log(np.linalg.det(submatrix))
    return total
def rotate_minors_product_radio(Sigma, Sigma_cand, num_samples=1000):
    k = Sigma.shape[0]
    @parfor(range(num_samples))
    def integral(i):
        Gamma = sample_orthogonal_matrix(k)
        GSGt = Gamma @ Sigma @ Gamma.T
        GSGt_cand = Gamma @ Sigma_cand @ Gamma.T
        return log_minor_sum(GSGt, k) - log_minor_sum(GSGt_cand, k)
    return np.exp(sum(integral) / num_samples)
