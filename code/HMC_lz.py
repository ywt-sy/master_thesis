import numpy as np
from parfor import parfor
from scipy.linalg import expm
from scipy.stats import wishart, invwishart, multivariate_normal
from data_simulation import generate_simulation_data
from Est2_JR import pi_Jeffreys, pi_Right
import matplotlib.pyplot as plt
import time

# ---------- Sigma into Euclidean Space----------
def sigma_to_lz(Sigma):
    """
    Convert a positive definite matrix Sigma into an unconstrained
    parameter vector:
    (l11, z21, z31, ..., zk1, l22, z32, ..., z_k2, ..., lkk)

    where:
        Sigma = L L^T
        L_ii = exp(l_ii)
        L_ij = z_ij for i > j
    """
    Sigma = np.asarray(Sigma)
    k = Sigma.shape[0]

    # Cholesky decomposition (lower triangular)
    L = np.linalg.cholesky(Sigma)

    params = []

    for j in range(k):
        # diagonal: log(L_jj)
        params.append(np.log(L[j, j]))

        # below-diagonal entries in column j
        for i in range(j + 1, k):
            params.append(L[i, j])

    return np.array(params)

def lz_to_sigma(params, k):
    """
    Convert unconstrained parameter vector back to Sigma.

    Parameters
    ----------
    params : array-like, shape (k*(k+1)//2,)
        (l11, z21, z31, ..., zk1, l22, z32, ..., lkk)
    k : int
        Dimension of Sigma

    Returns
    -------
    Sigma : ndarray, shape (k, k)
        Positive definite covariance matrix
    """
    params = np.asarray(params)

    L = np.zeros((k, k))
    idx = 0

    for j in range(k):
        # diagonal
        L[j, j] = np.exp(params[idx])
        idx += 1

        # below diagonal
        for i in range(j + 1, k):
            L[i, j] = params[idx]
            idx += 1

    Sigma = L @ L.T
    return Sigma

def lz_to_L(params, k):
    """
    params ordering:
      (l11, z21, z31, ..., zk1, l22, z32, ..., z_k2, ..., lkk)
    where L_ii = exp(l_ii), L_ij = z_ij (i > j).
    """
    params = np.asarray(params, dtype=float)
    L = np.zeros((k, k), dtype=float)

    idx = 0
    for j in range(k):
        L[j, j] = np.exp(params[idx])  # diagonal
        idx += 1
        for i in range(j + 1, k):      # below diagonal in column j
            L[i, j] = params[idx]
            idx += 1
    return L

def jacobian_Sigma_lz(params, k):
    """
    Build Jacobian J = d vec(Sigma) / d params
    
    vec(Sigma) is row-major flattening: Sigma[0,0], Sigma[0,1], ..., Sigma[k-1,k-1]
    
    Returns
    -------
    J : ndarray, shape (k*k, k*(k+1)//2)
    """
    params = np.asarray(params, dtype=float)
    n_params = k * (k + 1) // 2
    if params.size != n_params:
        raise ValueError(f"params length must be {n_params} for k={k}, got {params.size}")

    L = lz_to_L(params, k)
    Sigma = L @ L.T

    J = np.zeros((k * k, n_params), dtype=float)

    # helper: map (i,j) -> row index in vec(Sigma)
    def ridx(i, j):
        return i * k + j

    p = 0
    for j in range(k):
        # --- parameter is l_jj (diagonal log) ---
        a = j
        L_aa = L[a, a]

        # dSigma/ d l_aa
        J[ridx(a, a), p] = 2.0 * (L_aa ** 2)
        for i in range(a + 1, k):
            # (i > a, j = a)
            J[ridx(i, a), p] = L_aa * L[i, a]
            # (i = a, j > a)
            J[ridx(a, i), p] = L_aa * L[i, a]
        p += 1

        # --- parameters are z_{i,j} for i=j+1..k-1 ---
        b = j
        for a in range(j + 1, k):
            # parameter z_ab = L[a,b]
            # dSigma_ij/dz_ab is nonzero only if i=a or j=a
            # Case i=a: dSigma_{a, j}/dz_ab = L[j, b] for j>=b
            for jj in range(b, k):
                J[ridx(a, jj), p] = L[jj, b]
            # Case j=a: dSigma_{i, a}/dz_ab = L[i, b] for i>=b
            for ii in range(b, k):
                J[ridx(ii, a), p] = L[ii, b]
            p += 1

    return J

def _infer_k_from_nparams(n_params: int) -> int:
    # solve k(k+1)/2 = n_params
    k = int((np.sqrt(8*n_params + 1) - 1) / 2)
    if k * (k + 1) // 2 != n_params:
        raise ValueError(f"invalid n_params={n_params}, cannot infer k")
    return k

def _diag_param_indices(k: int):
    # param order in your lz_to_L: for each column j: [l_jj, z_{j+1,j}, ..., z_{k-1,j}]
    idxs = []
    idx = 0
    for j in range(k):
        idxs.append(idx)      # l_jj
        idx += 1 + (k - 1 - j)
    return idxs

def log_jacobian_lz(params: np.ndarray, k: int) -> float:
    # |dSigma/dparams| = 2^k * Π_j L_jj^{k+1-j} with L_jj = exp(l_jj)
    # => logJac = k log 2 + Σ_j (k+1-j) * l_jj  (j is 0-indexed)
    diag_idxs = _diag_param_indices(k)
    l_diag = params[diag_idxs]
    weights = np.array([k + 1 - j for j in range(k)], dtype=float)
    return float(np.dot(weights, l_diag))

def grad_theta(G: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Map Euclidean gradient wrt Sigma (matrix) to gradient wrt params=(l,z),
    including the Jacobian correction for the (l,z)->Sigma transform.
    """
    params = np.asarray(params, dtype=float)
    k = _infer_k_from_nparams(params.size)

    # ensure symmetry
    G = 0.5 * (G + G.T)

    # d vec(Sigma) / d params
    J = jacobian_Sigma_lz(params, k)

    # vec is row-major to match your Jacobian builder
    g_sigma_vec = G.reshape(-1, order="C")
    grad = J.T @ g_sigma_vec

    # add grad of logJac: only diagonal l_jj entries get (k+1-j)
    jac_corr = np.zeros_like(params)
    diag_idxs = _diag_param_indices(k)
    for j, idx in enumerate(diag_idxs):
        jac_corr[idx] = (k + 1 - j)

    return grad + jac_corr

def grad_lz(mu, Sigma, returns, method):
    """
    Returns ∇_q [log π(Σ)] (Euclidean gradient on Sym^+(d)).
    """
    (n,k) = returns.shape
    if method == "Jeffreys":
        pi = pi_Jeffreys
    
    if method == "Right":
        pi = pi_Right

    h = 1e-6
    para = sigma_to_lz(Sigma)
    G_2 = np.zeros(para.shape, dtype=float)
    for i in range(para.size):
        E = np.zeros_like(para)
        E[i] = 1.0
        S_1 = lz_to_sigma(para + h*E,k)
        f_plus = np.log(pi(returns, mu, S_1))
        S_2 = lz_to_sigma(para - h*E,k)
        f_minus = np.log(pi(returns, mu, S_2))
        deriv = (f_plus - f_minus) / (2*h)
        G_2[i] = deriv
    
    jac_corr = np.zeros(para.shape, dtype=float)
    j = 1
    for idx in diag_param_indices(k):
        # diagonal parameter l_jj
        jac_corr[idx] = k + 1 - j   # i = j+1 in math indexing
        j = j+1
    
    return G_2 + jac_corr
# ---------- energy ----------
def energy_lz(Sigma, p, log_pi_sigma):
    # target in (l,z): log π(Sigma(lz)) + logJac(lz)
    k = Sigma.shape[0]
    params = sigma_to_lz(Sigma)
    logJac = log_jacobian_lz(params, k)
    invM = inv_mass_vector(k)
    return float(-(log_pi_sigma + logJac) + 0.5 * np.sum((p**2) * invM))

# ---------- Euclidean gradient ----------
def grad(mu, Sigma, returns, method):
    """
    Returns ∇_Σ [ log π(Σ) + (d+1) log|Σ| ] (Euclidean gradient on Sym^+(d)).
    """
    # central differences on the upper triangle
    (n,k) = returns.shape
    mu_sample = np.mean(returns, axis=0)
    Sigma_sample = np.cov(returns, rowvar=False)
    S_star = Sigma_sample * (n-1) + np.outer(mu_sample-mu, mu_sample-mu) * n
    Sigma_inv = np.linalg.inv(Sigma)
    #(Jeffreys)
    if method == "Jeffreys":
        G_1 = 0.5 * (Sigma_inv @ S_star @ Sigma_inv - (n + k + 1) * Sigma_inv)

    if method == "Right":
        S = np.zeros_like(Sigma, dtype=float)
        for i in range(1, k+1):
            block = Sigma[:i, :i]
            block_inv = np.linalg.inv(block)
            S[:i, :i] += block_inv  # embed into top-left
        G_1 = 0.5 * (Sigma_inv @ S_star @ Sigma_inv - n * Sigma_inv) - S

    """
    #(Eular)
    if method == "Jeffreys":
        pi = pi_Jeffreys
    h = 1e-12
    G_2 = np.zeros(Sigma.shape, dtype=float)
    for i in range(k):
        for j in range(k):
            E = np.zeros_like(Sigma)
            E[i, j] = 1.0
            f_plus = np.log(pi(returns, mu, Sigma + h*E))
            f_minus = np.log(pi(returns, mu, Sigma - h*E))
            deriv = (f_plus - f_minus) / (2*h)
            G_2[i, j] = deriv
    G_2 = 0.5 * (G_2 + G_2.T)
    """
    return G_1

# ---------- Normal ----------
def diag_param_indices(k: int) -> np.ndarray:
    """
    params ordering: (l11, z21, z31, ..., lk1, l22, z32, ..., lkk)
    Returns indices of l_jj entries in the params vector.
    """
    idxs = []
    idx = 0
    for j in range(k):
        idxs.append(idx)                 # l_jj
        idx += 1 + (k - 1 - j)          # skip l_jj + all z below it in column j
    return np.array(idxs, dtype=int)

def sample_momentum_lz(rng, k: int) -> np.ndarray:
    n = k * (k + 1) // 2
    s_l = 1
    s_z = np.exp(s_l)
    p = rng.normal(0.0, s_z, size=n)          # default: z blocks
    l_idx = diag_param_indices(k)
    p[l_idx] = rng.normal(0.0, s_l, size=k)   # overwrite: l blocks
    p[0] = rng.normal(0.0, s_l*5)
    return p

def inv_mass_vector(k: int) -> np.ndarray:
    n = k * (k + 1) // 2
    s_l = 1
    s_z = np.exp(s_l)
    invM = np.ones(n) / (s_z**2)
    invM[diag_param_indices(k)] = 1.0 / (s_l**2)
    invM[0] = 1.0 / ((s_l*5)**2)
    return invM

# ---------- HMC ----------
def pdhmc_lz(returns, method, num_samples, eps, T=1, check_mu=0):
    rng = np.random.default_rng()   
    (n,k) = returns.shape
    if check_mu==0:
        mu_t = np.zeros(k)
    else:
        mu_t = np.mean(returns, axis=0)
    Sigma_t = np.cov(returns, rowvar=False)*1.1

    d = Sigma_t.shape[0]
    
    e_list = []
    posterior_Sigma = []
    Sigma_accepts = 0
    posterior_mu = []

    if method == "Jeffreys":
        pi = pi_Jeffreys
    
    if method == "Right":
        pi = pi_Right

    rng = np.random.default_rng()
    for _ in range(num_samples):
        # Sigma
        # momentum ~ N(0, G^{-1}(Σ_t)) using PSD square root
        vvech = sample_momentum_lz(rng, d)

        # initial energy
        e0 = energy_lz(Sigma_t, vvech, np.log(pi(returns, mu_t, Sigma_t)))
        Sigma_star = Sigma_t.copy()
        lz_star = sigma_to_lz(Sigma_star)
        for _ in range(T):
            # half position
            lz_star = lz_star + (eps / 2.0) * (vvech / inv_mass_vector(d))

            # update Sigma to match current position before computing gradient
            Sigma_star = lz_to_sigma(lz_star, d)
            # momentum full step: p += eps * ∇ log target(lz)
            #grad_total = grad(mu_t, Sigma_star, returns, method)          # ∇_Sigma log π(Sigma)
            grad_total_lz = grad_lz(mu_t, Sigma_star, returns, method)               # ∇_lz [log π(Sigma(lz)) + logJac]
            vvech = vvech + eps * grad_total_lz

            # half position
            lz_star = lz_star + (eps / 2.0) * (vvech / inv_mass_vector(d))

        Sigma_star = lz_to_sigma(lz_star,d)
        e1 = energy_lz(Sigma_star, vvech, np.log(pi(returns, mu_t, Sigma_star)))
        acc_prob = min(1.0, float(np.exp(e0 - e1)))
        if rng.random() < acc_prob:
            Sigma_accepts += 1   # only final transition is accepted
            e_list.append(e1)
            Sigma_t = Sigma_star
            #print(e1)
        else:
            e_list.append(e0)
        posterior_Sigma.append(Sigma_t)

        # mu
        if check_mu != 0:    
            mu_t = multivariate_normal.rvs(mean=np.mean(returns, axis=0), cov=Sigma_t/n)
        posterior_mu.append(mu_t)
            
    return {
        "mu": posterior_mu,
        "Sigma": posterior_Sigma,
        "acc_Sigma": Sigma_accepts / num_samples,
        "energy": e_list
    }

def run_HMC_parallel_lz(returns, method, num_samples, eps, T, check_mu, num_chains):
    @parfor(range(num_chains))
    def HMC_all(i):
        return pdhmc_lz(returns, method, num_samples, eps, T, check_mu)    
    results = HMC_all

    mu_chains = [x for r in results for x in r["mu"]]
    Sigma_chains = [x for r in results for x in r["Sigma"]]
    acc_Sigma_list = [r["acc_Sigma"] for r in results]
    e_list = [x for r in results for x in r["energy"]]

    print(f"Avg Sigma acceptance rate: {np.mean(acc_Sigma_list):.3f}")
    return mu_chains, Sigma_chains, e_list, round(np.mean(acc_Sigma_list),3)

# ---------- ESS ----------
def autocorr_fft_1d(x: np.ndarray) -> np.ndarray:
    """
    Unbiased-ish autocorrelation via FFT (normalized so rho[0]=1).
    Returns rho[0..N-1].
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    if n < 2:
        return np.ones(n)

    # next power of two for speed
    m = 1 << (2*n - 1).bit_length()
    fx = np.fft.rfft(x, n=m)
    acf = np.fft.irfft(fx * np.conjugate(fx), n=m)[:n]
    # normalize
    acf /= acf[0]
    return acf

def ess_1d_geyer(x: np.ndarray) -> float:
    """
    ESS for a single chain using Geyer's initial positive sequence on autocorrelation.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        return float(n)

    rho = autocorr_fft_1d(x)

    # Geyer initial positive sequence on pairs gamma_k = rho[2k-1] + rho[2k]
    tau = 1.0
    k = 1
    while 2*k < n:
        gamma = rho[2*k - 1] + rho[2*k]
        if gamma <= 0:
            break
        tau += 2.0 * gamma
        k += 1

    ess = n / tau
    # numerical guard
    return float(max(1.0, min(n, ess)))

def ess_many(series_2d: np.ndarray) -> np.ndarray:
    """
    series_2d shape: (N, P). Returns ESS for each column.
    """
    series_2d = np.asarray(series_2d, dtype=float)
    if series_2d.ndim != 2:
        raise ValueError("series_2d must be 2D with shape (N, P)")
    N, P = series_2d.shape
    out = np.empty(P, dtype=float)
    for j in range(P):
        out[j] = ess_1d_geyer(series_2d[:, j])
    return out

if __name__ == "__main__":
    # Samples
    n = 50
    k = 15
    check_mu=1
    vol_range = (0.05, 0.2)
    returns, mu, Sigma = generate_simulation_data(k, n, vol_range, check_mu=check_mu)
    print("True mu",mu)
    print("True Sigma", Sigma)

    # True Posterior
    mu_sample = np.mean(returns, axis=0)
    Sigma_sample = np.cov(returns, rowvar=False)
    print("Mean of Posterior", (Sigma_sample * (n-1) + np.outer(mu - mu_sample, mu - mu_sample) * n)/(n-k-1))
    #print("Varience of Posterior")
    num_samples = 1000
    True_samples = invwishart.rvs(df=n, scale=Sigma_sample * (n-1) + np.outer(mu_sample-mu, mu_sample-mu) * n, size=num_samples)
    print("Sampling mean of Sigma", np.mean(True_samples, axis=0))
    print("Sampling error of Sigma",np.sqrt(np.var(True_samples, axis=0, ddof=0)/num_samples))

    # HMC of Posterior
    rng = np.random.default_rng(0)
    method = "Right"
    num_chains = 5
    eps= 2e-5
    T = 3

    results = pdhmc_lz (returns, method, eps=eps, num_samples=5000, T=T, check_mu=check_mu)
    posterior_mu = results["mu"]
    posterior_Sigma = results["Sigma"]
    e_list = results["energy"]
    acc = results["acc_Sigma"]
    #posterior_mu,posterior_Sigma,e_list,acc = run_HMC_parallel_lz(returns, method, eps=eps, num_samples=num_samples, T=T, check_mu=check_mu, num_chains=num_chains)

    #print(e_list[:3])
    #posterior_mu, posterior_Sigma, e_list, acc = run_HMC_parallel_lz(returns, method, num_samples=1000, eps=eps, T=T, s=0.5, check_mu=check_mu, num_chains=num_chains)
    print("HMC mean of mu",np.mean(posterior_mu, axis=0))
    print("HMC mean of Sigma",np.mean(posterior_Sigma, axis=0))
    posterior_Sigma = np.array(posterior_Sigma)
    print("HMC error of Sigma",np.sqrt(np.var(posterior_Sigma, axis=0, ddof=0)/num_samples))

    diag = np.stack([np.diag(S) for S in posterior_Sigma], axis=0)  # (N, d)
    ess_diag = ess_many(diag)
    print("ESS diag:", ess_diag)
    print("min/median:", ess_diag.min(), np.median(ess_diag))

    # === Plot trace for each element of mu ===
    posterior_mu = np.array(posterior_mu)
    print(posterior_mu.shape)
    posterior_Sigma = np.array(posterior_Sigma)
    print(posterior_Sigma.shape)
    BIG = 18
    plt.rcParams.update({
        "font.size": BIG,
        "axes.labelsize": BIG,
        "xtick.labelsize": BIG,
        "ytick.labelsize": BIG+1,
        "axes.titlesize": BIG+2,
    })

    # === Plot trace for μ ===
    fig_mu, axes_mu = plt.subplots(k, 1, figsize=(10, 2*k), sharex=True)

    fig_mu.suptitle('μ', fontsize=BIG+4)

    for i in range(k):
        axes_mu[i].plot(posterior_mu[:, i])
        axes_mu[i].set_ylabel(f'μ[{i+1}]', fontsize=BIG+1)

    axes_mu[-1].set_xlabel('Iteration', fontsize=BIG+6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'result/{n}_{k}_trace_mu_lz.png', dpi=300)
    plt.close(fig_mu)

    # === Plot trace for diagonal of Σ ===
    fig_var, axes_var = plt.subplots(k, 1, figsize=(10, 2*k), sharex=True)

    fig_var.suptitle(
        f'Σ with acceptance ratio = {acc:.3f} (eps = {eps}, T = {T})',
        fontsize=BIG+4
    )

    for i in range(k):
        axes_var[i].plot(posterior_Sigma[:, i, i])
        axes_var[i].set_ylabel(f'Σ[{i+1},{i+1}]', fontsize=BIG+1)

    axes_var[-1].set_xlabel('Iteration', fontsize=BIG+6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'result/{n}_{k}_trace_sigma_diag_lz.png', dpi=300)
    plt.close(fig_var)
    
    plt.plot(e_list)
    plt.rcParams.update({
        "font.size": BIG,
        "axes.labelsize": BIG,
        "xtick.labelsize": BIG-5,
        "ytick.labelsize": BIG-5,
        "axes.titlesize": BIG,
    })
    plt.xlabel("Number of Sampling", fontsize=BIG-2)
    plt.ylabel("Energy", fontsize=BIG-2)
    plt.title("Energy of HMC Samples", fontsize=BIG-2)
     # Save figure
    plt.savefig(f"result/{n}_{k}_pdhmc_e_hist_lz.png", bbox_inches='tight')
    plt.close()  # <-- Do NOT show the figure
