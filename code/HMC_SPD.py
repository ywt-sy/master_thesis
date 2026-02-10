import numpy as np
from parfor import parfor
from scipy.linalg import expm
from scipy.stats import wishart, invwishart, multivariate_normal
from data_simulation import generate_simulation_data
from Est2_JR import pi_Jeffreys, pi_Right
import matplotlib.pyplot as plt
import time


# ---------- vec/vech tools ----------
def sym(A):
    return 0.5 * (A + A.T)

def vech(A):
    d = A.shape[0]
    i, j = np.triu_indices(d)
    return A[i, j].copy()

def invech(v, d):
    V = np.zeros((d, d), dtype=v.dtype)
    idx = 0
    for j in range(d):          # col
        for i in range(j,d):  # row, i>=j (lower triangle)
            V[i, j] = v[idx]
            V[j, i] = v[idx]
            idx += 1
    return V

def D_d(d):
    m = d * (d + 1) // 2
    D = np.zeros((d*d, m))
    col = 0
    for j in range(d):
        for i in range(j+1):
            E = np.zeros((d, d))
            E[i, j] = 1.0
            E[j, i] = 1.0
            D[:, col] = E.reshape(-1)
            col += 1
    return D

def D_d_plus(D):
    DtD = D.T @ D
    DtD_inv = np.linalg.inv(DtD)
    return DtD_inv @ D.T

# ---------- metric tensors ----------
def metric_G(Sigma):
    D = D_d(Sigma.shape[0])
    Sinv = np.linalg.inv(Sigma)
    return D.T @ (np.kron(Sinv, Sinv)) @ D

def metric_G_inv(Sigma):
    D = D_d(Sigma.shape[0])
    Dp = D_d_plus(D)
    return Dp @ (np.kron(Sigma, Sigma)) @ Dp.T

# ---------- energy ----------
def energy(Sigma, vvech, log_pi_val):
    #print(vvech.shape)
    d = Sigma.shape[0]
    sign, logdet = np.linalg.slogdet(Sigma)
    Gmat = metric_G(Sigma)
    #print(Gmat.shape)
    if not np.isfinite(logdet) or sign <= 0:
        return np.inf
    return float(-log_pi_val - ((d + 1)/2) * logdet + 0.5 * vvech.T @ Gmat @ vvech)

# ---------- Euclidean gradient ----------
def grad(mu, Sigma, returns, method):
    """
    Returns ∇_Σ [ log π(Σ) + (d+1) log|Σ| ] (Euclidean gradient on Sym^+(d)).
    """
    # central differences on the upper triangle
    (n,k) = returns.shape
    mu_samples = np.mean(returns, axis=0)
    Sigma_sample = np.cov(returns, rowvar=False)
    S_star = Sigma_sample * (n-1) + np.outer(mu_samples-mu, mu_samples-mu) * n
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
    return G_1 + ((k + 1)/2) * Sigma_inv

# ---------- geodesic flow (affine-invariant) ----------
def Eigenvalue_decomp(A):
    if A.shape[0]!=A.shape[1]:
        raise Exception("error! A is not square matrix!")
    if (A==A.T).all()==False: 
        A = sym(A)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    return A, eigenvectors, eigenvalues

def geodesic_flow(Sigma, V, t):
    Sigma, eigenvectors, eigenvalues = Eigenvalue_decomp(Sigma)
    Diag_2 = np.diag(eigenvalues**0.5)
    Div_2 = np.diag(eigenvalues**(-0.5))
    SVS = (eigenvectors @ Div_2 @ eigenvectors.T) @ V @ (eigenvectors @ Div_2 @ eigenvectors.T) 
    Sigma_t = (eigenvectors @ Diag_2 @ eigenvectors.T) @ expm(t * SVS) @ (eigenvectors @ Diag_2 @ eigenvectors.T)
    V_t = V @ (eigenvectors @ Div_2 @ eigenvectors.T) @ expm(t * SVS) @ (eigenvectors @Diag_2 @ eigenvectors.T)
    return Sigma_t, V_t

# ---------- Normal ----------
def sample_normal(Sigma, n_samples=1):
    """
    Sample from N(0, Sigma).
    """
    d = Sigma.shape[0]

    # Cholesky: Sigma = L L^T
    L = np.linalg.cholesky(Sigma)

    # Standard normal
    Z = np.random.randn(n_samples, d)

    # Transform
    return Z @ L.T

def pdhmc(returns, method, num_samples, eps, T=1, check_mu=0):
    rng = np.random.default_rng()   
    (n,k) = returns.shape
    if check_mu==0:
        mu_t = np.zeros(k)
    else:
        mu_t = np.mean(returns, axis=0)
    Sigma_t = np.cov(returns, rowvar=False)*1.1

    d = Sigma_t.shape[0]
    D = D_d(d)
    Dp = D_d_plus(D)
    
    e_list = []
    Sigma_samples = []
    Sigma_accepts = 0
    mu_samples = []

    if method == "Jeffreys":
        pi = pi_Jeffreys
    
    if method == "Right":
        pi = pi_Right

    for _ in range(num_samples):
        # Sigma
        # momentum ~ N(0, G^{-1}(Σ_t)) using PSD square root
        Ginv = metric_G_inv(Sigma_t)
        vvech = sample_normal(Ginv, n_samples=1).flatten()
        V = invech(vvech, d)

        # initial energy
        e0 = energy(Sigma_t, vvech, np.log(pi(returns, mu_t, Sigma_t)))
        Sigma_star = Sigma_t.copy()
        for _ in range(T):
            # half momentum
            grad_total = grad(mu_t, Sigma_star, returns, method)
            vvech = vvech + 0.5 * eps * metric_G_inv(Sigma_star) @ vech(grad_total)
            V = invech(vvech, d)

            # geodesic
            Sigma_star, V = geodesic_flow(Sigma_star, V, eps)

            # half momentum
            grad_total = grad(mu_t, Sigma_star, returns, method)
            vvech = vvech + 0.5 * eps * metric_G_inv(Sigma_star) @ vech(grad_total)
            V = invech(vvech, d)
            #print(Sigma_star)

        e1 = energy(Sigma_star, vvech, np.log(pi(returns, mu_t, Sigma_star)))
        acc_prob = min(1.0, float(np.exp(e0 - e1)))
        if rng.random() < acc_prob:
            Sigma_accepts += 1   # only final transition is accepted
            e_list.append(e1)
            Sigma_t = Sigma_star
            #print(e1)
        else:
            e_list.append(e0)
        Sigma_samples.append(Sigma_t)

        # mu
        if check_mu != 0:    
            mu_t = multivariate_normal.rvs(mean=np.mean(returns, axis=0), cov=Sigma_t/n)
        mu_samples.append(mu_t)
            
    return {
        "mu": mu_samples,
        "Sigma": Sigma_samples,
        "acc_Sigma": Sigma_accepts / num_samples,
        "energy": e_list
    }

def run_HMC_parallel(returns, method, num_samples, eps, T, check_mu, num_chains):
    @parfor(range(num_chains))
    def HMC_all(i):
        return pdhmc(returns, method, num_samples, eps, T, check_mu)    
    results = HMC_all

    mu_chains = [x for r in results for x in r["mu"]]
    Sigma_chains = [x for r in results for x in r["Sigma"]]
    acc_Sigma_list = [r["acc_Sigma"] for r in results]
    e_list = [x for r in results for x in r["energy"]]

    print(f"Avg Sigma acceptance rate: {np.mean(acc_Sigma_list):.3f}")
    return mu_chains, Sigma_chains, e_list#, round(np.mean(acc_Sigma_list),3)

if __name__ == "__main__":
    # Samples
    n = 50
    k = 5
    check_mu=1
    vol_range = (0.05, 0.2)
    returns, mu, Sigma = generate_simulation_data(k, n, vol_range, check_mu=check_mu)
    print("True mu",mu)
    print("True Sigma", Sigma)

    # True Posterior
    mu_samples = np.mean(returns, axis=0)
    Sigma_sample = np.cov(returns, rowvar=False)
    print("Mean of Posterior", (Sigma_sample * (n-1) + np.outer(mu - mu_samples, mu - mu_samples) * n)/(n-k-1))
    num_samples = 10000
    True_samples = invwishart.rvs(df=n, scale=Sigma_sample * (n-1) + np.outer(mu_samples-mu, mu_samples-mu) * n, size=num_samples)
    print("Sampling mean of Sigma", np.mean(True_samples, axis=0))
    print("Sampling error of Sigma",np.sqrt(np.var(True_samples, axis=0, ddof=0)/num_samples))

    # HMC of Posterior
    rng = np.random.default_rng(0)
    method = "Right"
    num_chains = 5
    eps=2e-3
    T=15
    results = pdhmc(returns, method, eps=eps, num_samples=5000, T=T, check_mu=check_mu)
    mu_samples = results["mu"]
    Sigma_samples = results["Sigma"]
    e_list = results["energy"]
    acc = results["acc_Sigma"]
    #print(e_list[:3])
    #mu_samples, Sigma_samples, e_list, acc = run_HMC_parallel(returns, method, num_samples=1000, eps=eps, T=T, check_mu=check_mu, num_chains=num_chains)
    print("HMC mean of mu",np.mean(mu_samples, axis=0))
    print("HMC mean of Sigma",np.mean(Sigma_samples, axis=0))
    Sigma_samples = np.array(Sigma_samples)
    print("HMC error of Sigma",np.sqrt(np.var(Sigma_samples, axis=0, ddof=0)/1000))

    # === Plot trace for each element of mu ===
    mu_samples = np.array(mu_samples)
    print(mu_samples.shape)
    Sigma_samples = np.array(Sigma_samples)
    print(Sigma_samples.shape)
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
        axes_mu[i].plot(mu_samples[:, i])
        axes_mu[i].set_ylabel(f'μ[{i+1}]', fontsize=BIG+1)

    axes_mu[-1].set_xlabel('Iteration', fontsize=BIG+6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'result/{n}_{k}_trace_mu_SPD.png', dpi=300)
    plt.close(fig_mu)



    # === Plot trace for diagonal of Σ ===
    fig_var, axes_var = plt.subplots(k, 1, figsize=(10, 2*k), sharex=True)

    fig_var.suptitle(
        f'Σ with acceptance ratio = {acc:.3f} (eps = {eps}, T = {T})',
        fontsize=BIG+4
    )

    for i in range(k):
        axes_var[i].plot(Sigma_samples[:, i, i])
        axes_var[i].set_ylabel(f'Σ[{i+1},{i+1}]', fontsize=BIG+1)

    axes_var[-1].set_xlabel('Iteration', fontsize=BIG+6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'result/{n}_{k}_trace_sigma_diag_SPD.png', dpi=300)
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
    plt.title("Energy of PDHMC Samples", fontsize=BIG-2)
     # Save figure
    plt.savefig(f"result/pdhmc_e_hist_SPD.png", bbox_inches='tight')
    plt.close()  # <-- Do NOT show the figure
