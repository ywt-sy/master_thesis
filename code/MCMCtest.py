import numpy as np
from scipy.stats import multivariate_normal as MVN
from scipy.stats import invwishart, multivariate_normal
from scipy.special import logsumexp
from data_simulation import generate_simulation_data
from MHMC import run_MH_parallel
from HMC_SPD import run_HMC_parallel
from HMC_lz import run_HMC_parallel_lz
from HMC_Stan import run_HMC_Stan, run_HMC_Stan0 
import time
import os
def run_MC_parallel(returns, method, S_post=1000,
                    num_chains=10, df=50, eps=1e-4, T=2, s=0.5, check_mu=0):
    
    if method == "jlz":
        mu_samples, Sigma_samples, e_list = run_HMC_parallel_lz(
            returns, "Jeffreys",
            num_samples=S_post,
            eps=eps,
            T=T,
            s=s,
            check_mu=check_mu,
            num_chains=num_chains
        )
    
    elif method == "jst" and check_mu==0:
        mu_samples, Sigma_samples = run_HMC_Stan0(
            returns, "Jeffreys",
            num_samples=S_post,
            num_chains=num_chains
        )
        e_list = None
    
    elif method == "jst" and check_mu==1:
        mu_samples, Sigma_samples = run_HMC_Stan(
            returns, "Jeffreys",
            num_samples=S_post,
            num_chains=num_chains
        )
        e_list = None


    elif method == "jhm":
        mu_samples, Sigma_samples, e_list = run_HMC_parallel(
            returns, "Jeffreys",
            num_samples=S_post,
            eps=eps,
            T=T,
            check_mu=check_mu,
            num_chains=num_chains
        )

    elif method == "jmc":
        mu_samples, Sigma_samples = run_MH_parallel(
            returns, "Jeffreys",
            num_samples=S_post,
            df=df,
            check_mu=check_mu,
            num_chains=num_chains
        )
        e_list = None

    elif method == "rlz":
        mu_samples, Sigma_samples, e_list = run_HMC_parallel_lz(
            returns, "Right",
            num_samples=S_post,
            eps=eps,
            T=T,
            s=s,
            check_mu=check_mu,
            num_chains=num_chains
        )

    elif method == "rst" and check_mu==0:
        mu_samples, Sigma_samples = run_HMC_Stan0(
            returns, "right",
            num_samples=S_post,
            num_chains=num_chains
        )
        e_list = None

    elif method == "rst" and check_mu==1:
        mu_samples, Sigma_samples = run_HMC_Stan(
            returns, "right",
            num_samples=S_post,
            num_chains=num_chains
        )
        e_list = None

    elif method == "rhm":
        mu_samples, Sigma_samples, e_list = run_HMC_parallel(
            returns, "Right",
            num_samples=S_post,
            eps=eps,
            T=T,
            check_mu=check_mu,
            num_chains=num_chains
        )

    elif method == "rmc":
        mu_samples, Sigma_samples = run_MH_parallel(
            returns, "Right",
            num_samples=S_post,
            df=df,
            check_mu=check_mu,
            num_chains=num_chains
        )
        e_list = None

    elif method == "rscale" and check_mu==0:
        mu_samples, Sigma_samples = run_HMC_Stan0(
            returns, "rscale",
            num_samples=S_post,
            num_chains=num_chains
        )
        e_list = None

    elif method == "rscale" and check_mu==1:
        mu_samples, Sigma_samples = run_HMC_Stan(
            returns, "rscale",
            num_samples=S_post,
            num_chains=num_chains
        )
        e_list = None

    else:
        raise ValueError(f"Unknown method: {method}")

    return mu_samples, Sigma_samples, e_list
# ------------------------------------------------------------
# Stable log mixture: log q(y) = log [ (1/S) Σ_s N(y; μ_s, Σ_s) ]
# ------------------------------------------------------------
def log_mixture_gaussian_pdf(Y, mus, Sigmas):
    """
    Stable evaluation of log[(1/S) * sum_s N(Y; mus[s], Sigmas[s])].
    Y: (J, k)
    mus: (S, k)
    Sigmas: (S, k, k)
    Returns: (J,) array of log mixture densities.
    """
    Y = np.asarray(Y, float)
    mus = np.asarray(mus, float)
    Sigmas = np.asarray(Sigmas, float)
    S = mus.shape[0]

    if Y.ndim == 2:
        Y = Y[:, None, :]   # (J, 1, k)

    J, m, k = Y.shape
    comp_logs = np.zeros((J, S))

    for s in range(S):
        mvn = MVN(mean=mus[s], cov=Sigmas[s])
        for t in range(m):
            comp_logs[:, s] += mvn.logpdf(Y[:, t, :])

    # log( (1/S) Σ exp(log p_s) ) = logsumexp - log S
    return logsumexp(comp_logs, axis=1) - np.log(S)

def predictive_loss_LJ(mu_true, Sigma_true,
                       method,        # "Jeffreys" or a callable (passed to run_MH_parallel)
                       n, k,
                       m=1,
                       L=10, J=5000, S_post=1000,
                       num_chains=10, df=50, eps=1e-4, T=2, s=0.5, check_mu=0,
                       seed=None):
    """
    Implements:
      - Draw L datasets x^{(ℓ)} ~ N_k(μ, Σ) of size n each
      - Posterior samples θ^{(s,ℓ)} ~ π(θ | x^{(ℓ)}) via run_MH_parallel(...)
      - Draw J test points y^{(j)} ~ N_k(μ, Σ)
      - Loss = -(1/(LJ)) Σ_{ℓ,j} log q(y^{(j)} | x^{(ℓ)}),
                where q(y|x^{(ℓ)}) ≈ (1/S_post) Σ_s N_k(y; μ^{(s,ℓ)}, Σ^{(s,ℓ)})
    """
    rng = np.random.default_rng(seed)

    mu_true = np.asarray(mu_true, float).ravel()
    Sigma_true = np.asarray(Sigma_true, float)
    k_true = mu_true.shape[0]
    assert k_true == k, "k does not match dimension of mu_true."

    # Pre-sample Y once (shared across ℓ)
    Y = rng.multivariate_normal(mu_true, Sigma_true, size=(J, m))  # (J, k)
    #print(Y)

    # FIRST term: (1/J) Σ_j log p(y^{(j)} | θ)
    mvn_true = MVN(mean=mu_true, cov=Sigma_true)
    first_term = float(np.mean(mvn_true.logpdf(Y)))

    # SECOND term
    total_logq = 0.0
    loss_L = []
    bad_chain = 0
    
    for ell in range(L):
        # Simulate x^{(ℓ)} under true θ
        X_ell = rng.multivariate_normal(mu_true, Sigma_true, size=n)

        # Obtain posterior samples for this dataset
        # run_MH_parallel should return shapes:
        #   mu_samples:    (S_post, k)
        #   Sigma_samples: (S_post, k, k)

        mu_samples, Sigma_samples, e_list = run_MC_parallel(
                                                X_ell,
                                                method,
                                                S_post=S_post,
                                                eps=eps,
                                                df=df,
                                                T=T,
                                                s=s,
                                                check_mu=check_mu,
                                                num_chains=num_chains
                                            )

        mu_samples = np.asarray(mu_samples, float)
        Sigma_samples = np.asarray(Sigma_samples, float)

        # q(y | x^{(ℓ)}) at each y^{(j)}: Gaussian mixture
        logq = log_mixture_gaussian_pdf(Y, mu_samples, Sigma_samples)  # (J,)   
        with open(f"result/simulation/{method}_lossfunc_J.dat", "a") as f:     
            f.write(f"{(n,k,J)} {float(np.mean(logq))} {float(np.sqrt(np.cov(logq)/J))}\n")
    
        loss_L.append(np.mean(logq))
        total_logq += np.sum(logq)
    with open(f"result/simulation/{n}_{k}_lossfunc.dat", "a") as f:
            f.write(f"{method} {(n,k,L-bad_chain)} {float(np.mean(loss_L))} {float(np.sqrt(np.cov(loss_L)/L))}\n")
    # Monte Carlo estimate (the -E[log q] term)
    loss_estimate = -(1.0 / (L * J)) * total_logq
    return loss_estimate, float(np.sqrt(np.cov(loss_L)/L))

# ------------------------------------------------------------
# Wrapper to compare priors (Jeffreys vs. right-invariant)
# ------------------------------------------------------------
def simu_testMCMC(
    n_simulation, n, k, m, num_chains,
    df_scale, eps, check_mu, T=2,s=0.5,
    L=100, J=5000, S_post=1000, seed=None,
    method_list= ["jhm", "jmc", "rhm", "rmc"]
):
    """
    Run predictive_loss_LJ across n_simulation repetitions for multiple methods.

    df_scale: (df_scale_for_Jeffreys, df_scale_for_Right)
      - methods starting with 'j' use df = n / df_scale[0]
      - methods starting with 'r' use df = n / df_scale[1]

    Returns:
      dict: {method: mean_loss}
    """
    rng = np.random.default_rng(seed)
    vol_range = (0.05, 0.2)

    # store all losses
    loss_dict = {method: [] for method in method_list}

    os.makedirs("result/simulation", exist_ok=True)

    for _ in range(n_simulation):
        # fresh truth per simulation
        _, mu_true, Sigma_true = generate_simulation_data(k, n, vol_range, check_mu=check_mu)

        for method in method_list:
            # choose df scale by prior family
            if method.startswith("j"):
                df = n / df_scale[0]
            elif method.startswith("r"):
                df = n / df_scale[1]
            else:
                raise ValueError(f"Unknown method prefix in {method}")

            loss, error = predictive_loss_LJ(
                mu_true, Sigma_true,
                method=method,
                n=n, k=k, m=m,
                L=L, J=J, S_post=S_post,
                num_chains=num_chains,
                df=df, eps=eps, T=T,s=s,
                check_mu=check_mu,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            loss_dict[method].append((loss,error))

    # compute averages
    result = {}
    for method, values in loss_dict.items():
        losses = [v[0] for v in values]
        errors = [v[1] for v in values]
        result[method] = {
            "avg_loss": float(np.mean(losses)),
            "avg_error": float(np.mean(errors)),
        }

    # print results nicely
    for method in method_list:
        print(f"{method}: avg loss = {result[method]['avg_loss']:.4f}, "
            f"avg error = {result[method]['avg_error']:.4f}")

    # -------------------------------------------
    # ★★★ Append results to file ★★★
    # -------------------------------------------

    output_path = "result/simulation/lossfunc_all.dat"
    os.makedirs("result/simulation", exist_ok=True)

    with open(output_path, "a") as f:
        for method in method_list:
            avg_loss = result[method]["avg_loss"]
            avg_error = result[method]["avg_error"]
            f.write(
                f"(n={n}, k={k}, m={m})  {method}  "
                f"{avg_loss:.6f}  {avg_error:.6f}\n"
            )

    return result

if __name__ == "__main__":
    n = 10
    k = 2
    method_list= ["rscale","rst"]
    df_scale = [1.6, 1.6]
    num_chains = 5    # MCMCの並列数
    n_simulation = 1   # シミュレーションの回数
    m = 10
    eps=1e-3
    T = 5 
    s = 0.5
    check_mu=0

    for k in [2,3]:
        if k == 2:
            n_list = [50,75,100]
        if k == 3:
            n_list = [50,75,100]
        for n in n_list:
            #(If accept for Sigma ^ -> df_scale ^, If accpet for mu ^ -> scale_mu ^)
            if k == 5: #(Checked OK)
                if n == 50: # About 371.471 seconds for 1 time
                    df_scale = [1.13,1.11]

                if n == 75: # About 113.436 seconds for 1 time
                    df_scale = [1.12,1.07] #for[JMCMC,Right,Rscale,Rrotate]

                if n == 100: # About 112.417 seconds for 1 time
                    df_scale = [1.12,1.07,1.07,1.07] #for[JMCMC,Right,Rscale,Rrotate]

            if k == 10: #(Checked OK)
                if n == 50: # About 189.214 seconds for 1 time
                    df_scale = [1.10,0.9] #for[JMCMC,Right,Rscale,Rrotate]

                if n == 75: # About 188.309 seconds for 1 time
                    df_scale = [1.08,0.94] #for[JMCMC,Right,Rscale,Rrotate]

                if n == 100: # About 185.239 seconds for 1 time
                    df_scale = [1.07,0.97] #for[JMCMC,Right,Rscale,Rrotate]

            if k == 15: #(Check Right latter
                if n == 50: # About 773.021 seconds for 1 time
                    df_scale = [0.93,0.8] #for[JMCMC,Right,Rscale,Rrotate]

                if n == 75: # About 309.385 seconds for 1 time
                    df_scale = [0.94,0.85] #for[JMCMC,Right,Rscale,Rrotate]

                if n == 100: # About 315.734 seconds for 1 time
                    df_scale = [0.95,0.95] #for[JMCMC,Right,Rscale,Rrotate]

            start = time.perf_counter()
            mean_loss = simu_testMCMC(n_simulation=n_simulation, n=n, k=k, m=m,num_chains=num_chains, df_scale=df_scale, eps=eps, T=T, s=s,
                                                                                        check_mu=check_mu, L=100, J=5000, S_post=1000,method_list= method_list)
            end = time.perf_counter()

            print("Time", round(end-start,3),"seconds")
