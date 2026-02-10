import numpy as np
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel
from data_simulation import generate_simulation_data

# ------------------ HMC Stan ------------------
def run_HMC_Stan0(returns, method, num_samples, num_chains, iter_warmup=1000):
    stan_file = f"code/priors/mvn_{method}_L0.stan"
    model = CmdStanModel(stan_file=stan_file)
    (n,k) = returns.shape
    stan_data = {
    "n": n,
    "k": k,
    "y": returns.tolist()
    }
    fit = model.sample(
    data=stan_data,
    seed=42,
    adapt_delta=0.99,
    chains=num_chains,
    parallel_chains=num_chains,
    iter_warmup=1000,
    iter_sampling=num_samples
    )
    posterior_Sigma = fit.stan_variable("Sigma")   # shape (S, k, k)

    return np.zeros((num_samples, k)), posterior_Sigma

def run_HMC_Stan(returns, method, num_samples, num_chains, iter_warmup=1000):
    stan_file = f"code/priors/mvn_{method}_L.stan"
    model = CmdStanModel(stan_file=stan_file)
    (n,k) = returns.shape
    stan_data = {
    "n": n,
    "k": k,
    "y": returns.tolist()
    }
    fit = model.sample(
    data=stan_data,
    seed=42,
    adapt_delta=0.99,
    chains=num_chains,
    parallel_chains=num_chains,
    iter_warmup=iter_warmup,
    iter_sampling=num_samples
    )
    posterior_mu = fit.stan_variable("mu")         # shape (S, k)
    posterior_Sigma = fit.stan_variable("Sigma")   # shape (S, k, k)
    
    return posterior_mu, posterior_Sigma

if __name__ == "__main__":
    np.random.seed(123)
    k = 2
    n = 10
    check_mu = 0
    vol_range = (0.05, 0.2)
    returns, mu, Sigma = generate_simulation_data(k, n, vol_range, check_mu=check_mu)
    print("True mu",mu)
    print("True Sigma", Sigma)

    # ------------------ Extract posteriors ------------------
    method = "rscale"
    num_chains = 5
    num_samples = 1000
    iter_warmup = 1000

    posterior_mu, posterior_Sigma = run_HMC_Stan0(returns, method, num_samples, num_chains, iter_warmup)

    print("Posterior mean of mu:\n", posterior_mu.mean(axis=0))
    mu_sample = np.mean(returns, axis=0)
    print("Posterior Mean of mu(Theor.):\n", mu_sample)

    print("Posterior mean of Sigma:\n",posterior_Sigma.mean(axis=0))
    Sigma_sample = np.cov(returns, rowvar=False)
    print("Posterior Mean of Sigma(Theor.):\n", Sigma_sample * (n-1)/(n-k-2))

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
    plt.savefig(f'result/{num_samples}_{iter_warmup}_trace_mu_stan.png', dpi=300)
    plt.close(fig_mu)

    # === Plot trace for diagonal of Σ ===
    fig_var, axes_var = plt.subplots(k, 1, figsize=(10, 2*k), sharex=True)

    fig_var.suptitle(
        'Σ',
        fontsize=BIG+4
    )

    for i in range(k):
        axes_var[i].plot(posterior_Sigma[:, i, i])
        axes_var[i].set_ylabel(f'Σ[{i+1},{i+1}]', fontsize=BIG+1)

    axes_var[-1].set_xlabel('Iteration', fontsize=BIG+6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'result/{num_samples}_{iter_warmup}_trace_sigma_diag_stan.png', dpi=300)
    plt.close(fig_var)