import numpy as np
from parfor import parfor
from scipy.stats import invwishart, multivariate_normal
from data_simulation import generate_simulation_data
from Est0_true import true_weights_and_moments
from Est2_JR import minors_product_radio, minors_product_radio_log
import matplotlib.pyplot as plt
import time

def MH(returns, method, num_samples, df, check_mu):
    (n,k) = returns.shape
    mu_list = []
    Sigma_list = []
 
    Sigma_init = np.cov(returns, rowvar=False)
    Sigma = Sigma_init
    mu_init = np.mean(returns, axis=0)

    if check_mu ==0:
        mu = np.zeros(k)
    else:
        mu = mu_init

    accepted_Sigma = 0

    for _ in range(num_samples):
        # Step 1: propose Sigma* from inverse-Wishart
        scale_matrix = Sigma_init * (n-1) + np.outer(mu_init - mu, mu_init - mu) * n
        Sigma_candidate = invwishart.rvs(df=df, scale=scale_matrix)

        # Step 2: acceptance ratio for Sigma
        if method == "Jeffreys":
            a = (k + 1) / 2
            term1 = 1
        
        else:
            a = 0
            if method == "Right":
                term1 = minors_product_radio_log(Sigma,Sigma_candidate)

            """
            elif method == "Rscale":
                term1 = scale_minors_product_radio(Sigma,Sigma_candidate)

            elif method == "Rrotate":
                term1 = rotate_minors_product_radio(Sigma,Sigma_candidate)

            """

        lambda_candidate = np.linalg.eigvalsh(Sigma_candidate)[::-1]
        lambda_t = np.linalg.eigvalsh(Sigma)[::-1]
        term2 = np.prod([(lambda_candidate[i] / lambda_t[i]) ** ((df + k + 1 - 2*a - n) /2) for i in range(k)])
        
        acceptance_ratio_Sigma = min(1, term1*term2)
        #print("all",term1*term2)

        if np.random.rand() < acceptance_ratio_Sigma:
            Sigma = Sigma_candidate
            accepted_Sigma += 1
        Sigma_list.append(Sigma)

        # Step 3: propose mu
        """
        mu_candidate = multivariate_normal.rvs(mean=mu, cov=Sigma * proposal_scale_mu)
        inv_Sigma = np.linalg.inv(Sigma)
        posterior_current_mu = np.exp(-0.5 * n * (mu_init - mu).T @ inv_Sigma @ (mu_init - mu))
        posterior_candidate_mu = np.exp(-0.5 * n * (mu_init - mu_candidate).T @ inv_Sigma @ (mu_init - mu_candidate))

        acceptance_ratio_mu = min(1, posterior_candidate_mu / posterior_current_mu)
        if np.random.rand() < acceptance_ratio_mu:
            mu = mu_candidate
            accepted_mu += 1
        mu_list.append(mu)
        """
        if check_mu != 0:  
            mu_candidate = multivariate_normal.rvs(mean=np.mean(returns, axis=0), cov=Sigma/n)
            mu = mu_candidate
        mu_list.append(mu)

    return {
        "mu": mu_list,
        "Sigma": Sigma_list,
        "acc_Sigma": accepted_Sigma / num_samples
    }

def run_MH_parallel(returns, method, num_samples, df, check_mu, num_chains):
    @parfor(range(num_chains))
    def MH_all(i):
        return MH(returns, method, num_samples, df, check_mu)    
    results = MH_all

    mu_chains = [x for r in results for x in r["mu"]]
    Sigma_chains = [x for r in results for x in r["Sigma"]]
    acc_Sigma_list = [r["acc_Sigma"] for r in results]

    print(f"Avg Sigma acceptance rate: {np.mean(acc_Sigma_list):.2f}")
    acceptance = np.mean(acc_Sigma_list)
    return mu_chains, Sigma_chains, round(acceptance,2)

# === Example use ===
if __name__ == "__main__":
    gamma = 50

    n = 50
    k = 15
    check_mu=1
    vol_range = (0.05, 0.2)
    returns, mu, Sigma = generate_simulation_data(k, n, vol_range, check_mu=check_mu)

    # Compute weights and moments
    df = n / 0.9
    num_chains = 5
    num_samples = 1000
    
    """
    "Jeffreys": for Jeffreys
    "Right": for Right-Invariant
    "Rscale": for Scale-Invariant Right.
    "Rrotate": for Rotate-Invariant Right.  (In order to save time: change num_samples)
    """
    start = time.perf_counter()
    mu_Right_sample, S_Right_sample, acc = run_MH_parallel(returns, "Right", num_samples=num_samples, df=df,check_mu=check_mu,num_chains=num_chains)
    S_Right = np.mean(S_Right_sample, axis=0) + np.cov(mu_Right_sample, rowvar=False)
    mu_Right = np.mean(mu_Right_sample, axis=0).reshape(-1, 1)
    weight, expected, variance = true_weights_and_moments(S_Right, mu_Right, gamma)
    end = time.perf_counter()

    print("Optimal portfolio weights w_{R, γ}:")
    print(weight)

    print("\nExpected Return R_{R, γ}:")
    print(expected)

    print("\nVariance V_{R, γ}:")
    print(variance)

    print("Time", round(end-start,3),"seconds")

    # === Plot trace for each element of mu ===
    mu_Right_sample = np.array(mu_Right_sample)
    S_Right_sample = np.array(S_Right_sample)
    # Global fontsize settings (change BIG to make everything larger)

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
        axes_mu[i].plot(mu_Right_sample[:, i])
        axes_mu[i].set_ylabel(f'μ[{i+1}]', fontsize=BIG+1)

    axes_mu[-1].set_xlabel('Iteration', fontsize=BIG+6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'result/{n}_{k}_trace_mu_MH.png', dpi=300)
    plt.close(fig_mu)



    # === Plot trace for diagonal of Σ ===
    fig_var, axes_var = plt.subplots(k, 1, figsize=(10, 2*k), sharex=True)

    fig_var.suptitle(
        f'Σ with acceptance ratio = {acc:.3f} (df = {round(df, 3)})',
        fontsize=BIG+4
    )

    for i in range(k):
        axes_var[i].plot(S_Right_sample[:, i, i])
        axes_var[i].set_ylabel(f'Σ[{i+1},{i+1}]', fontsize=BIG+1)

    axes_var[-1].set_xlabel('Iteration', fontsize=BIG+6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'result/{n}_{k}_trace_sigma_diag_MH.png', dpi=300)
    plt.close(fig_var)