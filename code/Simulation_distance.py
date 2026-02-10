import numpy as np
import time
import os
from parfor import parfor
from collections import defaultdict
from data_simulation import generate_simulation_data
from Est0_true import true_weights_and_moments
from Est1_sample import sample_weights_and_moments
from Est3_J import  Jeffreys_weights_and_moments
from Est4_BL import BL_weights_and_moments
from MCMCtest import run_MC_parallel

def simu_distance_compare(n_simulation, n, k, gamma, num_chains, df_scale, eps, check_mu):
    avg_dist_P = defaultdict(list)
    avg_dist_true = defaultdict(list)
    vol_range = (0.05, 0.2)

    for sim in range(n_simulation):
        returns, mu, Sigma = generate_simulation_data(k, n, vol_range, check_mu)
        mu = mu.reshape(-1,1)
        print(mu.shape)
        weights, expected_return, variance = true_weights_and_moments(Sigma, mu, gamma)
        true_func = float(weights.T @ mu - (gamma / 2) * (weights.T @ Sigma @ weights))
        print(true_func)

        # --- Sample ---
        x_sample = np.mean(returns, axis=0).reshape(-1, 1)
        S_sample = np.cov(returns, rowvar=False)
        weights, expected_return, variance = sample_weights_and_moments(returns, gamma)
        sample_func = float(expected_return - (gamma / 2) * variance)
        sample_func_true = float(weights.T @ mu - (gamma / 2) * (weights.T @ Sigma @ weights))
        print("Sample", sample_func,sample_func_true)

        # --- Jeffreys ---
        weights, expected_return, variance = Jeffreys_weights_and_moments(returns, gamma)
        Jeffreys_func = float(expected_return - (gamma / 2) * variance)
        Jeffreys_func_true = float(weights.T @ mu - (gamma / 2) * (weights.T @ Sigma @ weights))
        print("Jeffreys", Jeffreys_func,Jeffreys_func_true)

        # --- JMC ---
        df = n / df_scale[0]
        mu_JMC_sample, S_JMC_sample,_ = run_MC_parallel(returns, "jst", 1000,
                                                      num_chains=5, df=df, eps=1e-4,check_mu=check_mu)
        S_JMC = np.mean(S_JMC_sample, axis=0) + np.cov(mu_JMC_sample, rowvar=False)
        mu_JMC = np.mean(mu_JMC_sample, axis=0).reshape(-1, 1)
        weights, expected, variance = true_weights_and_moments(S_JMC, mu_JMC, gamma)
        jmc_func = float(expected - (gamma / 2) * variance)
        jmc_func_true = float(weights.T @ mu - (gamma / 2) * (weights.T @ Sigma @ weights))
        print("J(MC)", jmc_func,jmc_func_true)

        # --- rst ---
        df = n / df_scale[1]
        mu_rmc_sample, S_rmc_sample,_ = run_MC_parallel(returns, "rst", 1000,
                                                      num_chains=5, df=df, eps=1e-4,check_mu=check_mu)
        S_rmc = np.mean(S_rmc_sample, axis=0) + np.cov(mu_rmc_sample, rowvar=False)
        mu_rmc = np.mean(mu_rmc_sample, axis=0).reshape(-1, 1)
        weights, expected, variance = true_weights_and_moments(S_rmc, mu_rmc, gamma)
        rmc_func = float(expected - (gamma / 2) * variance)
        rmc_func_true = float(weights.T @ mu - (gamma / 2) * (weights.T @ Sigma @ weights))
        print("Right(MC)", rmc_func,rmc_func_true)

        # Record distances
        def record(method, f_p, f_true_est):
            avg_dist_P[method].append(f_p - true_func)
            avg_dist_true[method].append(f_true_est - true_func)

        record("Sample", sample_func, sample_func_true)
        record("Jeffreys", Jeffreys_func, Jeffreys_func_true)
        record("Jeffreys(MC)", jmc_func, jmc_func_true)
        record("Right(MC)", rmc_func, rmc_func_true)

    # Average distances over all simulations
    dist_P = {k: np.mean(v) for k, v in avg_dist_P.items()}
    dist_true = {k: np.mean(v) for k, v in avg_dist_true.items()}

    # Save
    print("\nAverage f_{P,γ} - f_{true}:")
    for method, val in dist_P.items():
        print(f"{method:10s}: {val:.6f}")

    print("\nAverage f_{true est,γ} - f_{true}:")
    for method, val in dist_true.items():
        print(f"{method:10s}: {val:.6f}")

    # Define output directory
    output_dir = f"result/simulation"
    os.makedirs(output_dir, exist_ok=True)
    # Define output file path
    output_dat = os.path.join(output_dir, f"distance_{n}_{k}.dat")

    # Save to .dat file (LaTeX-friendly)
    with open(output_dat, "w") as f:
        f.write("method         f_P_gamma     f_true_gamma\n")
        for method in sorted(dist_P.keys()):
            line = f"{method:<14s} {dist_P[method]:.6f}    {dist_true[method]:.6f}"
            f.write(line + "\n")

    print(f"\n✅ Distance summary saved to {output_dat}")
    return dist_P, dist_true

if __name__ == "__main__":
    n = 50           # 候補: 50, 75, 100
    k = 5           # 候補: 5, 10, 15

    gamma = 50
    num_chains = 5    # MCMCの並列数
    n_simulation = 1   # シミュレーションの回数
    eps = 1e-4
    check_mu = 1

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
    dist_P, dist_true = simu_distance_compare(n_simulation, n, k, gamma, num_chains,df_scale,eps,check_mu)
    end = time.perf_counter()

    print("Time", round(end-start,3),"seconds")