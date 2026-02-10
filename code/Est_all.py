import numpy as np
import matplotlib.pyplot as plt
import time
from Est0_true import true_weights_and_moments
import numpy as np
from Est1_sample import sample_weights_and_moments
from Est3_J import  Jeffreys_weights_and_moments
from Est4_BL import BL_weights_and_moments
from MHMC import run_MH_parallel
from HMC_SPD import run_HMC_parallel
from HMC_Stan import run_HMC_Stan
import os

def test_Est(data_way,gamma,n,k,df_scale,eps,check_mu,num_chains):
    if data_way == "real":
        check_mu = 1
    # Load previously saved mu and Sigma
    returns = np.load(f"data/{data_way}/{n}_{k}_returns.npy")
    print("\n(n,k)",(n,k))
    
    os.makedirs(f"result/{data_way}/0_mu_Sigma", exist_ok=True)  # Creates the folder if it doesn't exist
    os.makedirs(f"result/{data_way}/1_portfolio", exist_ok=True)  # Creates the folder if it doesn't exist


    if data_way == "simulation":
        # True
        # Load previously saved mu and Sigma
        mu = np.load(f"data/simulation/{n}_{k}_mu_true.npy").reshape(-1, 1)
        Sigma = np.load(f"data/simulation/{n}_{k}_Sigma_true.npy")

        weights, expected_return, variance = true_weights_and_moments(Sigma, mu, gamma)
        print("True: Optimal portfolio weights w_{P, γ}")
        print(weights)
        print("\nTrue: Expected Return R_{P, γ}")
        print(expected_return)
        print("\nTrue: Variance V_{P, γ}")
        print(variance)
        with open(f"result/simulation/1_portfolio/{n}_{k}_true.dat", "w") as f:
            f.write(f"{expected_return} {variance}\n")
        #np.savez("result/simulation/1_portfolio_{n}_{k}_EF/true.npz", mu=mu, Sigma=Sigma)
           
        """
        # Black-Litterman
        r0 = 10
        d0 = 10
        mu = np.load(f"data/simulation/{n}_{k}_mu.npy")
        weights, expected_return, variance = BL_weights_and_moments(returns, mu, Sigma, gamma, r0, d0)
        print("(BL): Optimal portfolio weights w_{BL, γ}")
        print(weights)
        print("\n(BL): Expected Return R_{BL, γ}")
        print(expected_return)
        print("\n(BL): Variance V_{BL, γ}")
        print(variance)
        with open(f"result/simulation/1_portfolio/{n}_{k}_bl.dat", "w") as f:
            f.write(f"{expected_return} {variance}\n")
        with open(f"result/simulation/1_portfolio/{n}_{k}_bl_true.dat", "w") as f:
            f.write(f"{float(weights.T @ mu)} {weights.T @ Sigma @ weights}\n")
        """
    x_sample = np.mean(returns, axis=0).reshape(-1,1)
    S_sample = np.cov(returns, rowvar=False)

    # Sample
    weights, expected_return, variance = sample_weights_and_moments(returns, gamma)
    print("(Sample): Optimal portfolio weights w_{S, γ}")
    print(weights)
    print("\n(Sample): Expected Return R_{S, γ}")
    print(expected_return)
    print("\n(Sample): Variance V_{S, γ}")
    print(variance)
    with open(f"result/{data_way}/1_portfolio/{n}_{k}_sample.dat", "w") as f:
        f.write(f"{expected_return} {variance}\n")
    np.savez(f"result/{data_way}/0_mu_Sigma/{n}_{k}_sample.npz", mu=x_sample, Sigma=S_sample)

    if data_way == "simulation":
        with open(f"result/{data_way}/1_portfolio/{n}_{k}_sample_true.dat", "w") as f:
            f.write(f"{float(weights.T @ mu)} {weights.T @ Sigma @ weights}\n")

    # Jeffreys
    weights, expected_return, variance = Jeffreys_weights_and_moments(returns, gamma)
    print("(Jeffreys): Optimal portfolio weights w_{J, γ}")
    print(weights)
    print("\n(Jeffreys): Expected Return R_{J, γ}")
    print(expected_return)
    print("\n(Jeffreys): Variance V_{J, γ}")
    print(variance)
    with open(f"result/{data_way}/1_portfolio/{n}_{k}_Jeffreys.dat", "w") as f:
        f.write(f"{expected_return} {variance}\n")
    if data_way == "simulation":
        with open(f"result/{data_way}/1_portfolio/{n}_{k}_Jeffreys_true.dat", "w") as f:
            f.write(f"{float(weights.T @ mu)} {weights.T @ Sigma @ weights}\n")

    # HMC_Stan
    # Jst
    mu_Jst_sample, S_Jst_sample = run_HMC_Stan(returns, "Jeffreys", 1000, num_chains=num_chains)
    S_Jst = np.mean(S_Jst_sample, axis=0) + np.cov(mu_Jst_sample, rowvar=False)
    mu_Jst = np.mean(mu_Jst_sample, axis=0).reshape(-1, 1)
    weight, expected, variance = true_weights_and_moments(S_Jst, mu_Jst, gamma)

    print("(Jst): Optimal portfolio weights w_{R, γ}")
    print(weight)
    print("\n(Jst): Expected Return R_{R, γ}")
    print(expected)
    print("\n(Jst): Variance V_{R, γ}")
    print(variance)
    with open(f"result/{data_way}/1_portfolio/{n}_{k}_jst.dat", "w") as f:
        f.write(f"{expected} {variance}\n")
    np.savez(f"result/{data_way}/0_mu_Sigma/{n}_{k}_jst.npz", mu=mu_Jst, Sigma=S_Jst)
    np.savez(f"result/{data_way}/0_mu_Sigma/{n}_{k}_jst_sample.npz", mu=mu_Jst_sample, Sigma=S_Jst_sample)

    if data_way == "simulation":
        with open(f"result/{data_way}/1_portfolio/{n}_{k}_jst_true.dat", "w") as f:
            f.write(f"{float(weight.T @ mu)} {weight.T @ Sigma @ weight}\n")

     # Right
    """  
    "Right": for Right-Invariant
    "Rscale": for Scale-Invariant Right.
    "Rrotate": for Rotate-Invariant Right.  (In order to save time: change num_samples)
    """
    mu_rst_sample, S_rst_sample = run_HMC_Stan(returns, "Right",1000, num_chains=num_chains)
    S_rst = np.mean(S_rst_sample, axis=0) + np.cov(mu_rst_sample, rowvar=False)
    mu_rst = np.mean(mu_rst_sample, axis=0).reshape(-1, 1)
    weight, expected, variance = true_weights_and_moments(S_rst, mu_rst, gamma)

    print("(RMHst): Optimal portfolio weights w_{R, γ}")
    print(weight)
    print("\n(RMHst): Expected Return R_{R, γ}")
    print(expected)
    print("\n(RMHst): Variance V_{R, γ}")
    print(variance)
    with open(f"result/{data_way}/1_portfolio/{n}_{k}_rst.dat", "w") as f:
        f.write(f"{expected} {variance}\n")
    np.savez(f"result/{data_way}/0_mu_Sigma/{n}_{k}_rst.npz", mu=mu_rst, Sigma=S_rst)
    np.savez(f"result/{data_way}/0_mu_Sigma/{n}_{k}_rst_sample.npz", mu=mu_rst_sample, Sigma=S_rst_sample)

    if data_way == "simulation":
        with open(f"result/{data_way}/1_portfolio/{n}_{k}_rst_true.dat", "w") as f:
            f.write(f"{float(weight.T @ mu)} {weight.T @ Sigma @ weight}\n")

    """
    # MCMC
    # JMC
    df = n /df_scale[0]
    mu_JMC_sample, S_JMC_sample = run_MH_parallel(returns, "Jeffreys", 1000, df=df, check_mu=check_mu, num_chains=num_chains)
    S_JMC = np.mean(S_JMC_sample, axis=0) + np.cov(mu_JMC_sample, rowvar=False)
    mu_JMC = np.mean(mu_JMC_sample, axis=0).reshape(-1, 1)
    weight, expected, variance = true_weights_and_moments(S_JMC, mu_JMC, gamma)

    print("(JMC): Optimal portfolio weights w_{R, γ}")
    print(weight)
    print("\n(JMC): Expected Return R_{R, γ}")
    print(expected)
    print("\n(JMC): Variance V_{R, γ}")
    print(variance)
    with open(f"result/{data_way}/1_portfolio/{n}_{k}_jmc.dat", "w") as f:
        f.write(f"{expected} {variance}\n")
    np.savez(f"result/{data_way}/0_mu_Sigma/{n}_{k}_jmc.npz", mu=mu_JMC, Sigma=S_JMC)
    np.savez(f"result/{data_way}/0_mu_Sigma/{n}_{k}_jmc_sample.npz", mu=mu_JMC_sample, Sigma=S_JMC_sample)

    if data_way == "simulation":
        with open(f"result/{data_way}/1_portfolio/{n}_{k}_jmc_true.dat", "w") as f:
            f.write(f"{float(weight.T @ mu)} {weight.T @ Sigma @ weight}\n")

     # Right
    df = n /df_scale[1]
    """ 
    """ 
    "Right": for Right-Invariant
    "Rscale": for Scale-Invariant Right.
    "Rrotate": for Rotate-Invariant Right.  (In order to save time: change num_samples)
    """
    """
    mu_rmc_sample, S_rmc_sample = run_MH_parallel(returns, "Right",1000, df=df,check_mu=check_mu, num_chains=num_chains)
    S_rmc = np.mean(S_rmc_sample, axis=0) + np.cov(mu_rmc_sample, rowvar=False)
    mu_rmc = np.mean(mu_rmc_sample, axis=0).reshape(-1, 1)
    weight, expected, variance = true_weights_and_moments(S_rmc, mu_rmc, gamma)

    print("(RMHMC): Optimal portfolio weights w_{R, γ}")
    print(weight)
    print("\n(RMHMC): Expected Return R_{R, γ}")
    print(expected)
    print("\n(RMHMC): Variance V_{R, γ}")
    print(variance)
    with open(f"result/{data_way}/1_portfolio/{n}_{k}_rmc.dat", "w") as f:
        f.write(f"{expected} {variance}\n")
    np.savez(f"result/{data_way}/0_mu_Sigma/{n}_{k}_rmc.npz", mu=mu_rmc, Sigma=S_rmc)
    np.savez(f"result/{data_way}/0_mu_Sigma/{n}_{k}_rmc_sample.npz", mu=mu_rmc_sample, Sigma=S_rmc_sample)

    if data_way == "simulation":
        with open(f"result/{data_way}/1_portfolio/{n}_{k}_rmc_true.dat", "w") as f:
            f.write(f"{float(weight.T @ mu)} {weight.T @ Sigma @ weight}\n")
    """
    """
    # HMC
    # Jhm
    mu_Jhm_sample, S_Jhm_sample, e_list_Jhm = run_HMC_parallel(returns, "Jeffreys", 1000, eps=eps, T=1, check_mu=check_mu, num_chains=num_chains)
    S_Jhm = np.mean(S_Jhm_sample, axis=0) + np.cov(mu_Jhm_sample, rowvar=False)
    mu_Jhm = np.mean(mu_Jhm_sample, axis=0).reshape(-1, 1)
    weight, expected, variance = true_weights_and_moments(S_Jhm, mu_Jhm, gamma)

    print("(Jhm): Optimal portfolio weights w_{R, γ}")
    print(weight)
    print("\n(Jhm): Expected Return R_{R, γ}")
    print(expected)
    print("\n(Jhm): Variance V_{R, γ}")
    print(variance)
    with open(f"result/{data_way}/1_portfolio/{n}_{k}_jhm.dat", "w") as f:
        f.write(f"{expected} {variance}\n")
    np.savez(f"result/{data_way}/0_mu_Sigma/{n}_{k}_jhm.npz", mu=mu_Jhm, Sigma=S_Jhm)
    np.savez(f"result/{data_way}/0_mu_Sigma/{n}_{k}_jhm_sample.npz", mu=mu_Jhm_sample, Sigma=S_Jhm_sample)
    
    plt.plot(e_list_Jhm)
    plt.xlabel("Number of Sampling")
    plt.ylabel("Energy")
    plt.title("Energy of PDHMC Samples")
     # Save figure
    plt.savefig(f"result/{data_way}/0_mu_Sigma/{n}_{k}_jhm_e_hist.png")
    plt.close()  # <-- Do NOT show the figure

    if data_way == "simulation":
        with open(f"result/{data_way}/1_portfolio/{n}_{k}_jhm_true.dat", "w") as f:
            f.write(f"{float(weight.T @ mu)} {weight.T @ Sigma @ weight}\n")

    # Right
    """
    """  
    "Right": for Right-Invariant
    "Rscale": for Scale-Invariant Right.
    "Rrotate": for Rotate-Invariant Right.  (In order to save time: change num_samples)
    """
    """
    mu_rhm_sample, S_rhm_sample, e_list_rhm = run_HMC_parallel(returns, "Right", 1000, eps=eps, T=1, check_mu=check_mu, num_chains=num_chains)
    S_rhm = np.mean(S_rhm_sample, axis=0) + np.cov(mu_rhm_sample, rowvar=False)
    mu_rhm = np.mean(mu_rhm_sample, axis=0).reshape(-1, 1)
    weight, expected, variance = true_weights_and_moments(S_rhm, mu_rhm, gamma)

    print("(RHMC): Optimal portfolio weights w_{R, γ}")
    print(weight)
    print("\n(RHMC): Expected Return R_{R, γ}")
    print(expected)
    print("\n(RHMC): Variance V_{R, γ}")
    print(variance)
    with open(f"result/{data_way}/1_portfolio/{n}_{k}_rhm.dat", "w") as f:
        f.write(f"{expected} {variance}\n")
    np.savez(f"result/{data_way}/0_mu_Sigma/{n}_{k}_rhm.npz", mu=mu_rhm, Sigma=S_rhm)
    np.savez(f"result/{data_way}/0_mu_Sigma/{n}_{k}_rhm_sample.npz", mu=mu_rhm_sample, Sigma=S_rhm_sample)

    plt.plot(e_list_rhm)
    plt.xlabel("Number of Sampling")
    plt.ylabel("Energy")
    plt.title("Energy of PDHMC Samples")
     # Save figure
    plt.savefig(f"result/{data_way}/0_mu_Sigma/{n}_{k}_rhm_e_hist.png")
    plt.close()  # <-- Do NOT show the figure

    if data_way == "simulation":
        with open(f"result/{data_way}/1_portfolio/{n}_{k}_rhm_true.dat", "w") as f:
            f.write(f"{float(weight.T @ mu)} {weight.T @ Sigma @ weight}\n")
    """
    return "done"

if __name__ == "__main__":
    k = 5
    gamma = 50
    num_chains = 5 ##### Parallel
    for n in [25,50,75,100]:
        if n == 26:
            if k == 5:
                df_scale = [1.15,1.11] #for[JMCMC,Right,Rscale,Rrotate]
            if k == 10:
                df_scale = [1.10,0.95] #for[JMCMC,Right,Rscale,Rrotate]
            if k == 15:
                df_scale = [0.93,0.93] #for[JMCMC,Right,Rscale,Rrotate]

        if n == 52:
            if k == 5:
                df_scale = [1.13,1.11] #for[JMCMC,Right,Rscale,Rrotate]
            if k == 10:
                df_scale = [1.10,1.00] #for[JMCMC,Right,Rscale,Rrotate]
            if k == 15:
                df_scale = [0.93,0.98] #for[JMCMC,Right,Rscale,Rrotate]

        if n == 78:
            if k == 5:
                df_scale = [1.13,1.11] #for[JMCMC,Right,Rscale,Rrotate]
            if k == 10:
                df_scale = [1.1,0.95] #for[JMCMC,Right,Rscale,Rrotate]
            if k == 15:
                df_scale = [0.93,0.9] #for[JMCMC,Right,Rscale,Rrotate]
        else:
            df_scale = [0,0]

        test_Est("simulation",n=n,k=k,gamma=gamma,df_scale=df_scale,eps=1e-4, check_mu=1,num_chains=num_chains)