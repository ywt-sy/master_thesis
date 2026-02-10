import numpy as np

from predict_intervals import Plugin_predict_interval, Jeffreys_predict_interval, MCMC_predict_interval
from Est0_true import true_weights_and_moments
from Est1_sample import sample_weights_and_moments
from Est3_J import  Jeffreys_weights_and_moments
import os

def Predictive_interval_plot(data, method, alpha):
  results = []
  n, k  = data.shape
  os.makedirs(f"result/real/2_EF", exist_ok=True)  # Creates the folder if it doesn't exist

  for gamma in [5, 10, 50]:
        if method  == "sample":
         _, expected_return, variance = sample_weights_and_moments(data, gamma)
         (lower, upper) = Plugin_predict_interval(expected_return, variance, alpha)

        if method == "Jeffreys":
           _, expected_return, variance = Jeffreys_weights_and_moments(data, gamma)
           (lower, upper) = Jeffreys_predict_interval(data, gamma, alpha)

        if method == "jst":
           mu = np.load(f"result/real/0_mu_Sigma/{n}_{k}_jst.npz")["mu"]
           Sigma = np.load(f"result/real/0_mu_Sigma/{n}_{k}_jst.npz")["Sigma"]
           mu_sample = np.load(f"result/real/0_mu_Sigma/{n}_{k}_jst_sample.npz")["mu"]
           S_sample = np.load(f"result/real/0_mu_Sigma/{n}_{k}_jst_sample.npz")["Sigma"]
           _, expected_return, variance = true_weights_and_moments(Sigma, mu, gamma)
           (lower, upper) = MCMC_predict_interval(mu_sample, S_sample, gamma, alpha, num_samples=5)
          
        if method == "rst":
           mu = np.load(f"result/real/0_mu_Sigma/{n}_{k}_rst.npz")["mu"]
           Sigma = np.load(f"result/real/0_mu_Sigma/{n}_{k}_rst.npz")["Sigma"]
           mu_sample = np.load(f"result/real/0_mu_Sigma/{n}_{k}_rst_sample.npz")["mu"]
           S_sample = np.load(f"result/real/0_mu_Sigma/{n}_{k}_rst_sample.npz")["Sigma"]
           _, expected_return, variance = true_weights_and_moments(Sigma, mu, gamma)
           (lower, upper) = MCMC_predict_interval(mu_sample, S_sample, gamma, alpha, num_samples=5)

        if method == "rrotate":
           mu = np.load(f"result/real/0_mu_Sigma/{n}_{k}_rrotate.npz")["mu"]
           Sigma = np.load(f"result/real/0_mu_Sigma/{n}_{k}_rrotate.npz")["Sigma"]
           mu_sample = np.load(f"result/real/0_mu_Sigma/{n}_{k}_rrotate_sample.npz")["mu"]
           S_sample = np.load(f"result/real/0_mu_Sigma/{n}_{k}_rrotate_sample.npz")["Sigma"]
           _, expected_return, variance = true_weights_and_moments(Sigma, mu, gamma)
           (lower, upper) = MCMC_predict_interval(mu_sample, S_sample, gamma, alpha, num_samples=5)

        if method == "rscale":
            mu = np.load(f"result/real/0_mu_Sigma/{n}_{k}_rscale.npz")["mu"]
            Sigma = np.load(f"result/real/0_mu_Sigma/{n}_{k}_rscale.npz")["Sigma"]
            mu_sample = np.load(f"result/real/0_mu_Sigma/{n}_{k}_rscale_sample.npz")["mu"]
            S_sample = np.load(f"result/real/0_mu_Sigma/{n}_{k}_rscale_sample.npz")["Sigma"]
            _, expected_return, variance = true_weights_and_moments(Sigma, mu, gamma)
            (lower, upper) = MCMC_predict_interval(mu_sample, S_sample, gamma, alpha, num_samples=5)
        
        results.append((gamma, expected_return, variance, lower, upper))

  with open(f"result/real/2_EF/{n}_{k}_{method}_error.dat", "w") as f:
    for gamma, R, V, lower, upper in results:
        f.write(f"{gamma} {V} {R} {lower} {upper}\n")
  return results


if __name__ == "__main__":
    n = 26*4
    k = 15
    alpha  = 0.05
    data = np.load(f"data/real/{n}_{k}_returns.npy")
    method_list = ["sample","Jeffreys","jst","rst"] #,"rscale","rrotate"
    for method in method_list:
      results = Predictive_interval_plot(data, method, alpha=alpha)
    print("done")