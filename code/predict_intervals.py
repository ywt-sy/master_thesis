from datetime import datetime
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
from Est0_true import true_weights_and_moments
from Est1_sample import sample_weights_and_moments
from Est3_J import  ckn, Jeffreys_weights_and_moments
from MCMCtest import run_MC_parallel, log_mixture_gaussian_pdf
import os


# Estimative
def Plugin_predict_interval(expected_return, variance, alpha):
    """
    Calculate the prediction
     interval for a sample-based expected return.

    Parameters:
        expected_return (float): The expected return (mean).
        variance (float): The expected variance.
        alpha (float): Significance level for the prediction interval (default: 0.05 for 95% interval).
        num_simulations (int): Number of simulations for the distribution (default: 10,000).

    Returns:
        tuple: Lower and upper bounds of the prediction interval.
    """

    # Compute the z-scores for the confidence interval
    z_lower = norm.ppf(alpha / 2)
    z_upper = norm.ppf(1 - alpha / 2)

    # Calculate the confidence interval
    lower_quantile = expected_return + z_lower * np.sqrt(variance)
    upper_quantile = expected_return + z_upper * np.sqrt(variance)

    return (lower_quantile, upper_quantile)

# Jeffterys
def Jeffreys_predict_interval(returns, gamma, alpha):
  (n, k) = returns.shape
  # x, Sigma
  x  = np.mean(returns, axis=0).reshape(-1, 1)
  Sigma = np.cov(returns, rowvar=False) * (n-1)

  # step a, b:
  # Calculate the portfolio point for gamma = 50
  weights, _,_ = Jeffreys_weights_and_moments(returns, gamma)

  # step c:
  # Simulate posterior predictive returns and compute quantiles for the interval
  portfolio_returns = []  # Placeholder for storing simulated returns

  # Run simulation as per Algorithm 1 to generate posterior predictive returns for gamma = 50
  B = 10000  # Number of simulations
  for b in range(B):
      # Generate τ1 and τ2 from Student's t-distribution
      tau1_b = stats.t.rvs(df=n - k)
      tau2_b = stats.t.rvs(df=n - k + 1)

      # Compute posterior predictive return for the given sample
      term1 = np.dot(weights.T, x)
      term2 = np.sqrt(np.dot(np.dot(weights.T, Sigma), weights))
      adjustment_factor = (tau1_b / np.sqrt(n * (n - k))) + np.sqrt(1 + (tau1_b**2) / (n - k)) * (tau2_b / np.sqrt(n - k + 1))
      X_p_t_b = term1 + term2 * adjustment_factor

      # Store the computed return
      portfolio_returns.append(X_p_t_b)  # Extract scalar from 1x1 matrix

  # step d:
  # Calculate quantiles for the confidence interval
  lower_quantile = np.quantile(portfolio_returns, alpha / 2)
  upper_quantile = np.quantile(portfolio_returns, 1 - alpha / 2)
  return (lower_quantile,upper_quantile)

# MCMC
def MCMC_predict_interval(mu_Right_sample, S_Right_sample, gamma, alpha, num_samples=5):
  # step a, b:
  S_Right = np.mean(S_Right_sample, axis=0) + np.cov(mu_Right_sample, rowvar=False)
  mu_Right = np.mean(mu_Right_sample, axis=0).reshape(-1, 1)

  weights, _,_ = true_weights_and_moments(S_Right, mu_Right, gamma)
  # step c:
  # Simulate posterior predictive returns and compute quantiles for the interval
  portfolio_returns = []  # Placeholder for storing simulated returns

  # Run simulation as per Algorithm 1 to generate posterior predictive returns for gamma = 50
  for b in range(len(mu_Right_sample)):
      mu = np.dot(weights.T, mu_Right_sample[b])
      Sigma = np.dot(weights.T, np.dot(S_Right_sample[b], weights))

      # Generate num_samples from N(mu_b, Sigma_b)
      X_p_t_b = np.random.normal(mu, np.sqrt(Sigma), num_samples)

      # Store the computed return
      portfolio_returns.append(list(X_p_t_b))  # Extract scalar from 1x1 matrix

  # step d:
  # Calculate quantiles for the confidence interval
  lower_quantile = np.quantile(portfolio_returns, alpha / 2)
  upper_quantile = np.quantile(portfolio_returns, 1 - alpha / 2)
  return (lower_quantile,upper_quantile)

# Define a function to calculate weights, expected return, and variance as per the user-provided date
def portfolio_at_date(method, target_date_str, returns_data, gamma, n_weeks, num_chains, df, eps, alpha=0.05):
    # Parse the target date
    target_date = pd.to_datetime(target_date_str)
    # Combine returns data into a single DataFrame
    returns_df = pd.DataFrame(returns_data)

    # Get the index of the specified target date
    if target_date not in returns_df.index:
        raise ValueError("The specified date is not present in the data.")

    # Calculate the past n (n_weeks) weekly returns before the target date
    target_index = returns_df.index.get_loc(target_date)
    if target_index < n_weeks:
        raise ValueError("Not enough historical data before the specified date to calculate weights.")

    # Use the past n_weeks weeks of data
    past_returns = returns_df.iloc[target_index - n_weeks:target_index]

    # Convert to numpy array for calculation
    returns = past_returns.values

    if method == 'Sample':
        # サンプル値を使用して重み、期待リターン、分散を計算
        mu_sample = np.mean(returns, axis=0).reshape(1, -1)
        S_sample = np.cov(returns, rowvar=False)[None, :, :] 
        weights, expected_return, variance = sample_weights_and_moments(returns, gamma)

    elif method == 'Jeffreys':
      # Jeffreys
        mu_sample = np.mean(returns, axis=0).reshape(1, -1)
        S_sample = n_weeks * ckn(n_weeks, k) * np.cov(returns, rowvar=False)[None, :, :]
        weights, expected_return, variance = Jeffreys_weights_and_moments(returns, gamma)
    
    else:
        # MC
        mu_sample, S_sample, _ = run_MC_parallel(returns, method, 1000, df=df, eps=eps,num_chains=num_chains, check_mu=1)
        S = np.mean(S_sample, axis=0) + np.cov(mu_sample, rowvar=False)
        mu = np.mean(mu_sample, axis=0).reshape(-1, 1)
        weights, expected_return, variance = true_weights_and_moments(S, mu, gamma)

    # Calculate the portfolio
    # Calculate the return for the target date using the computed weights
    target_returns = returns_df.iloc[target_index].values
    portfolio_return = np.dot(weights, target_returns)

    w = weights.reshape(-1, 1)  
    Y = np.asarray(portfolio_return.reshape(-1, 1), float)
    mus = np.asarray(mu_sample @ w, float)
    Sigmas = np.einsum('ia,sab,bj->sij', w.T, S_sample, w)
    logq = log_mixture_gaussian_pdf(Y, mus, Sigmas)
    print(np.mean(returns, axis=0).reshape(1, -1))
    print(np.cov(returns, rowvar=False))

    if method == 'Sample':
      (lower_quantile, upper_quantile) = Plugin_predict_interval(expected_return, variance, alpha)

    elif method in ['Jeffreys', "jst", "jst"]:
      # Simulate posterior predictive returns
      (lower_quantile,upper_quantile) = Jeffreys_predict_interval(returns, gamma, alpha)


    else:
      (lower_quantile,upper_quantile) = MCMC_predict_interval(mu_sample, S_sample, gamma, alpha, num_samples=5)

    return weights, expected_return, variance, portfolio_return, (lower_quantile, upper_quantile),float(logq)

def weekly_portfolio_returns(method, start_date, end_date, returns_data, gamma, n_weeks, num_chains, df, eps, alpha):
    """
    Calculate weekly expected returns, actual returns, and standard deviations for a portfolio.

    Parameters:
    - start_date (str): Start date of the calculation period (YYYY-MM-DD).
    - end_date (str): End date of the calculation period (YYYY-MM-DD).
    - returns_data (DataFrame): Historical returns data used for calculations.
    - gamma (float): Risk tolerance parameter.
    - k (int): Number of assets in the portfolio.
    - method (str): Method for calculating portfolio weights ('Bayesian', 'Sample', etc.).

    Returns:
    - dict: A dictionary with keys 'dates', 'expected_returns', 'actual_returns', and 'std_devs'.
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    weekly_dates = pd.date_range(start=start_date, end=end_date, freq='W-SUN')
    weekly_dates_str = [date.strftime("%Y-%m-%d") for date in weekly_dates]

    # Initialize lists to store expected returns, actual returns, and standard deviations
    expected_returns = []
    actual_returns = []
    std_devs = []
    total_logq = 0.0
    loss_L = []

    for date_str in weekly_dates_str:
        try:
            # Calculate expected return, variance, weights, and actual portfolio return
            weights, expected_return, variance, portfolio_return, (lower_quantile, upper_quantile),logq = portfolio_at_date(
                method, date_str, returns_data, gamma, n_weeks, num_chains, df, eps, alpha
            )

            # Store results
            expected_returns.append(expected_return)
            actual_returns.append(portfolio_return)
            std_devs.append(((lower_quantile, upper_quantile)))  # Convert variance to standard deviation

            # q(y | x^{(ℓ)}) at each y^{(j)}: Gaussian mixture  
            with open(f"result/real/{method}_lossfunc.dat", "a") as f:     
                f.write(f"{(n,k)} {float(logq)}\n")
    
            loss_L.append(logq)
            total_logq += np.sum(logq)

        except ValueError as e:
            # Handle errors and log
            print(f"Skipping date {date_str} due to error: {e}")
            expected_returns.append(None)
            actual_returns.append(None)
            std_devs.append((None, None))

    # Filter out None values for plotting
    valid_indices = [i for i, val in enumerate(expected_returns) if val is not None]
    plot_dates = [weekly_dates[i] for i in valid_indices]
    plot_expected_returns = [expected_returns[i] for i in valid_indices]
    plot_actual_returns = [actual_returns[i] for i in valid_indices]
    plot_std_devs = [std_devs[i] for i in valid_indices]
    with open(f"result/real/{n}_{k}_lossfunc.dat", "a") as f:
            f.write(f"{method} {(n,k,len(loss_L))} {float(np.mean(loss_L))} {float(np.sqrt(np.cov(loss_L)/len(loss_L)))}\n")

    return {
        "dates": plot_dates,
        "expected_returns": plot_expected_returns,
        "actual_returns": plot_actual_returns,
        "posterior_intervals": plot_std_devs,
    }

if __name__ == "__main__":
    # === Parameters ===
    start_date = "2023-11-02"
    end_date   = "2025-11-02"

    # Model/Simulation settings
    gamma             = 50
    n_chains          = 5
    alpha             = 0.05
    k                 = 5
    n_weeks           = 13
    n                 = 52 * 2 + n_weeks
    method_number     = 0 # 4 for "rscale", 5 for "rrotate"
    method_list = ['Sample',"Jeffreys","jst","rst","rscale"]
    """
    k                 = 15 # 5,10,15
    n_weeks           = 26*4
    n                 = 52 * 2 + n_weeks
    """
    for k in [3]:
        for i in [3]:
            method = method_list[i]
            for data_num in [0]:
                print(method, data_num)
                if n_weeks == 26:
                    if k == 5:
                        df_scale = [1.15,1.11] #for[JMCMC,Right,Rscale,Rrotate]
                    if k == 10:
                        df_scale = [1.10,0.95] #for[JMCMC,Right,Rscale,Rrotate]
                    if k == 15:
                        df_scale = [0.93,0.93] #for[JMCMC,Right,Rscale,Rrotate]

                if n_weeks == 52:
                    if k == 5:
                        df_scale = [1.13,1.11] #for[JMCMC,Right,Rscale,Rrotate]
                    if k == 10:
                        df_scale = [1.10,1.00] #for[JMCMC,Right,Rscale,Rrotate]
                    if k == 15:
                        df_scale = [0.93,0.98] #for[JMCMC,Right,Rscale,Rrotate]

                if n_weeks == 78:
                    if k == 5:
                        df_scale = [1.13,1.11] #for[JMCMC,Right,Rscale,Rrotate]
                    if k == 10:
                        df_scale = [1.08,0.95] #for[JMCMC,Right,Rscale,Rrotate]
                    if k == 15:
                        df_scale = [0.95,0.9] #for[JMCMC,Right,Rscale,Rrotate]
                else:
                    df_scale=[1,1]

                # === Load Data ===
                returns_path = f"data/real/{n}_{k}_returns_{data_num+1}.pkl"
                returns_data = pd.read_pickle(returns_path)

                # === Create Output Directory ===
                output_dir = f"result/real/4_predict_years"
                os.makedirs(output_dir, exist_ok=True)

                # === Run Prediction ===
                df = n_weeks / df_scale[method_number//2-1]
                eps = 1e-4

                result = weekly_portfolio_returns(
                    method=method,
                    start_date=start_date,
                    end_date=end_date,
                    returns_data=returns_data,
                    gamma=gamma,
                    n_weeks=n_weeks,
                    num_chains=n_chains,
                    df=df,
                    eps=eps,
                    alpha=alpha
                )

                # === Save Results ===
                output_file = os.path.join(output_dir, f"{n_weeks}_{k}_{method}{data_num+1}.dat")
                with open(output_file, "w") as f:
                    for date, exp_ret, act_ret, (low, up) in zip(
                        result["dates"],
                        result["expected_returns"],
                        result["actual_returns"],
                        result["posterior_intervals"]
                    ):
                        f.write(f"{date.strftime('%Y-%m-%d')} {exp_ret:.6f} {act_ret:.6f} {low:.6f} {up:.6f}\n")
