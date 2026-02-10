import numpy as np 
import os
from Est0_true import true_weights_and_moments
from Est1_sample import sample_weights_and_moments
from Est3_J import  Jeffreys_weights_and_moments

def real_efficient_frontier(Sigma, x, Sigma_true, x_true):
    """
    Efficient frontier under true model:
    x, Sigma are the estimated mean/covariance,
    but risk is evaluated under (mu_true, Sigma_true).
    """

    k = Sigma.shape[0]

    # Ensure column vectors
    x = x.reshape(-1, 1)
    x_true = x_true.reshape(-1, 1)

    Sigma_inv = np.linalg.inv(Sigma)

    # Unit vector
    ones = np.ones((k, 1))

    # Q matrix
    denom = (ones.T @ Sigma_inv @ ones)  # shape (1,1)
    Q = Sigma_inv - (Sigma_inv @ ones @ ones.T @ Sigma_inv) / denom

    # Terms for mean and variance
    term1_R = (ones.T @ Sigma_inv @ x_true) / denom                # (1,1)
    term1_V = (ones.T @ Sigma_inv @ Sigma_true @ Sigma_inv @ ones) / (denom ** 2)

    # gammas: avoid 0 to prevent division by zero
    gammas = np.linspace(60, 0, 100, endpoint=False)

    R_values = []
    V_values = []

    for gamma in gammas:
        # Each expression is (1,1) → cast to scalar with float() or .item()
        R = term1_R + (1.0 / gamma) * (x.T @ Q @ x_true)
        V = term1_V + (1.0 / gamma**2) * (x.T @ Q @ Sigma_true @ Q @ x)

        R_values.append(float(R))  # or R.item()
        V_values.append(float(V))

    return np.array(R_values), np.array(V_values)

def true_efficient_frontier(Sigma, x):
    """
    efficient frontier を計算し、横軸を分散、縦軸を期待リターンでプロットします。

    Parameters:
        Sigma (ndarray): 共分散行列 S
        x (ndarray): 平均ベクトル x
    """

    # 共分散行列の逆行列を計算
    k  = len(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)

    # 単位ベクトル
    ones = np.ones((k, 1))

    # Q_{t-1} 行列の計算
    Q = Sigma_inv - (Sigma_inv @ ones @ ones.T @ Sigma_inv) / (ones.T @ Sigma_inv @ ones)

    # Global Minimum Variance Portfolio (GMV) の期待リターンと分散を計算
    R_GMV = (ones.T @ Sigma_inv @ x) / (ones.T @ Sigma_inv @ ones)
    V_GMV = 1 / (ones.T @ Sigma_inv @ ones)

    # 傾きパラメータ s を計算
    slope = x.T @ Q @ x

    # Efficient Frontier のプロット用データを生成
    R_values = np.linspace(R_GMV, R_GMV + 0.1, 100)  # 期待リターンの範囲を指定
    V_values = V_GMV + (1 / slope) * (R_values - R_GMV)**2

    # 1次元に変換
    R_values = R_values.flatten()
    V_values = V_values.flatten()

    return R_values, V_values

if __name__ == "__main__":
    gamma = 50
    k = 5
    os.makedirs(f"result/simulation/2_EF", exist_ok=True)  # Creates the folder if it doesn't exist


    # true
    for n in [25,50,75,100]:
        mu_true = np.load(f"data/simulation/{n}_{k}_mu_true.npy")
        Sigma_true = np.load(f"data/simulation/{n}_{k}_Sigma_true.npy")
        R_values, V_values = true_efficient_frontier(Sigma_true, mu_true)
        with open(f"result/simulation/2_EF/{n}_{k}_true.dat", "w") as f:
            for V, R in zip(V_values, R_values):
                f.write(f"{V:.6f} {R:.6f}\n")
        _, expected_return, variance = true_weights_and_moments(Sigma_true,mu_true, gamma)
        with open(f"result/simulation/2_EF/{n}_{k}_true_{gamma}.dat", "w") as f:
            f.write(f"{variance:.6f} {expected_return:.6f}\n")


        # sample
        mu = np.load(f"result/simulation/0_mu_Sigma/{n}_{k}_sample.npz")["mu"]
        Sigma = np.load(f"result/simulation/0_mu_Sigma/{n}_{k}_sample.npz")["Sigma"]
        R_values, V_values = true_efficient_frontier(Sigma, mu)
        with open(f"result/simulation/2_EF/{n}_{k}_sample.dat", "w") as f:
            for V, R in zip(V_values, R_values):
                f.write(f"{V:.6f} {R:.6f}\n")
        R_values, V_values = real_efficient_frontier(Sigma, mu, Sigma_true, mu_true)
        with open(f"result/simulation/2_EF/{n}_{k}_sample_real.dat", "w") as f:
            for V, R in zip(V_values, R_values):
                f.write(f"{V:.6f} {R:.6f}\n")
        w, expected_return, variance = true_weights_and_moments(Sigma,mu, gamma)
        with open(f"result/simulation/2_EF/{n}_{k}_sample_{gamma}.dat", "w") as f:
            f.write(f"{variance:.6f} {expected_return:.6f}\n")
            f.write(f"{w.T @ Sigma_true @ w:.6f} {w.T @ mu_true:.6f}\n")

            
        # JMCMC
        mu = np.load(f"result/simulation/0_mu_Sigma/{n}_{k}_jst.npz")["mu"]
        Sigma = np.load(f"result/simulation/0_mu_Sigma/{n}_{k}_jst.npz")["Sigma"]
        R_values, V_values = true_efficient_frontier(Sigma, mu)
        with open(f"result/simulation/2_EF/{n}_{k}_jst.dat", "w") as f:
            for V, R in zip(V_values, R_values):
                f.write(f"{V:.6f} {R:.6f}\n")
        R_values, V_values = real_efficient_frontier(Sigma, mu, Sigma_true, mu_true)
        with open(f"result/simulation/2_EF/{n}_{k}_jst_real.dat", "w") as f:
            for V, R in zip(V_values, R_values):
                f.write(f"{V:.6f} {R:.6f}\n")
        w, expected_return, variance = true_weights_and_moments(Sigma,mu, gamma)
        with open(f"result/simulation/2_EF/{n}_{k}_jst_{gamma}.dat", "w") as f:
            f.write(f"{variance:.6f} {expected_return:.6f}\n")
            f.write(f"{w.T @ Sigma_true @ w:.6f} {w.T @ mu_true:.6f}\n")

        # right
        mu = np.load(f"result/simulation/0_mu_Sigma/{n}_{k}_rst.npz")["mu"]
        Sigma = np.load(f"result/simulation/0_mu_Sigma/{n}_{k}_rst.npz")["Sigma"]
        R_values, V_values = true_efficient_frontier(Sigma, mu)
        with open(f"result/simulation/2_EF/{n}_{k}_rst.dat", "w") as f:
            for V, R in zip(V_values, R_values):
                f.write(f"{V:.6f} {R:.6f}\n")
        R_values, V_values = real_efficient_frontier(Sigma, mu, Sigma_true, mu_true)
        with open(f"result/simulation/2_EF/{n}_{k}_rst_real.dat", "w") as f:
            for V, R in zip(V_values, R_values):
                f.write(f"{V:.6f} {R:.6f}\n")
        w, expected_return, variance = true_weights_and_moments(Sigma,mu, gamma)
        with open(f"result/simulation/2_EF/{n}_{k}_rst_{gamma}.dat", "w") as f:
            f.write(f"{variance:.6f} {expected_return:.6f}\n")
            f.write(f"{w.T @ Sigma_true @ w:.6f} {w.T @ mu_true:.6f}\n")

        """
        # rscale
        mu = np.load(f"result/simulation/0_mu_Sigma/{n}_{k}_rscale.npz")["mu"]
        Sigma = np.load(f"result/simulation/0_mu_Sigma/{n}_{k}_rscale.npz")["Sigma"]
        R_values, V_values = true_efficient_frontier(Sigma, mu)
        with open(f"result/simulation/2_EF/{n}_{k}_rscale.dat", "w") as f:
            for V, R in zip(V_values, R_values):
                f.write(f"{V:.6f} {R:.6f}\n")

        # rrotate
        mu = np.load(f"result/simulation/0_mu_Sigma/{n}_{k}_rrotate.npz")["mu"]
        Sigma = np.load(f"result/simulation/0_mu_Sigma/{n}_{k}_rrotate.npz")["Sigma"]
        R_values, V_values = true_efficient_frontier(Sigma, mu)
        with open(f"result/simulation/2_EF/{n}_{k}_rrotate.dat", "w") as f:
            for V, R in zip(V_values, R_values):
                f.write(f"{V:.6f} {R:.6f}\n")
        
        """