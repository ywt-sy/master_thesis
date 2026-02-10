import numpy as np 
import os

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
    n = 26*4
    k = 10
    os.makedirs(f"result/real/2_EF", exist_ok=True)  # Creates the folder if it doesn't exist


    # true
    """
    mu = np.load("data/simulation/mu.npy")
    Sigma = np.load("data/simulation/Sigma.npy")
    R_values, V_values = true_2_efficient_frontier(Sigma, mu)
    with open("result/simulation/2_EF/true.dat", "w") as f:
        for V, R in zip(V_values, R_values):
            f.write(f"{V:.6f} {R:.6f}\n")
    """

    # sample
    mu = np.load(f"result/real/0_mu_Sigma/{n}_{k}_sample.npz")["mu"]
    Sigma = np.load(f"result/real/0_mu_Sigma/{n}_{k}_sample.npz")["Sigma"]
    R_values, V_values = true_efficient_frontier(Sigma, mu)
    with open(f"result/real/2_EF/{n}_{k}_sample.dat", "w") as f:
        for V, R in zip(V_values, R_values):
            f.write(f"{V:.6f} {R:.6f}\n")
          
    # JMCMC
    mu = np.load(f"result/real/0_mu_Sigma/{n}_{k}_jst.npz")["mu"]
    Sigma = np.load(f"result/real/0_mu_Sigma/{n}_{k}_jst.npz")["Sigma"]
    R_values, V_values = true_efficient_frontier(Sigma, mu)
    with open(f"result/real/2_EF/{n}_{k}_jst.dat", "w") as f:
        for V, R in zip(V_values, R_values):
            f.write(f"{V:.6f} {R:.6f}\n")
    
    # right
    mu = np.load(f"result/real/0_mu_Sigma/{n}_{k}_rst.npz")["mu"]
    Sigma = np.load(f"result/real/0_mu_Sigma/{n}_{k}_rst.npz")["Sigma"]
    R_values, V_values = true_efficient_frontier(Sigma, mu)
    with open(f"result/real/2_EF/{n}_{k}_rst.dat", "w") as f:
        for V, R in zip(V_values, R_values):
            f.write(f"{V:.6f} {R:.6f}\n")

    """
    # rscale
    mu = np.load(f"result/real/0_mu_Sigma/{n}_{k}_rscale.npz")["mu"]
    Sigma = np.load(f"result/real/0_mu_Sigma/{n}_{k}_rscale.npz")["Sigma"]
    R_values, V_values = true_efficient_frontier(Sigma, mu)
    with open(f"result/real/2_EF/{n}_{k}_rscale.dat", "w") as f:
        for V, R in zip(V_values, R_values):
            f.write(f"{V:.6f} {R:.6f}\n")

    # rrotate
    mu = np.load(f"result/real/0_mu_Sigma/{n}_{k}_rrotate.npz")["mu"]
    Sigma = np.load(f"result/real/0_mu_Sigma/{n}_{k}_rrotate.npz")["Sigma"]
    R_values, V_values = true_efficient_frontier(Sigma, mu)
    with open(f"result/real/2_EF/{n}_{k}_rrotate.dat", "w") as f:
        for V, R in zip(V_values, R_values):
            f.write(f"{V:.6f} {R:.6f}\n")
    
    """