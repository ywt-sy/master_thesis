import numpy as np
import os

rho = 0.6
def generate_simulation_data(k, n, volatility_range):
    # 平均ベクトルμを生成 (μi ~ U(-0.01, 0.01))
    mu = np.random.uniform(-0.01, 0.01, k)

    # 標準偏差のダイアゴナル行列Dを生成
    d = np.random.uniform(volatility_range[0], volatility_range[1], k)
    D = np.diag(d)

    # 相関行列R = (1 - ρ)I + ρJ
    I_k = np.identity(k)
    J_k = np.ones((k, k))
    R = (1 - rho) * I_k + rho * J_k

    # 共分散行列Σを生成 (Σ = D * R * D)
    Sigma = D @ R @ D

    # 資産リターンを多変量正規分布から生成
    returns = np.random.multivariate_normal(mu, Sigma, n)
    return returns, mu, Sigma

def generate_simulation_data0(k, n, volatility_range):
    # 平均ベクトルμを生成 (μi ~ U(-0.01, 0.01))
    mu = np.zeros(k)

    # 標準偏差のダイアゴナル行列Dを生成
    d = np.random.uniform(volatility_range[0], volatility_range[1], k)
    D = np.diag(d)

    # 相関行列R = (1 - ρ)I + ρJ
    I_k = np.identity(k)
    J_k = np.ones((k, k))
    R = (1 - rho) * I_k + rho * J_k

    # 共分散行列Σを生成 (Σ = D * R * D)
    Sigma = D @ R @ D

    # 資産リターンを多変量正規分布から生成
    returns = np.random.multivariate_normal(mu, Sigma, n)
    return returns, mu, Sigma

# === Run script and save example ===
if __name__ == "__main__":
    k = 5  # number of assets
    n = 25*4  # number of periods
    vol_range = (0.05, 0.2)
    output_dir = f"data/simulation"
    os.makedirs(output_dir, exist_ok=True)
    if n == 25:
        returns, mu, Sigma = generate_simulation_data(k, n, vol_range)
        np.save(f"data/simulation/{n}_{k}_returns.npy", returns)
        np.save(f"data/simulation/{n}_{k}_mu_true.npy", mu)
        np.save(f"data/simulation/{n}_{k}_Sigma_true.npy", Sigma)

    else:
        mu = np.load(f"data/simulation/25_{k}_mu_true.npy")
        Sigma = np.load(f"data/simulation/25_{k}_Sigma_true.npy")
        returns = np.random.multivariate_normal(mu, Sigma, n)
        np.save(f"data/simulation/{n}_{k}_mu_true.npy", mu)
        np.save(f"data/simulation/{n}_{k}_Sigma_true.npy", Sigma)
        np.save(f"data/simulation/{n}_{k}_returns.npy", returns)

    print("mu:\n", mu)
    print("\nSigma:\n", Sigma)
    print("\nReturns (first 5 rows):\n", returns[:5])
    print("\nSample size", returns.shape)

    # Save results (optional)
    #np.save(f"data/simulation/{n}_{k}_returns.npy", returns)
    #np.save(f"data/simulation/{n}_{k}_mu.npy", mu)
    #np.save(f"data/simulation/{n}_{k}_Sigma.npy", Sigma)