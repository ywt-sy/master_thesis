import numpy as np

def generate_simulation_data(k, n, volatility_range,check_mu=0):
    # 平均ベクトルμを生成 (μi ~ U(-0.01, 0.01))
    rho = 0.6
    if check_mu == 0:
        mu = np.zeros(k)
    else:
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
    Sigma = (Sigma + Sigma.T)/2

    # 資産リターンを多変量正規分布から生成
    returns = np.random.multivariate_normal(mu, Sigma, n)
    return returns, mu, Sigma
