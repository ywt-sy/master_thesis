import numpy as np
from Est0_true import true_weights_and_moments

# === Load saved mu and Sigma ===
# === Black-Litterman prior generator ===
# Black-Littermanモデル用の事前パラメータ
def generate_black_litterman_prior(mu, Sigma, k):
    # 事前平均m0の生成 (m0 = μ + 0.5 * ε, εi ~ U(-0.001, 0.001))
    epsilon = np.random.uniform(-0.01, 0.01, k)
    m0 = mu + 0.5 * epsilon

    # 事前共分散S0の生成 (S0 = Σ + 0.5 * Λ, δi ~ U(0.001, 0.005))
    delta = np.random.uniform(0.001, 0.005, k)
    Lambda = np.diag(delta ** 2)
    S0 = Sigma + 0.5 * Lambda

    return m0, S0

def q_kn(n, k, r0, d0):
    """q_k,nの計算"""
    return 1 / (n + d0 - 2 * k - 1) + (2 * n + r0 + d0 - 2 * k - 1) / ((n + r0) * (n + d0 - 2 * k - 1) * (n + d0 - 2 * k - 2))

def Black_Litterman_posterior(returns, mu, Sigma, r0):
    """
    Black-Littermanモデルにおける事後平均リターンと事後共分散行列を計算します。

    Parameters:
        returns: sample
        r0 (float): Black-Littermanの精度パラメータ
        d0 (float): Black-Littermanの精度パラメータ

    Returns:
        ndarray: 事後平均リターン x_l (k, 1)
        ndarray: 事後共分散行列 S_l (k, k)
    """
    (n, k) = returns.shape
    m0, S0 = generate_black_litterman_prior(mu, Sigma, k)
    x_sample  = np.mean(returns, axis=0)
    S_sample = np.cov(returns, rowvar=False) * (n-1)

    x_l = (n * x_sample + r0 * m0) / (n + r0)
    S_l = S_sample + S0 + (n * r0) * ((m0 - x_l) @ (m0 - x_l).T) / (n + r0)
    return x_l.reshape(-1,1), S_l

def BL_weights_and_moments(returns, mu, Sigma, gamma, r0, d0):
    """
    式 (21) に基づき、Black-Littermanモデルの重みを, 式 (22) と (23) に基づき、Black-Littermanモデルの期待リターンと分散を計算します。

    Parameters:
        returns: sample
        gamma (float): リスク許容度
        r0 (float): Black-Littermanの精度パラメータ
        d0 (float): Black-Littermanの精度パラメータ

    Returns:
        ndarray: Black-Littermanモデルの最適ポートフォリオ重み w_{BL} (k,)
    """
    (n, k) = returns.shape
    x_l, S_l = Black_Litterman_posterior(returns, mu, Sigma, r0)

    # 定数 q_k,n の計算
    qkn = q_kn(n, k, r0, d0)

    weights, expected_return, variance = true_weights_and_moments(qkn * S_l, x_l, gamma)

    return weights, expected_return, variance

# === Example use ===
if __name__ == "__main__":
    gamma = 50
    r0 =  10
    d0 = 10

    mu = np.load("data/simulation/50_5/mu.npy")
    print(mu.shape)
    Sigma = np.load("data/simulation/50_5/Sigma.npy")

    # Load previously saved mu and Sigma
    returns = np.load("data/simulation/50_5/returns.npy")
    print("\n(n,k)",returns.shape)

    # Compute weights and moments
    weights, expected_return, variance = BL_weights_and_moments(returns, mu, Sigma, gamma, r0, d0)

    print("Optimal portfolio weights w_{BL, γ}:")
    print(weights)

    print("\nExpected Return R_{BL, γ}:")
    print(expected_return)

    print("\nVariance V_{BL, γ}:")
    print(variance)