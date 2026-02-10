import numpy as np

def ckn(n, k):
    """c_k,nの計算"""
    return 1 / (n - k - 1) + (2 * n - k - 1) / (n * (n - k - 1) * (n - k - 2))

def Jeffreys_weights_and_moments(returns, gamma):
    """
    式 (8) の重み w_{MV, γ}, 式 (9) と (10) に基づき、期待リターンと分散を計算します。

    Parameters:
        returns: sample
        gamma (float): リスク許容度

    Returns:
        ndarray: 最適ポートフォリオ重み w_{MV, γ}
    """
    # 定数 c_k,n の計算
    (n, k) = returns.shape
    ck_n = ckn(n, k)
    
    # x, Sigma
    x  = np.mean(returns, axis=0).reshape(-1, 1)
    Sigma = np.cov(returns, rowvar=False) * (n-1)

    # 共分散行列の逆行列を計算
    Sigma_inv = np.linalg.inv(Sigma)

    # 単位ベクトル
    ones = np.ones((k, 1))

    # Q_{t-1} 行列の計算
    Q = Sigma_inv - (Sigma_inv @ ones @ ones.T @ Sigma_inv) / (ones.T @ Sigma_inv @ ones)

    # 重みの計算
    first_term = (Sigma_inv @ ones) / (ones.T @ Sigma_inv @ ones)
    second_term = (1 / (gamma * ck_n)) * (Q @ x)
    w_J_gamma = first_term + second_term

    # 期待リターン R_{MV, γ} の計算 (式9)
    first_term_return = (ones.T @ Sigma_inv @ x) / (ones.T @ Sigma_inv @ ones)
    second_term_return = (1 / (gamma * ck_n)) * (x.T @ Q @ x)
    R_J_gamma = first_term_return + second_term_return

    # 分散 V_{MV, γ} の計算 (式10)
    first_term_variance = ck_n / (ones.T @ Sigma_inv @ ones)
    second_term_variance = (1 / (ck_n * gamma ** 2)) * (x.T @ Q @ x)
    V_J_gamma = first_term_variance + second_term_variance

    return w_J_gamma.flatten(), float(R_J_gamma), float(V_J_gamma)

# === Example use ===
if __name__ == "__main__":
    gamma = 50

    # Load previously saved mu and Sigma
    returns = np.load("simulated_returns.npy")
    print("\n(n,k)",returns.shape)

    # Compute weights and moments
    weights, expected_return, variance = Jeffreys_weights_and_moments(returns, gamma)

    print("Optimal portfolio weights w_{J, γ}:")
    print(weights)

    print("\nExpected Return R_{J, γ}:")
    print(expected_return)

    print("\nVariance V_{J, γ}:")
    print(variance)