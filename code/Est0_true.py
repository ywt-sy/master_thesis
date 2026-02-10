import numpy as np

def true_weights_and_moments(Sigma, mu, gamma):
    """
    式 (28) と (29) に基づき、真値を用いて重み、期待リターン、分散を計算します。

    Parameters:
        Sigma (ndarray): 真の共分散行列 Σ
        mu (ndarray): 真の平均リターンベクトル μ
        gamma (float): リスク許容度

    Returns:
        ndarray: 最適ポートフォリオ重み w_{P, γ}
        float: 期待リターン R_{P, γ}
        float: 分散 V_{P, γ}
    """
    # 共分散行列の逆行列を計算
    Sigma_inv = np.linalg.inv(Sigma)

    # 単位ベクトル
    ones = np.ones((mu.shape[0], 1))

    # 重み計算 (式28)
    R = Sigma_inv - (Sigma_inv @ ones @ ones.T @ Sigma_inv) / (ones.T @ Sigma_inv @ ones)
    first_term_weight = Sigma_inv @ ones / (ones.T @ Sigma_inv @ ones)
    second_term_weight = (1 / gamma) * (R @ mu)
    w_p_gamma = first_term_weight + second_term_weight

    # 期待リターン (式29)
    first_term_return = (ones.T @ Sigma_inv @ mu) / (ones.T @ Sigma_inv @ ones)
    second_term_return = (1 / gamma) * (mu.T @ R @ mu)
    R_p_gamma = first_term_return + second_term_return

    # 分散 (式29)
    first_term_variance = 1 / (ones.T @ Sigma_inv @ ones)
    second_term_variance = (1 / (gamma ** 2)) * (mu.T @ R @ mu)
    V_p_gamma = first_term_variance + second_term_variance

    return w_p_gamma.flatten(), float(R_p_gamma), float(V_p_gamma)

# === Example use ===
if __name__ == "__main__":
    gamma = 50
    n = 50
    k = 5

    # Load previously saved mu and Sigma
    mu = np.load(f"data/simulation/{n}_{k}_mu.npy").reshape(-1, 1)
    Sigma = np.load(f"data/simulation/{n}_{k}_Sigma.npy")

    # Compute weights and moments
    weights, expected_return, variance = true_weights_and_moments(Sigma, mu, gamma)

    print("Optimal portfolio weights w_{P, γ}:")
    print(weights)

    print("\nExpected Return R_{P, γ}:")
    print(expected_return)

    print("\nVariance V_{P, γ}:")
    print(variance)