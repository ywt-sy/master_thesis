import numpy as np
from Est0_true import true_weights_and_moments

def sample_weights_and_moments(returns, gamma):
    """
    式 (28) と (29) に基づき、真値を用いて重み、期待リターン、分散を計算します。

    Parameters:
        returns: sample
        gamma (float): リスク許容度

    Returns:
        ndarray: 最適ポートフォリオ重み w_{P, γ}
        float: 期待リターン R_{P, γ}
        float: 分散 V_{P, γ}
    """
    x_sample = np.mean(returns, axis=0).reshape(-1, 1)
    S_sample = np.cov(returns, rowvar=False)

    # Save results
    #np.save("mu_sample.npy", x_sample)
    #np.save("Sigma_sample.npy", S_sample)

    weights, expected_return, variance = true_weights_and_moments(S_sample, x_sample, gamma)
    return weights, expected_return, variance

# === Example use ===
if __name__ == "__main__":
    gamma = 50

    # Load previously saved mu and Sigma
    returns = np.load("stock_returns.npy")
    print("\n(n,k)",returns.shape)

    # Compute weights and moments
    weights, expected_return, variance = sample_weights_and_moments(returns, gamma)

    print("Optimal portfolio weights w_{S, γ}:")
    print(weights)

    print("\nExpected Return R_{S, γ}:")
    print(expected_return)

    print("\nVariance V_{S, γ}:")
    print(variance)