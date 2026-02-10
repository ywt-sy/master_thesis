functions {
  // ---------------------------------------------------------------
  // General-k scale-right prior using Cholesky factor L:
  // For each subset I, Sigma_I = Sigma[I,I] = L[I,] * L[I,]' (rows of L)
  // log π(Σ) = - Σ_i (1/C(k,i)) Σ_{|I|=i} log |Σ_I|
  // ---------------------------------------------------------------
  real scale_right_prior_log_chol(matrix L, int k) {
    real log_val = 0;

    int max_mask = 1;
    for (i in 1:k) max_mask *= 2;  // 2^k

    for (i in 1:k) {
      real sum_log = 0;
      int m_count = 0;

      for (mask in 1:(max_mask - 1)) {
        array[k] int idx;
        int idx_len = 0;
        int tmp = mask;

        // decode subset indices from bitmask
        for (j in 1:k) {
          if (tmp % 2 == 1) {
            idx_len += 1;
            idx[idx_len] = j;
          }
          tmp = tmp / 2;
        }

        if (idx_len == i) {
          // rows of L corresponding to subset I
          matrix[i, k] Li = L[idx[1:i], 1:k];

          // Sigma_I = Li * Li'  (i x i SPD)
          matrix[i, i] Sigma_sub = tcrossprod(Li);

          // log |Sigma_I|
          real log_det_sub = log_determinant(Sigma_sub);

          sum_log += -log_det_sub;
          m_count += 1;
        }
      }

      log_val += sum_log / m_count;
    }

    return log_val;
  }
}

data {
  int<lower=0> n;
  int<lower=1> k;
  array[n] vector[k] y;
}

parameters {
  vector[k] mu;                 // mean vector
  cholesky_factor_cov[k] L;   // Sigma = L * L'
}

transformed parameters {
  // If you want the full covariance matrix explicitly:
  cov_matrix[k] Sigma;
  Sigma = multiply_lower_tri_self_transpose(L);  // Sigma = L * L'
}

model {
  // prior on mu
  mu ~ normal(0, 10);

  // prior (general-k scale-right), computed via L
  target += scale_right_prior_log_chol(L, k);

  // likelihood
  y ~ multi_normal_cholesky(mu, L);
}
