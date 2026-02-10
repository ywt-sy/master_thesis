functions {
  // ---------------------------------------------------------------
  // General-k scale-right prior:
  // log π(Σ) = - Σ_{i=1..k} (1/choose(k,i)) Σ_{I ⊂ {1..k}, |I| = i} log |Σ_I|
  // Enumerates all subsets using bitmasks.
  // ---------------------------------------------------------------
  real scale_right_prior_log(matrix Sigma, int k) {
    real log_val = 0;

    // total subsets = 2^k
    int max_mask = 1;
    for (i in 1:k)
      max_mask *= 2;  // ∴ max_mask = 2^k

    // subset sizes i = 1 .. k
    for (i in 1:k) {
      real sum_log = 0;
      int m_count = 0;

      // loop over all nonempty subsets via bitmask
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

        // skip subsets whose size ≠ i
        if (idx_len == i) {
          real log_det_sub =
            log_determinant(Sigma[idx[1:i], idx[1:i]]);

          // add -log(det)
          sum_log += -log_det_sub;
          m_count += 1;
        }
      }

      // add average log(1/|Σ_I|)
      log_val += sum_log / m_count;

    }

    return log_val;
  }
}

data {
  int<lower=0> n;               // number of observations
  int<lower=1> k;               // dimension
  array[n] vector[k] y;         // data points
}

parameters {
  vector[k] mu;                 // mean vector
  cov_matrix[k] Sigma;          // k×k SPD covariance matrix
}

model {
  // prior on mu
  mu ~ normal(0, 10);

  // your general-k prior
  target += scale_right_prior_log(Sigma, k);

  // likelihood
  y ~ multi_normal(mu, Sigma);
}
