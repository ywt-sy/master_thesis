data {
  int<lower=0> n;               // number of observations
  int<lower=1> k;               // dimension
  array[n] vector[k] y;         // y[n] is k-dimensional
}

parameters {
  cholesky_factor_cov[k] L;     // Cholesky factor of Sigma (Sigma = L * L')
}

transformed parameters {
  // If you want the full covariance matrix explicitly:
  cov_matrix[k] Sigma;
  Sigma = multiply_lower_tri_self_transpose(L);  // Sigma = L * L'
}

model {

  // ---- prior on L via its diagonal s_i = L[i,i] ----
  {
    vector[k] s = diagonal(L);  // s[i] = L_ii, all positive

    for (i in 1:k) {
      // log prior: - (k + 1 - i) * log(s_i)
      target += - (k + 1 - i) * log(s[i]);
    }
  }

  // likelihood using Cholesky parameterization with mean fixed at 0
  y ~ multi_normal_cholesky(rep_vector(0, k), L);
}
