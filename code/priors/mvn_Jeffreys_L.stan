data {
  int<lower=0> n;               // number of observations
  int<lower=1> k;               // dimension
  array[n] vector[k] y;         // y[n] is k-dimensional
}

parameters {
  vector[k] mu;                 // mean vector
  cholesky_factor_cov[k] L;     // Cholesky factor of Sigma (Sigma = L * L')
}

transformed parameters {
  // If you want the full covariance matrix explicitly:
  cov_matrix[k] Sigma;
  Sigma = multiply_lower_tri_self_transpose(L);  // Sigma = L * L'
}

model {
  // prior on mu
  mu ~ normal(0, 10);

  // ---- prior on L via its diagonal s_i = L[i,i] ----
  {
    vector[k] s = diagonal(L);  // s[i] = L_ii, all positive

    for (i in 1:k) {
      // log prior: - i * log(s_i)
      target += - i * log(s[i]);
    }
  }

  // likelihood using Cholesky parameterization
  y ~ multi_normal_cholesky(mu, L);
}
