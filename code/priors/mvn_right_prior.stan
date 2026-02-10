data {
  int<lower=0> n;               // number of observations
  int<lower=1> k;               // dimension
  array[n] vector[k] y;         // y[n] is k-dimensional
}
parameters {
  vector[k] mu;                 // mean vector
  cov_matrix[k] Sigma;          // k x k covariance matrix
}
model {
  // Prior on mu:
  mu ~ normal(0, 10);

  // Right-invariant prior on Sigma:
  // pi_R(Sigma) ∝ 1 / prod_{l=1}^k |Sigma_l|
  for (l in 1:k) {
    target += -log_determinant(Sigma[1:l, 1:l]);
  }

  // Likelihood:
  y ~ multi_normal(mu, Sigma);
}