functions {
  void test_combinations(int k) {

    // max_mask = 2^k
    int max_mask = 1;
    for (i in 1:k)
      max_mask *= 2;

    // subset sizes i = 1..k
    for (i in 1:k) {
      print("=== subsets of size ", i, " ===");

      // loop through all bitmask subsets
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
          tmp = tmp / 2;   // int / int → 切り捨てでOK
        }

        // only subsets of correct size
        if (idx_len == i) {
          print("subset: ", idx[1:idx_len]);
        }
      }
    }
  }
}

transformed data {
  int k = 3;              // ★ ここを変えれば次元変えられる
  test_combinations(k);   // 実行時に一回だけ呼ばれる
}

parameters {
  real dummy;             // ダミー parameter（HMC 用の飾り）
}

model {
  dummy ~ normal(0, 1);   // 適当な尤度（何でもいい）
}
