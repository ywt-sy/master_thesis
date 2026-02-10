from cmdstanpy import CmdStanModel

model = CmdStanModel(stan_file="code/test_combinations.stan")

# 1サンプルだけで十分（comb は transformed data で既に全部出る）
fit = model.sample(iter_warmup=1, iter_sampling=1, chains=1,show_console=True)

# 出力はターミナル (stdout) に
