import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    n = 104
    k_list = [5,10,15]
    method_list = ["jst","rst"]
    for method in method_list:
        for k in k_list:
            # Load the .npz file
            data = np.load(f'{n}_{k}_{method}_sample.npz')  # <- Replace with your actual filename

            # Assume keys are 'mu_samples' and 'Sigma_samples'
            # mu_samples shape: (num_samples, k)
            # Sigma_samples shape: (num_samples, k, k)
            mu_samples = data['mu']
            Sigma_samples = data['Sigma']

            num_samples, k = mu_samples.shape

            # === Plot trace for each element of mu ===
            fig_mu, axes_mu = plt.subplots(k, 1, figsize=(10, 2*k), sharex=True)
            fig_mu.suptitle('Trace Plots for Each Element of μ', fontsize=14)

            for i in range(k):
                axes_mu[i].plot(mu_samples[:, i])
                axes_mu[i].set_ylabel(f'μ[{i+1}]')

            axes_mu[-1].set_xlabel('Iteration')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(f'trance_plot/{n}_{k}_trace_mu_{method}.png')
            plt.close(fig_mu)

            # === Plot trace for diagonal of Sigma (variances) ===
            fig_var, axes_var = plt.subplots(k, 1, figsize=(10, 2*k), sharex=True)
            fig_var.suptitle('Trace Plots for Diagonal Elements of Σ (Variances)', fontsize=14)

            for i in range(k):
                axes_var[i].plot(Sigma_samples[:, i, i])
                axes_var[i].set_ylabel(f'Σ[{i+1},{i+1}]')

            axes_var[-1].set_xlabel('Iteration')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(f'trance_plot/{n}_{k}_trace_sigma_diag_{method}.png')
            plt.close(fig_var)

            # === Optional: Off-diagonal Sigma ===
            # Uncomment below to include off-diagonal trace plots
            # fig_off, axes_off = plt.subplots(k*(k-1)//2, 1, figsize=(10, 2*k), sharex=True)
            # fig_off.suptitle('Trace Plots for Off-diagonal Elements of Σ', fontsize=14)
            # idx = 0
            # for i in range(k):
            #     for j in range(i+1, k):
            #         axes_off[idx].plot(Sigma_samples[:, i, j])
            #         axes_off[idx].set_ylabel(f'Σ[{i+1},{j+1}]')
            #         idx += 1
            # axes_off[-1].set_xlabel('Iteration')
            # plt.tight_layout(rect=[0, 0, 1, 0.96])
            # plt.savefig('trace_sigma_offdiag.png')
            # plt.close(fig_off)
