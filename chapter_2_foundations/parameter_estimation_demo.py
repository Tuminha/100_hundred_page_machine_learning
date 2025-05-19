"""
Demonstrates Parameter Estimation concepts with visualizations and dental examples,
based on Chapter 2 concepts of 'The Hundred-Page Machine Learning Book'.

Covers:
- Maximum Likelihood Estimation (MLE)
- Method of Moments (MoM)
- Bayesian Estimation
- Properties of estimators (unbiased, efficient, consistent)
- Dental application: Implant success rates
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize

print("\n--- Parameter Estimation: Mathematical Foundations ---")
print("Parameter estimation is the process of finding the best values for model parameters")
print("that best describe our observed data. We'll explore different methods and their properties.")

# --- 1. Normal Distribution Parameter Estimation ---
print("\n--- 1. Normal Distribution Parameter Estimation ---")
print("Example: Estimating mean (μ) and variance (σ²) of a normal distribution")

# Generate sample data from a normal distribution
np.random.seed(42)  # for reproducibility
true_mu = 170  # true mean (e.g., implant length in mm)
true_sigma = 10  # true standard deviation
n_samples = 100
samples = np.random.normal(true_mu, true_sigma, n_samples)

# Method of Moments (MoM) estimation
mom_mu = np.mean(samples)
mom_sigma2 = np.var(samples, ddof=0)  # population variance

# Maximum Likelihood Estimation (MLE)
mle_mu = np.mean(samples)  # same as MoM for normal distribution
mle_sigma2 = np.var(samples, ddof=0)  # same as MoM for normal distribution

print(f"\nTrue parameters: μ = {true_mu}, σ² = {true_sigma**2}")
print(f"MoM estimates: μ̂ = {mom_mu:.2f}, σ̂² = {mom_sigma2:.2f}")
print(f"MLE estimates: μ̂ = {mle_mu:.2f}, σ̂² = {mle_sigma2:.2f}")

# Visualize the estimation
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Histogram with estimated and true distributions
x = np.linspace(true_mu - 4*true_sigma, true_mu + 4*true_sigma, 100)
ax1.hist(samples, bins=20, density=True, alpha=0.6, label='Sample Data')
ax1.plot(x, stats.norm.pdf(x, true_mu, true_sigma), 'r-', 
         label=f'True Distribution\nμ={true_mu}, σ={true_sigma}')
ax1.plot(x, stats.norm.pdf(x, mle_mu, np.sqrt(mle_sigma2)), 'g--',
         label=f'Estimated Distribution\nμ̂={mle_mu:.1f}, σ̂={np.sqrt(mle_sigma2):.1f}')
ax1.set_title('Normal Distribution: True vs Estimated Parameters')
ax1.set_xlabel('Value')
ax1.set_ylabel('Density')
ax1.legend()

# Plot 2: Convergence of estimates with sample size
sample_sizes = np.arange(10, n_samples + 1, 10)
mu_estimates = [np.mean(samples[:n]) for n in sample_sizes]
sigma2_estimates = [np.var(samples[:n], ddof=0) for n in sample_sizes]

ax2.plot(sample_sizes, mu_estimates, 'b-', label='μ̂ estimates')
ax2.plot(sample_sizes, sigma2_estimates, 'g-', label='σ̂² estimates')
ax2.axhline(y=true_mu, color='b', linestyle='--', alpha=0.5, label='True μ')
ax2.axhline(y=true_sigma**2, color='g', linestyle='--', alpha=0.5, label='True σ²')
ax2.set_title('Convergence of Parameter Estimates')
ax2.set_xlabel('Sample Size')
ax2.set_ylabel('Parameter Value')
ax2.legend()

plt.tight_layout()
plt.show()

# --- 2. Binomial Distribution: Dental Implant Success Rates ---
print("\n--- 2. Dental Application: Implant Success Rate Estimation ---")
print("Example: Estimating success rate of a new implant technique")

# True success rate (unknown in real life)
true_p = 0.85
n_trials = 50
successes = np.random.binomial(n_trials, true_p)

print(f"\nDental Implant Study:")
print(f"Number of trials: {n_trials}")
print(f"Number of successes: {successes}")

# MLE estimation
mle_p = successes / n_trials

# Bayesian estimation with different priors
# Prior 1: Uniform (no prior knowledge)
alpha1, beta1 = 1, 1
# Prior 2: Optimistic (based on similar techniques)
alpha2, beta2 = 8, 2  # prior mean = 0.8
# Prior 3: Conservative (based on traditional methods)
alpha3, beta3 = 4, 6  # prior mean = 0.4

# Posterior parameters
post_alpha1 = alpha1 + successes
post_beta1 = beta1 + (n_trials - successes)
post_alpha2 = alpha2 + successes
post_beta2 = beta2 + (n_trials - successes)
post_alpha3 = alpha3 + successes
post_beta3 = beta3 + (n_trials - successes)

# Calculate posterior means
post_mean1 = post_alpha1 / (post_alpha1 + post_beta1)
post_mean2 = post_alpha2 / (post_alpha2 + post_beta2)
post_mean3 = post_alpha3 / (post_alpha3 + post_beta3)

print(f"\nParameter Estimates:")
print(f"MLE: p̂ = {mle_p:.3f}")
print(f"Bayesian (Uniform Prior): p̂ = {post_mean1:.3f}")
print(f"Bayesian (Optimistic Prior): p̂ = {post_mean2:.3f}")
print(f"Bayesian (Conservative Prior): p̂ = {post_mean3:.3f}")

# Visualize the estimation
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Prior and posterior distributions
x = np.linspace(0, 1, 100)
ax1.plot(x, stats.beta.pdf(x, alpha1, beta1), 'b--', label='Uniform Prior')
ax1.plot(x, stats.beta.pdf(x, alpha2, beta2), 'g--', label='Optimistic Prior')
ax1.plot(x, stats.beta.pdf(x, alpha3, beta3), 'r--', label='Conservative Prior')
ax1.plot(x, stats.beta.pdf(x, post_alpha1, post_beta1), 'b-', label='Posterior (Uniform)')
ax1.plot(x, stats.beta.pdf(x, post_alpha2, post_beta2), 'g-', label='Posterior (Optimistic)')
ax1.plot(x, stats.beta.pdf(x, post_alpha3, post_beta3), 'r-', label='Posterior (Conservative)')
ax1.axvline(x=true_p, color='k', linestyle='--', label='True Success Rate')
ax1.axvline(x=mle_p, color='k', linestyle=':', label='MLE Estimate')
ax1.set_title('Prior and Posterior Distributions')
ax1.set_xlabel('Success Rate (p)')
ax1.set_ylabel('Density')
ax1.legend()

# Plot 2: Confidence intervals for different sample sizes
sample_sizes = np.arange(10, n_trials + 1, 5)
mle_estimates = [np.random.binomial(n, true_p) / n for n in sample_sizes]
z = 1.96  # 95% confidence interval

# Calculate confidence intervals
ci_lower = [p - z * np.sqrt(p * (1-p) / n) for p, n in zip(mle_estimates, sample_sizes)]
ci_upper = [p + z * np.sqrt(p * (1-p) / n) for p, n in zip(mle_estimates, sample_sizes)]

ax2.plot(sample_sizes, mle_estimates, 'b-', label='MLE estimates')
ax2.fill_between(sample_sizes, ci_lower, ci_upper, alpha=0.2, label='95% CI')
ax2.axhline(y=true_p, color='r', linestyle='--', label='True Success Rate')
ax2.set_title('Convergence of Success Rate Estimates')
ax2.set_xlabel('Sample Size')
ax2.set_ylabel('Estimated Success Rate')
ax2.legend()

plt.tight_layout()
plt.show()

# --- 3. Properties of Estimators ---
print("\n--- 3. Properties of Good Estimators ---")
print("Demonstrating unbiasedness, efficiency, and consistency")

# Generate multiple samples to demonstrate properties
n_experiments = 1000
sample_sizes = [10, 50, 100, 500]
estimates = {n: [] for n in sample_sizes}

for n in sample_sizes:
    for _ in range(n_experiments):
        sample = np.random.normal(true_mu, true_sigma, n)
        estimates[n].append(np.mean(sample))

# Visualize estimator properties
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Distribution of estimates for different sample sizes
for n in sample_sizes:
    ax1.hist(estimates[n], bins=30, alpha=0.5, label=f'n={n}')

ax1.axvline(x=true_mu, color='r', linestyle='--', label='True μ')
ax1.set_title('Distribution of Mean Estimates')
ax1.set_xlabel('Estimated Mean')
ax1.set_ylabel('Frequency')
ax1.legend()

# Plot 2: Variance of estimates vs sample size
variances = [np.var(estimates[n]) for n in sample_sizes]
ax2.plot(sample_sizes, variances, 'bo-')
ax2.set_title('Variance of Estimates vs Sample Size')
ax2.set_xlabel('Sample Size')
ax2.set_ylabel('Variance of Estimates')
ax2.set_xscale('log')
ax2.set_yscale('log')

plt.tight_layout()
plt.show()

print("\nKey Insights:")
print("1. MLE and MoM give identical results for normal distribution parameters")
print("2. Bayesian estimates incorporate prior knowledge, useful for small samples")
print("3. As sample size increases, estimates converge to true values (consistency)")
print("4. Variance of estimates decreases with sample size (efficiency)")
print("5. Different priors lead to different posterior estimates in Bayesian approach")
print("\nDental Application:")
print("For implant success rates, Bayesian methods are particularly valuable when:")
print("- Sample sizes are small")
print("- We have strong prior knowledge from similar techniques")
print("- We need to incorporate expert opinion into our estimates") 