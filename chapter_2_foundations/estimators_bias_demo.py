"""
Demonstrates the concepts of estimators, bias, and unbiased estimators,
focusing on the sample mean and sample variance.

Based on Chapter 2 concepts of 'The Hundred-Page Machine Learning Book'.

- Estimator: A rule/formula to estimate a population parameter from a sample.
- Population Parameter: True characteristic of the entire population (e.g., μ, σ²).
- Sample Statistic: Value calculated from a sample (e.g., x_bar, s²).
- Bias: Systematic difference between an estimator's expected value and the true parameter.
  - Bias(θ_hat) = E[θ_hat] - θ
- Unbiased Estimator: An estimator where E[θ_hat] = θ.

This script will:
1. Define a known population (e.g., normal distribution).
2. Repeatedly draw samples from this population.
3. For each sample, calculate:
    - Sample mean (x_bar)
    - Biased sample variance (denominator n)
    - Unbiased sample variance (denominator n-1, Bessel's correction)
4. Average these estimates over many samples.
5. Compare averages to true population parameters (μ_true, σ²_true).
6. Visualize distributions of estimates and highlight bias (or lack thereof).
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# --- Configuration ---
POPULATION_MEAN_TRUE = 50
POPULATION_STD_TRUE = 10
POPULATION_VAR_TRUE = POPULATION_STD_TRUE**2

N_SAMPLES = 1000  # Number of samples to draw
SAMPLE_SIZE = 10   # Size of each individual sample

print(f"--- Estimators and Bias Demonstration ---")
print(f"Population Parameters: μ_true = {POPULATION_MEAN_TRUE}, σ²_true = {POPULATION_VAR_TRUE:.2f}\n")

# Lists to store the estimates from each sample
sample_means = []
biased_sample_variances = []
unbiased_sample_variances = []

# --- Simulation: Drawing samples and calculating statistics ---
print(f"Simulating {N_SAMPLES} samples of size {SAMPLE_SIZE}...")
np.random.seed(42) # for reproducibility
for _ in range(N_SAMPLES):
    # Draw a sample from the defined normal population
    sample = np.random.normal(loc=POPULATION_MEAN_TRUE, scale=POPULATION_STD_TRUE, size=SAMPLE_SIZE)

    # 1. Calculate Sample Mean (x_bar)
    current_sample_mean = np.mean(sample)
    sample_means.append(current_sample_mean)

    # 2. Calculate Biased Sample Variance (denominator n)
    # sum((xi - x_bar)^2) / n
    biased_var = np.var(sample) # numpy.var by default uses n
    biased_sample_variances.append(biased_var)

    # 3. Calculate Unbiased Sample Variance (denominator n-1)
    # sum((xi - x_bar)^2) / (n-1)
    unbiased_var = np.var(sample, ddof=1) # ddof=1 for n-1 denominator
    unbiased_sample_variances.append(unbiased_var)

print("Simulation complete.\n")

# --- Analysis: Averaging the Estimates ---
avg_sample_mean = np.mean(sample_means)
avg_biased_sample_variance = np.mean(biased_sample_variances)
avg_unbiased_sample_variance = np.mean(unbiased_sample_variances)

print("--- Averaged Estimates from Simulations ---")
print(f"Average of Sample Means (E[x̄]): {avg_sample_mean:.4f} (True μ: {POPULATION_MEAN_TRUE})")
bias_mean = avg_sample_mean - POPULATION_MEAN_TRUE
print(f"  -> Bias for x̄: {bias_mean:.4f}")

print(f"Average of Biased Sample Variances (E[s_n²]): {avg_biased_sample_variance:.4f} (True σ²: {POPULATION_VAR_TRUE:.2f})")
bias_var_n = avg_biased_sample_variance - POPULATION_VAR_TRUE
print(f"  -> Bias for s_n² (denominator n): {bias_var_n:.4f}")

print(f"Average of Unbiased Sample Variances (E[s²]): {avg_unbiased_sample_variance:.4f} (True σ²: {POPULATION_VAR_TRUE:.2f})")
bias_var_n_minus_1 = avg_unbiased_sample_variance - POPULATION_VAR_TRUE
print(f"  -> Bias for s² (denominator n-1): {bias_var_n_minus_1:.4f}\n")

print("Note: Small non-zero biases for theoretically unbiased estimators are due to finite number of simulation samples.")
print("As N_SAMPLES -> ∞, these biases should approach 0.\n")


# --- Visualization ---
print("--- Visualizing Distributions of Estimates ---")

# 1. Distribution of Sample Means (x_bar)
plt.figure(figsize=(12, 5))
plt.hist(sample_means, bins=30, edgecolor='black', alpha=0.7, label=f'Distribution of {N_SAMPLES} Sample Means (x̄)')
plt.axvline(POPULATION_MEAN_TRUE, color='red', linestyle='dashed', linewidth=2, label=f'True Population Mean μ = {POPULATION_MEAN_TRUE}')
plt.axvline(avg_sample_mean, color='blue', linestyle='dashed', linewidth=2, label=f'Average of Sample Means E[x̄] ≈ {avg_sample_mean:.2f}')
plt.title(f'Distribution of Sample Means (Sample Size n={SAMPLE_SIZE})')
plt.xlabel('Sample Mean (x̄)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, linestyle='--')
plt.show()

# 2. Distribution of Biased Sample Variances (s_n^2)
plt.figure(figsize=(12, 5))
plt.hist(biased_sample_variances, bins=30, edgecolor='black', alpha=0.7, label=f'Distribution of {N_SAMPLES} Biased Variances (s_n²)')
plt.axvline(POPULATION_VAR_TRUE, color='red', linestyle='dashed', linewidth=2, label=f'True Population Variance σ² = {POPULATION_VAR_TRUE:.2f}')
plt.axvline(avg_biased_sample_variance, color='blue', linestyle='dashed', linewidth=2, label=f'Average of Biased Variances E[s_n²] ≈ {avg_biased_sample_variance:.2f}')
expected_biased_var = ((SAMPLE_SIZE - 1) / SAMPLE_SIZE) * POPULATION_VAR_TRUE
plt.axvline(expected_biased_var, color='green', linestyle='dotted', linewidth=2, label=f'Expected E[s_n²] = (n-1)/n * σ² ≈ {expected_biased_var:.2f}')
plt.title(f'Distribution of Biased Sample Variances (s_n², denominator n, Sample Size n={SAMPLE_SIZE})')
plt.xlabel('Biased Sample Variance (s_n²)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, linestyle='--')
plt.show()

# 3. Distribution of Unbiased Sample Variances (s^2)
plt.figure(figsize=(12, 5))
plt.hist(unbiased_sample_variances, bins=30, edgecolor='black', alpha=0.7, label=f'Distribution of {N_SAMPLES} Unbiased Variances (s²)')
plt.axvline(POPULATION_VAR_TRUE, color='red', linestyle='dashed', linewidth=2, label=f'True Population Variance σ² = {POPULATION_VAR_TRUE:.2f}')
plt.axvline(avg_unbiased_sample_variance, color='blue', linestyle='dashed', linewidth=2, label=f'Average of Unbiased Variances E[s²] ≈ {avg_unbiased_sample_variance:.2f}')
plt.title(f'Distribution of Unbiased Sample Variances (s², denominator n-1, Sample Size n={SAMPLE_SIZE})')
plt.xlabel('Unbiased Sample Variance (s²)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, linestyle='--')
plt.show()

print("""
Observations:
1. The distribution of sample means (x̄) is centered around the true population mean (μ_true).
   The average of sample means is very close to μ_true, demonstrating it's an UNBIASED estimator.

2. The distribution of biased sample variances (s_n², using n in denominator) is centered
   to the LEFT of the true population variance (σ²_true).
   The average of these s_n² values is consistently LESS than σ²_true, demonstrating NEGATIVE BIAS.
   The green dotted line shows the theoretical expected value E[s_n²] = ((n-1)/n) * σ², which matches the simulation.

3. The distribution of unbiased sample variances (s², using n-1 in denominator) is centered
   around the true population variance (σ²_true).
   The average of these s² values is very close to σ²_true, demonstrating it's an UNBIASED estimator (after Bessel's correction).
""") 