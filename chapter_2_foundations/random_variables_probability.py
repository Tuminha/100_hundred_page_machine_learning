"""
Demonstrates concepts related to Random Variables and Probability Distributions
from Chapter 2 of 'The Hundred-Page Machine Learning Book'.

Includes examples for:
- Discrete Random Variables (using Bernoulli/Binomial)
  - Probability Mass Function (PMF)
  - Expected Value, Variance, Standard Deviation
  - Visualization of PMF
- Continuous Random Variables (using Normal Distribution)
  - Probability Density Function (PDF)
  - Expected Value (Mean), Variance, Standard Deviation
  - Calculating probability within a range (Area under PDF)
  - Visualization of PDF and probability area
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats # Use scipy.stats for distributions

print("--- Discrete Random Variables ---")
# Concept: Variable takes countable values (e.g., 0, 1, 2, ...)

# --- General Example: A biased coin flip --- 
# Random Variable X: 0 if Tails, 1 if Heads
# Assume P(Heads) = p = 0.6
p_heads = 0.6
# This is a Bernoulli distribution
rv_coin = stats.bernoulli(p_heads)

# Possible outcomes (k)
k_values_coin = [0, 1]
# Probability Mass Function (PMF): P(X=k)
pmf_coin = rv_coin.pmf(k_values_coin)

print(f"General Example: Biased Coin Flip (P(Heads)={p_heads})")
print(f" Possible Outcomes (k): {k_values_coin}")
print(f" PMF P(X=k): {pmf_coin}") # Output: [0.4 0.6]

# Expected Value (Mean): E[X] = Σ k * P(X=k)
mean_coin = rv_coin.mean() # E[X] = p for Bernoulli
print(f" Expected Value (Mean): {mean_coin:.2f}") # Output: 0.60

# Variance: Var(X) = E[(X - E[X])^2]
variance_coin = rv_coin.var() # Var(X) = p*(1-p) for Bernoulli
print(f" Variance: {variance_coin:.2f}") # Output: 0.24

# Standard Deviation: SD(X) = sqrt(Var(X))
std_dev_coin = rv_coin.std()
print(f" Standard Deviation: {std_dev_coin:.2f}") # Output: 0.49

# Visualize PMF
fig_pmf, ax_pmf = plt.subplots(figsize=(6, 4))
ax_pmf.bar(k_values_coin, pmf_coin, tick_label=['Tails (0)', 'Heads (1)'], width=0.5, alpha=0.7)
ax_pmf.set_ylabel("Probability Mass P(X=k)")
ax_pmf.set_title("PMF of Biased Coin Flip")
ax_pmf.set_ylim(0, 1)
ax_pmf.grid(axis='y', linestyle='--', alpha=0.6)
print("Displaying plot for PMF (Coin Flip)...")
plt.show()


# --- Implant Dentistry Example: Implant Success --- 
# Random Variable Z: 0 if Failure, 1 if Success
# Assume P(Success) = p_success = 0.90
p_success = 0.90
rv_implant = stats.bernoulli(p_success)

k_values_implant = [0, 1]
pmf_implant = rv_implant.pmf(k_values_implant)

print(f"\nDental Example: Implant Success (P(Success)={p_success})")
print(f" Possible Outcomes (k): {k_values_implant}")
print(f" PMF P(Z=k): {pmf_implant}") # Output: [0.1 0.9]

mean_implant = rv_implant.mean()
variance_implant = rv_implant.var()
std_dev_implant = rv_implant.std()

print(f" Expected Value (Mean): {mean_implant:.2f}") # Output: 0.90
print(f" Variance: {variance_implant:.2f}") # Output: 0.09
print(f" Standard Deviation: {std_dev_implant:.2f}") # Output: 0.30
# Interpretation: The average outcome is 0.9 (reflecting high success rate).
# The standard deviation of 0.3 indicates some variability (existence of failures).

# (Visualization would be similar to the coin flip, just different heights)


print("\n--- Continuous Random Variables ---")
# Concept: Variable takes any value within a range (e.g., height, ISQ)

# --- Implant Dentistry Example: ISQ Value --- 
# Assume ISQ values (Y) follow a Normal (Gaussian) distribution
# Parameters for Normal: loc = mean (μ), scale = standard deviation (σ)
mean_isq = 70  # Assume average ISQ μ = 70
std_dev_isq = 5   # Assume standard deviation σ = 5

# Create the normal distribution object
rv_isq = stats.norm(loc=mean_isq, scale=std_dev_isq)

print(f"\nDental Example: ISQ Value (Normal Distribution, μ={mean_isq}, σ={std_dev_isq})")

# Expected Value (Mean)
# For Normal distribution, the 'loc' parameter is the mean
print(f" Expected Value (Mean): {rv_isq.mean():.2f}") # Output: 70.00

# Variance
# For Normal distribution, variance is scale^2 (σ^2)
print(f" Variance (σ^2): {rv_isq.var():.2f}") # Output: 25.00

# Standard Deviation
# For Normal distribution, the 'scale' parameter is the std dev
print(f" Standard Deviation (σ): {rv_isq.std():.2f}") # Output: 5.00

# Probability Density Function (PDF): f(y)
# Gives relative likelihood. PDF value itself is NOT a probability.
# Example: PDF at ISQ=70 (the mean) vs ISQ=75 (1 std dev away)
isq_point_1 = 70
isq_point_2 = 75
pdf_val_1 = rv_isq.pdf(isq_point_1)
pdf_val_2 = rv_isq.pdf(isq_point_2)
print(f" PDF value at ISQ={isq_point_1}: {pdf_val_1:.4f}")
print(f" PDF value at ISQ={isq_point_2}: {pdf_val_2:.4f} (Lower means less likely)")

# Probability within a range: P(a <= Y <= b) = Area under PDF
# Example: Probability ISQ is between 65 (μ-σ) and 75 (μ+σ)
lower_bound = 65
upper_bound = 75
# Use Cumulative Distribution Function (CDF): P(Y <= y)
# P(a <= Y <= b) = CDF(b) - CDF(a)
prob_in_range = rv_isq.cdf(upper_bound) - rv_isq.cdf(lower_bound)
print(f" Probability ISQ is between {lower_bound} and {upper_bound}: {prob_in_range:.4f}") # Should be ~0.68 for +/- 1 sigma

# Visualize PDF and probability area
fig_pdf, ax_pdf = plt.subplots(figsize=(8, 5))

# Range for plotting x-axis (e.g., mean +/- 4 std devs)
x_min = mean_isq - 4 * std_dev_isq
x_max = mean_isq + 4 * std_dev_isq
x_values = np.linspace(x_min, x_max, 200)

# PDF curve
pdf_values = rv_isq.pdf(x_values)
ax_pdf.plot(x_values, pdf_values, 'b-', lw=2, label='Normal PDF')

# Shade area between lower_bound and upper_bound
x_fill = np.linspace(lower_bound, upper_bound, 100)
y_fill = rv_isq.pdf(x_fill)
ax_pdf.fill_between(x_fill, y_fill, color='lightblue', alpha=0.6, label=f'Area P({lower_bound}≤ISQ≤{upper_bound})')

ax_pdf.set_xlabel("ISQ Value")
ax_pdf.set_ylabel("Probability Density f(y)")
ax_pdf.set_title(f"PDF of ISQ (Normal Dist. μ={mean_isq}, σ={std_dev_isq})")
ax_pdf.legend()
ax_pdf.grid(True, linestyle='--', alpha=0.6)
print("Displaying plot for PDF (ISQ) and probability range...")
plt.show() 