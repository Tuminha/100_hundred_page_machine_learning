"""
Demonstrates the use of Sigma (Σ) notation, particularly for calculating
the Sum of Squared Errors (SSE) in a simple regression context, based on
Chapter 2 concepts of 'The Hundred-Page Machine Learning Book'.

SSE = Σ_{i=1}^{N} (y_i - ŷ_i)²
Where:
- Σ represents summation.
- i is the index from 1 to N (number of data points).
- y_i is the actual target value for the i-th data point.
- ŷ_i (y-hat) is the predicted target value for the i-th data point.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

print("--- Sigma Notation (Σ) Introduction ---")
# Concept: A compact way to represent the sum of a sequence of numbers.
# Σ_{i=m}^{n} expression(i)
# Sum the results of 'expression(i)' as the index 'i' goes from 'm' to 'n'.

# General Example: Summing elements of a vector
# Σ_{i=1}^{5} v_i where v = [10, 20, 30, 40, 50]
v = np.array([10, 20, 30, 40, 50])
sum_v = np.sum(v) # NumPy's sum directly applies Σ notation concept
print(f"General Example: Vector v = {v}")
print(f"Sum using np.sum (Σ v_i): {sum_v}") # Output: 150
print("-"*35)

print("\n--- Sigma Notation in ML: Sum of Squared Errors (SSE) ---")
# Context: Evaluating how well a regression model fits the data.

# 1. Generate Sample Data (e.g., Initial ISQ vs Final ISQ)
np.random.seed(0) # for reproducibility
initial_isq = np.random.rand(30, 1) * 30 + 50 # Simulate Initial ISQ values (50-80 range)
# Simulate Final ISQ with a linear trend + noise
# final_isq = true_intercept + true_slope * initial_isq + noise
final_isq = 5 + 0.8 * initial_isq + np.random.randn(30, 1) * 8
final_isq = final_isq.ravel() # Make y 1D
initial_isq_flat = initial_isq.ravel() # Keep a flat version for plotting info

print(f"Generated {len(final_isq)} data points (e.g., Initial vs Final ISQ).")

# 2. Fit a Simple Linear Regression Model
model = LinearRegression()
model.fit(initial_isq, final_isq)
print("Trained Linear Regression model.")

# 3. Get Predictions (ŷ_i)
predictions = model.predict(initial_isq)

# 4. Calculate Errors (Residuals): (y_i - ŷ_i)
errors = final_isq - predictions

# 5. Calculate Squared Errors: (y_i - ŷ_i)²
squared_errors = errors ** 2

# 6. Calculate Sum of Squared Errors (SSE) using Sigma notation (via np.sum)
# SSE = Σ_{i=1}^{N} squared_errors_i
sse = np.sum(squared_errors)

print(f"Actual Final ISQ (y_i) sample: {final_isq[:5]:.2f}...")
print(f"Predicted Final ISQ (ŷ_i) sample: {predictions[:5]:.2f}...")
print(f"Errors (y_i - ŷ_i) sample: {errors[:5]:.2f}...")
print(f"Squared Errors sample: {squared_errors[:5]:.2f}...")
print(f"\nSum of Squared Errors (SSE = Σ (y_i - ŷ_i)²): {sse:.2f}")

# 7. Visualize the Regression and Errors
fig, ax = plt.subplots(figsize=(10, 6))

# Plot actual data points
ax.scatter(initial_isq, final_isq, edgecolors='k', label='Actual Data (y_i)')

# Plot regression line (predictions)
ax.plot(initial_isq, predictions, color='red', linewidth=2, label=f'Regression Line (ŷ_i)')

# Plot error lines (residuals)
for i in range(len(final_isq)):
    ax.plot([initial_isq[i], initial_isq[i]], [final_isq[i], predictions[i]], 'g--', alpha=0.6) # Vertical lines for errors
# Add a label for the first error line only to avoid clutter
ax.plot([initial_isq[0], initial_isq[0]], [final_isq[0], predictions[0]], 'g--', alpha=0.6, label='Errors (y_i - ŷ_i)')

ax.set_xlabel("Initial ISQ (Feature)")
ax.set_ylabel("Final ISQ (Target)")
ax.set_title("Sigma Notation Example: Sum of Squared Errors (SSE) in Regression")

# Display SSE value on plot
sse_text = f"SSE = Σ(error)² = {sse:.2f}"
ax.text(0.05, 0.95, sse_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)
print("Displaying Regression plot with errors and SSE...")
plt.show() 