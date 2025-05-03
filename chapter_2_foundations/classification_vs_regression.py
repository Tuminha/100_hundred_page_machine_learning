"""
Illustrates the difference between Classification and Regression tasks in
Supervised Machine Learning, based on Chapter 2 concepts of
'The Hundred-Page Machine Learning Book'.

- Classification: Predicts discrete categories/labels.
- Regression: Predicts continuous numerical values.

Includes visualizations for both types of tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# --- Classification Example --- 
print("--- Classification Task ---")
print("Goal: Predict a discrete category (e.g., Implant Success/Failure)")

# Generate 2D data suitable for classification
# Imagine Feature 1 = Scaled BIC, Feature 2 = Scaled ISQ
X_clf, y_clf = make_blobs(n_samples=150, centers=2, n_features=2,
                          cluster_std=1.8, random_state=30) # Use a different state for variety

# Create and train a classification model (Logistic Regression)
# Using a pipeline for potential scaling (good practice, though not strictly needed for LogReg)
clf_model = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)
clf_model.fit(X_clf, y_clf)

print("Trained a Logistic Regression classifier.")

# --- Visualization for Classification ---

fig_clf, ax_clf = plt.subplots(figsize=(8, 6))

# Plot decision boundary
scaler = clf_model.named_steps['standardscaler']
log_reg = clf_model.named_steps['logisticregression']
X_clf_scaled = scaler.transform(X_clf)

x_min, x_max = X_clf_scaled[:, 0].min() - 1, X_clf_scaled[:, 0].max() + 1
y_min, y_max = X_clf_scaled[:, 1].min() - 1, X_clf_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax_clf.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.4)
scatter = ax_clf.scatter(X_clf_scaled[:, 0], X_clf_scaled[:, 1], c=y_clf, cmap='coolwarm', edgecolors='k')

# Add legend/labels for clarity
handles, _ = scatter.legend_elements()
ax_clf.legend(handles, ['Class 0 (e.g., Failure)', 'Class 1 (e.g., Success)'], title="Actual Classes")
ax_clf.set_xlabel("Feature 1 (e.g., Scaled BIC)")
ax_clf.set_ylabel("Feature 2 (e.g., Scaled ISQ)")
ax_clf.set_title("Classification Example: Predicting Categories")
ax_clf.text(xx.min() + 0.5, yy.min() + 0.5, 'Predicted Class 0 Region', color='#6788FF', ha='left', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
ax_clf.text(xx.max() - 0.5, yy.max() - 0.5, 'Predicted Class 1 Region', color='#FF8888', ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

ax_clf.grid(True, linestyle='--', alpha=0.6)
print("Displaying Classification plot...")
plt.show()


# --- Regression Example --- 
print("\n--- Regression Task ---")
print("Goal: Predict a continuous numerical value (e.g., Final ISQ)")

# Generate 1D data suitable for regression
# Imagine X = Initial ISQ, y = Final ISQ (with some linear relationship + noise)
X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)

# Ensure y_reg is 1D for plotting
y_reg = y_reg.ravel()

# Create and train a regression model (Linear Regression)
reg_model = LinearRegression()
reg_model.fit(X_reg, y_reg)

print("Trained a Linear Regression model.")

# Make predictions on the training data range to plot the line
x_plot = np.linspace(X_reg.min(), X_reg.max(), 100).reshape(-1, 1)
y_pred_plot = reg_model.predict(x_plot)

# --- Visualization for Regression ---

fig_reg, ax_reg = plt.subplots(figsize=(8, 6))

# Plot original data points
ax_reg.scatter(X_reg, y_reg, edgecolors='k', label='Actual Data Points')

# Plot the learned regression line
ax_reg.plot(x_plot, y_pred_plot, color='red', linewidth=2, label='Learned Regression Line')

ax_reg.set_xlabel("Feature (e.g., Initial ISQ)")
ax_reg.set_ylabel("Target Value (e.g., Final ISQ)")
ax_reg.set_title("Regression Example: Predicting a Continuous Value")
ax_reg.legend()
ax_reg.grid(True, linestyle='--', alpha=0.6)
print("Displaying Regression plot...")
plt.show() 