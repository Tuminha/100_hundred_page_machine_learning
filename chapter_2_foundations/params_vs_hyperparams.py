"""
Illustrates the difference between Model Parameters and Hyperparameters
using a simple SVM classification example, based on Chapter 2 concepts of
'The Hundred-Page Machine Learning Book'.

- Hyperparameters: Settings chosen *before* training (e.g., C and kernel in SVM).
- Parameters: Values *learned* by the model *during* training (e.g., support vectors,
             coefficients defining the SVM decision boundary).

We visualize how changing the hyperparameter 'C' leads to different learned models
(different decision boundaries, representing different learned parameters), using an RBF kernel
to better highlight the effect.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# --- Generate Sample Data ---
# Simulate some 2D data (e.g., representing simplified BIC vs ISQ features)
# where classes are reasonably separable but might have some overlap.
X, y = make_blobs(n_samples=100, centers=2, n_features=2,
                  cluster_std=1.5, random_state=42)

# --- Add an Outlier --- 
# Create a point with low F1/F2 values but assign it to Class 1 (blue group)
# This simulates an unexpected outcome (e.g., failure despite good initial metrics? Or vice-versa)
# We need to choose coordinates that will fall within the Class 0 region.
# Based on random_state=42, Class 0 is lower-left. Let's place it there.
outlier_point = np.array([[-2, -3]]) # Low F1, Low F2
outlier_label = np.array([1])        # But assign to Class 1

print(f"\nOriginal data shape: X={X.shape}, y={y.shape}")
# Add the outlier to the dataset
X = np.concatenate((X, outlier_point), axis=0)
y = np.concatenate((y, outlier_label), axis=0)
print(f"Data shape after adding outlier: X={X.shape}, y={y.shape}\n")

# --- Define Hyperparameter Choices ---
# We will vary the SVM's regularization hyperparameter 'C' AND the kernel coefficient 'gamma'.

hyperparameter_C_low = 0.1
hyperparameter_gamma_low = 0.1 # Low gamma -> smoother boundary

hyperparameter_C_high = 100.0
hyperparameter_gamma_high = 10.0 # High gamma -> more complex, sensitive boundary

# --- Create and Train Models with Different Hyperparameters (on data *with* outlier) ---

# Model 1: Low C, Low Gamma (Hyperparameters)
svm_low_c_low_gamma = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf', C=hyperparameter_C_low, gamma=hyperparameter_gamma_low)
)
svm_low_c_low_gamma.fit(X, y)
# Learns parameters guided by low C & low gamma -> expects smooth boundary, tolerant of outlier

# Model 2: High C, High Gamma (Hyperparameters)
svm_high_c_high_gamma = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf', C=hyperparameter_C_high, gamma=hyperparameter_gamma_high)
)
svm_high_c_high_gamma.fit(X, y)
# Learns parameters guided by high C & high gamma -> expects complex boundary, sensitive to outlier

print(f"Hyperparameters set before training:")
print(f" Model 1: C = {hyperparameter_C_low}, gamma = {hyperparameter_gamma_low}")
print(f" Model 2: C = {hyperparameter_C_high}, gamma = {hyperparameter_gamma_high}")
# Parameters for RBF SVM are more complex (support vectors, dual coefficients, intercept)
# Accessing them directly is less illustrative than seeing the boundary difference.
print("\nModel Parameters (e.g., support vectors, dual coefs, intercept) are LEARNED during .fit()")
print("--> Visualizing the decision boundary shows the effect of different Hyperparameters on learned Parameters.\n")

# --- Visualization --- 
# Show how the different hyperparameters lead to different learned models

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Hyperparameters (C, gamma) vs Learned Parameters (Decision Boundary)', fontsize=16)

# Helper function to plot decision boundary
def plot_decision_boundary(model, X, y, ax, title):
    scaler = model.named_steps.get('standardscaler')
    svm_clf = model.named_steps.get('svc')

    # Scale data for plotting consistency with decision boundary
    X_scaled = scaler.transform(X)

    # Create a mesh grid
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Predict on the mesh grid
    Z = svm_clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot contour and data points
    ax.contourf(xx, yy, Z, cmap='RdBu', alpha=0.3)
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='RdBu', edgecolors='k')

    # Add text labels for class regions
    # Find approximate centers of the colored regions based on meshgrid extent
    # Adjust positions as needed based on visual output
    text_x_pos_class0 = xx.min() + (xx.max() - xx.min()) * 0.1 # Position for Class 0 label (Red region)
    text_y_pos_class0 = yy.min() + (yy.max() - yy.min()) * 0.1
    text_x_pos_class1 = xx.min() + (xx.max() - xx.min()) * 0.9 # Position for Class 1 label (Blue region)
    text_y_pos_class1 = yy.min() + (yy.max() - yy.min()) * 0.9

    # Check the predicted class at these points to ensure label placement
    # (Simple check: assume corners/extremes are representative for blob data)
    # A more robust method might query Z at specific points
    ax.text(text_x_pos_class0, text_y_pos_class0, 'Class 0 Region',
            fontsize=10, ha='left', va='bottom', color='darkred',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    ax.text(text_x_pos_class1, text_y_pos_class1, 'Class 1 Region',
            fontsize=10, ha='right', va='top', color='darkblue',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    # Plot support vectors if available (optional, for illustration)
    # ax.scatter(svm_clf.support_vectors_[:, 0], svm_clf.support_vectors_[:, 1],
    #            s=100, facecolors='none', edgecolors='k', label='Support Vectors')

    ax.set_title(title)
    ax.set_xlabel("Feature 1 (e.g., Scaled BIC)")
    ax.set_ylabel("Feature 2 (e.g., Scaled ISQ)")
    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, labelleft=False)

# Plot for Low C, Low Gamma
plot_decision_boundary(svm_low_c_low_gamma, X, y, ax[0],
                       f'HYPERPARAMETERS: C={hyperparameter_C_low}, gamma={hyperparameter_gamma_low}\n(Learned PARAMETERS define this boundary)')

# Plot for High C, High Gamma
plot_decision_boundary(svm_high_c_high_gamma, X, y, ax[1],
                       f'HYPERPARAMETERS: C={hyperparameter_C_high}, gamma={hyperparameter_gamma_high}\n(Learned PARAMETERS define this boundary)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
print("Displaying plot showing effect of Hyperparameters C and gamma on learned model boundary...")
plt.show()

print("\nSummary:")
print("- We SET Hyperparameters (like C and gamma) before training.")
print("- The model LEARNS Parameters (defining the boundary) during training.")
print("- Different Hyperparameters lead to different learned Parameters.") 