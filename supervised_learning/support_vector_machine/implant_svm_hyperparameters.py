"""
Visualizes the effect of hyperparameters C and Gamma on the decision boundary
of an RBF Kernel SVM using simulated implant data (BIC vs ISQ).

This script is intended for conceptual understanding:
- Uses training data for visualization to show fitting behaviour.
- Demonstrates boundary changes with Low/High C and Low/High Gamma.

Assumes data (X_train, y_train) might be generated elsewhere or uses placeholder data.
(Currently uses data generation within for self-contained demo)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs # For data generation if run standalone
from sklearn.model_selection import train_test_split # For data generation if run standalone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# Removed metrics imports as they are not used here directly

# --- Understanding C and Gamma with RBF Kernel (ELI12 Analogy) ---
#
# Imagine building a fence between cat (Class 1) and dog (Class -1) areas in a yard,
# based on where you've seen some cats and dogs (your training data).
# The RBF kernel lets you build a curvy fence.
#
# Gamma (γ) - Reach of Influence:
#   - LOW Gamma: Like big spotlights around key posts (support vectors). Influence
#     spreads far. Fence becomes very SMOOTH, ignoring small details.
#     Good for: General trends. Risk: Too simple (underfitting).
#   - HIGH Gamma: Like laser pointers on posts. Influence is very local. Fence
#     becomes very WIGGLY, fitting training points closely.
#     Good for: Complex patterns. Risk: Following noise (overfitting).
#
# C (Penalty) - Strictness of Builder:
#   - LOW C: Relaxed builder. Allows some training points on the wrong side if it
#     helps make a WIDER, simpler fence path overall.
#     Good for: Noisy data, avoiding overfitting. Risk: Too simple (underfitting).
#   - HIGH C: Perfectionist builder. Tries VERY hard to classify all training points
#     correctly, even if it means a NARROWER, more complex fence.
#     Good for: Clean data where points are reliable. Risk: Too complex (overfitting).
#
# BIC/ISQ Example:
#   - High Gamma might perfectly circle outlier implants (e.g., a failed one with high ISQ)
#     but might be too specific to this data (overfitting).
#   - High C would force the boundary to bend sharply to classify borderline BIC/ISQ cases,
#     risking overfitting if those cases aren't representative.
#   - Low Gamma/Low C leads to a smoother, more general boundary, potentially better for
#     predicting future implants but might miss subtle patterns.
#
# Goal: Find a balance using techniques like GridSearchCV.

# --- Visualizing C and Gamma Effects (RBF Kernel) ---

# Plotting function (modified for detailed text)
def plot_rbf_demo_boundary(clf, X, y, axes, title, explanation):
    # Check if scaler exists, handle case where pipeline might just be SVC
    if hasattr(clf, 'steps'): # It's a pipeline
        scaler = clf.named_steps.get('scaler')
        svm_clf = clf.named_steps.get('svm_clf')
        if svm_clf is None:
             print("Error: Could not find svm_clf step in pipeline")
             return
    else: # Assume it's just the SVC classifier
        scaler = None
        svm_clf = clf
        if not isinstance(svm_clf, svm.SVC):
            print("Error: Expected Pipeline or svm.SVC")
            return

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # Use finer mesh for smoother contours
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), 
                         np.arange(y_min, y_max, 0.05))
                         
    # Prepare grid points for prediction (scale if scaler exists)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    if scaler:
        grid_points_scaled = scaler.transform(grid_points)
        Z = svm_clf.predict(grid_points_scaled)
    else:
        Z = svm_clf.predict(grid_points) # Predict on original if no scaler
        
    Z = Z.reshape(xx.shape)
    # Use contourf for filled contours, contour for lines if needed
    cs = axes.contourf(xx, yy, Z, cmap='RdBu', alpha=0.2) 
        
    # Scatter plot for training data
    scatter = axes.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='k', s=30)
    
    axes.set_xlabel('Feature 1 (e.g., BIC - Scaled if pipeline)') 
    axes.set_ylabel('Feature 2 (e.g., ISQ - Scaled if pipeline)')
    axes.set_title(title, fontsize=11) # Slightly smaller title
    axes.tick_params(axis='both', which='both', bottom=False, top=False,
                     left=False, right=False, labelbottom=False, labelleft=False)
    axes.set_xlim(xx.min(), xx.max())
    axes.set_ylim(yy.min(), yy.max())
    # Add explanation text below plot - adjusted y position and fontsize
    axes.text(0.5, -0.15, explanation, # Lowered y-position further below axes
              ha='center', va='top', transform=axes.transAxes, 
              fontsize=8.5, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4)) # Applied text_props here

# --- Main Execution (for standalone demo) ---
if __name__ == "__main__":
    # --- Generate Data (Only if run directly) ---
    # Using make_blobs similar to baseline script for consistency in visualization
    print("Generating simulated data for visualization demo...")
    centers = [[55, 70], [35, 55]] # Successful (1) vs. Failed (-1)
    X, y = make_blobs(n_samples=100, centers=centers, cluster_std=10, random_state=42)
    y[y == 0] = -1 # Convert labels to -1, 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"Using {len(X_train)} training samples for C/Gamma visualization.")
    
    print("\n--- Visualizing Effects of C and Gamma (RBF Kernel) ---")

    # Define illustrative values for C and Gamma
    C_low = 0.1
    C_high = 100.0
    gamma_low = 0.1  # Gamma values depend heavily on data scaling
    gamma_high = 10.0 # These values are chosen relative to each other for illustration on scaled data

    # Define shared properties for text boxes
    text_props = dict(boxstyle='round', facecolor='wheat', alpha=0.4)

    # Create the four pipelines with different C and gamma
    pipeline_low_c_low_gamma = Pipeline([("scaler", StandardScaler()), ("svm_clf", svm.SVC(kernel="rbf", C=C_low, gamma=gamma_low, probability=True))])
    pipeline_high_c_low_gamma = Pipeline([("scaler", StandardScaler()), ("svm_clf", svm.SVC(kernel="rbf", C=C_high, gamma=gamma_low, probability=True))])
    pipeline_low_c_high_gamma = Pipeline([("scaler", StandardScaler()), ("svm_clf", svm.SVC(kernel="rbf", C=C_low, gamma=gamma_high, probability=True))])
    pipeline_high_c_high_gamma = Pipeline([("scaler", StandardScaler()), ("svm_clf", svm.SVC(kernel="rbf", C=C_high, gamma=gamma_high, probability=True))])

    # Fit the pipelines
    pipeline_low_c_low_gamma.fit(X_train, y_train)
    pipeline_high_c_low_gamma.fit(X_train, y_train)
    pipeline_low_c_high_gamma.fit(X_train, y_train)
    pipeline_high_c_high_gamma.fit(X_train, y_train)

    # Create a 2x2 plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Effect of C and Gamma on RBF SVM Decision Boundary (Training Data)\n(Red Dots = Success (+1), Blue Dots = Failure (-1))', fontsize=16)

    # Define explanations for each quadrant
    expl_lc_lg = (f"Low C ({C_low}), Low Gamma ({gamma_low})\n" 
                  f"Characteristics: Smooth boundary, Tolerant of errors.\n"
                  f"Risk: Underfitting (too simple for complex BIC/ISQ patterns).\n"
                  f"BIC/ISQ Context: General separation, might misclassify outliers.")

    expl_hc_lg = (f"High C ({C_high}), Low Gamma ({gamma_low})\n"
                  f"Characteristics: Smooth boundary, Strict about errors.\n"
                  f"Risk: Can slightly overfit if single points force boundary shifts.\n"
                  f"BIC/ISQ Context: Tries hard to separate with smooth curve.")

    expl_lc_hg = (f"Low C ({C_low}), High Gamma ({gamma_high})\n"
                  f"Characteristics: Complex boundary, Tolerant of errors.\n"
                  f"Risk: Moderate. Allows complexity but Low C prevents extreme overfitting to outliers.\n"
                  f"BIC/ISQ Context: Allows complex shapes; note smaller region around top-left outlier.")

    expl_hc_hg = (f"High C ({C_high}), High Gamma ({gamma_high})\n"
                  f"Characteristics: Complex boundary, Strict about errors.\n"
                  f"Risk: High Overfitting. Boundary tightly fits training points, including noise/outliers.\n"
                  f"BIC/ISQ Context: Creates specific 'islands' to classify outliers correctly (see top-left).")

    # Plot each scenario with detailed explanation
    plot_rbf_demo_boundary(pipeline_low_c_low_gamma, X_train, y_train, axes[0, 0], 'Low C, Low Gamma', expl_lc_lg)
    plot_rbf_demo_boundary(pipeline_high_c_low_gamma, X_train, y_train, axes[0, 1], 'High C, Low Gamma', expl_hc_lg)
    plot_rbf_demo_boundary(pipeline_low_c_high_gamma, X_train, y_train, axes[1, 0], 'Low C, High Gamma', expl_lc_hg)
    plot_rbf_demo_boundary(pipeline_high_c_high_gamma, X_train, y_train, axes[1, 1], 'High C, High Gamma', expl_hc_hg)

    # Adjust layout AFTER plotting and adding text
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, hspace=0.4, wspace=0.3)
    plt.show()

    print("\nObserve how:")
    print("- Low Gamma creates smoother boundaries (top row).")
    print("- High Gamma creates more complex, localized boundaries (bottom row).")
    print("- Low C allows more misclassifications for a potentially wider margin (left column).")
    print("- High C tries harder to classify points correctly, potentially narrowing margin (right column).")
    print("- The combination (e.g., High C / High Gamma) can lead to overfitting shapes.") 