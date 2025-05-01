"""
Establishes and evaluates a baseline Linear SVM model for simulated
implant data (BIC vs ISQ -> Success/Failure). Follows standard ML practices:
- Data Generation (Simulated)
- Train/Test Split
- Feature Scaling (StandardScaler)
- Pipeline (Scaler + Linear SVM)
- Evaluation (Classification Report, Confusion Matrix)
- Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix

# --- Plotting Helper Function ---
def plot_svm_boundary(clf_pipeline, X_train, y_train, X_test, y_test, title, ax):
    """Helper function to plot test data points, decision boundary, margins, and support vectors learned from training data."""
    # Ensure the classifier in the pipeline is fitted
    # Get SVM step name dynamically
    svm_step_name = None
    for name, step in clf_pipeline.steps:
        if isinstance(step, svm.SVC):
            svm_step_name = name
            break
    if svm_step_name is None:
        print("Error: Could not find svm.SVC step in the pipeline.")
        return
    if not hasattr(clf_pipeline.named_steps[svm_step_name], 'support_vectors_'):
        print("Classifier not fitted yet.")
        return

    # Get the scaler and SVM from the pipeline
    scaler = clf_pipeline.named_steps['standardscaler'] # Assumes name 'standardscaler'
    svm_clf = clf_pipeline.named_steps[svm_step_name]

    # Create a mesh grid based on the original data range for plotting
    X_combined = np.vstack((X_train, X_test))
    xlim = X_combined[:, 0].min() - 1, X_combined[:, 0].max() + 1
    ylim = X_combined[:, 1].min() - 1, X_combined[:, 1].max() + 1

    xx = np.linspace(xlim[0], xlim[1], 50)
    yy = np.linspace(ylim[0], ylim[1], 50)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    # Scale the grid points using the *fitted* scaler for prediction
    xy_scaled = scaler.transform(xy)

    # Get the decision function values from the SVM part of the pipeline
    Z = svm_clf.decision_function(xy_scaled).reshape(XX.shape)

    # Plot decision boundary and margins on the original feature scale
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # Highlight support vectors - plot their *original* (unscaled) values
    # Get support vector indices from the fitted SVM
    support_vector_indices = svm_clf.support_
    # Get the corresponding original training data points
    support_vectors_unscaled = X_train[support_vector_indices]
    ax.scatter(support_vectors_unscaled[:, 0], support_vectors_unscaled[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k', label=f'Support Vectors ({len(support_vectors_unscaled)})')

    # Scatter plot the *test* data points
    scatter = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50, cmap='viridis', label='Test Data Points')

    ax.set_xlabel('Bone-Implant Contact (BIC %)')
    ax.set_ylabel('Implant Stability Quotient (ISQ)')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# --- Main Execution ---
if __name__ == "__main__":
    # --- Data Generation (Simulated) ---
    centers = [[55, 70], [35, 55]] # Successful (1) vs. Failed (-1)
    X, y = make_blobs(n_samples=100, centers=centers, cluster_std=10,
                      random_state=42) # Fixed random_state for reproducibility
    y[y == 0] = -1 # Convert labels to -1, 1

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                        random_state=42, stratify=y) # Fixed random_state

    print(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # --- Baseline Model Pipeline ---
    baseline_model = make_pipeline(
        StandardScaler(),
        svm.SVC(kernel='linear', C=1, random_state=42) # Fixed random_state
    )

    print("\nBaseline Pipeline:")
    print(baseline_model)

    # --- Training ---
    print("\nTraining the baseline model...")
    baseline_model.fit(X_train, y_train)
    print("Training complete.")

    # --- Prediction on Test Set ---
    print("\nMaking predictions on the test set...")
    y_pred = baseline_model.predict(X_test)

    # --- Evaluation ---
    print("\n--- Baseline Linear SVM Performance (on Test Set) ---")
    report = classification_report(y_test, y_pred, target_names=['Failure (-1)', 'Success (1)'])
    print("Classification Report:")
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print(f"(Row=True, Col=Predicted: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]})")

    # --- Visualization ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    plot_svm_boundary(baseline_model, X_train, y_train, X_test, y_test,
                      'Baseline Linear SVM: Test Set Results & Decision Boundary', ax)
    plt.show()

    print("\n--- End of Baseline Evaluation ---") 