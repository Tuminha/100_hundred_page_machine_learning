"""
Tunes and evaluates an RBF Kernel SVM model using GridSearchCV for simulated
implant data (BIC vs ISQ -> Success/Failure).

Compares performance against the baseline linear model.
Steps:
- Data Generation (Simulated - same as baseline)
- Train/Test Split
- Feature Scaling (StandardScaler)
- RBF SVM Pipeline
- Hyperparameter Grid Definition (C, gamma)
- GridSearchCV Execution (5-fold CV, F1 Macro scoring)
- Best Parameter Reporting & Saving
- Final Model Training (Implicit in GridSearchCV refit)
- Test Set Evaluation (Classification Report, Confusion Matrix)
- Visualization of Best RBF Model Boundary
"""

import numpy as np
import matplotlib.pyplot as plt
import json # To save parameters
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline # Use Pipeline instead of make_pipeline for explicit naming
from sklearn.metrics import classification_report, confusion_matrix

# --- Plotting Helper Function (Adapted for RBF - No Margins) ---
def plot_rbf_boundary(clf_pipeline, X_train, y_train, X_test, y_test, title, ax):
    """Plots the decision boundary for an RBF SVM pipeline on test data."""
    # Ensure the classifier in the pipeline is fitted
    svm_step_name = 'svm_clf' # Assuming this name in the pipeline
    if not hasattr(clf_pipeline.named_steps[svm_step_name], 'support_vectors_'):
        print("Classifier not fitted yet.")
        return

    scaler = clf_pipeline.named_steps['scaler'] # Assuming name 'scaler'
    svm_clf = clf_pipeline.named_steps[svm_step_name]

    # Create mesh grid
    X_combined = np.vstack((X_train, X_test))
    xlim = X_combined[:, 0].min() - 1, X_combined[:, 0].max() + 1
    ylim = X_combined[:, 1].min() - 1, X_combined[:, 1].max() + 1

    xx = np.linspace(xlim[0], xlim[1], 100) # Finer mesh for RBF
    yy = np.linspace(ylim[0], ylim[1], 100)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    # Scale grid points and predict
    xy_scaled = scaler.transform(xy)
    Z = svm_clf.predict(xy_scaled).reshape(XX.shape)

    # Plot decision boundary contour
    ax.contourf(XX, YY, Z, cmap='viridis', alpha=0.3)

    # Highlight support vectors (optional, can be many for RBF)
    # support_vectors_unscaled = X_train[svm_clf.support_]
    # ax.scatter(support_vectors_unscaled[:, 0], support_vectors_unscaled[:, 1], s=100,
    #            linewidth=1, facecolors='none', edgecolors='k', label=f'Support Vectors ({len(support_vectors_unscaled)})')

    # Scatter plot the *test* data points
    scatter = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50, cmap='viridis', edgecolors='k', label='Test Data Points')

    ax.set_xlabel('Bone-Implant Contact (BIC %)')
    ax.set_ylabel('Implant Stability Quotient (ISQ)')
    ax.set_title(title)
    # ax.legend(loc='upper left') # Legend can get crowded, title has info
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# --- Main Execution ---
if __name__ == "__main__":
    # --- Data Generation (Simulated - Same as baseline_linear_svm.py) ---
    centers = [[55, 70], [35, 55]] # Successful (1) vs. Failed (-1)
    X, y = make_blobs(n_samples=100, centers=centers, cluster_std=10,
                      random_state=42) # Fixed random_state for reproducibility
    y[y == 0] = -1 # Convert labels to -1, 1

    # --- Train/Test Split (Same as baseline_linear_svm.py) ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                        random_state=42, stratify=y) # Fixed random_state

    print(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # --- RBF SVM Pipeline Definition ---
    # We define steps with names for easier parameter access in GridSearchCV
    rbf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm_clf', svm.SVC(kernel='rbf', probability=True, random_state=42)) # probability=True often useful, random_state for reproducibility
    ])

    print("\nRBF SVM Pipeline:")
    print(rbf_pipeline)

    # --- Hyperparameter Grid Definition ---
    # Define ranges for C and gamma, often logarithmic spacing is effective
    param_grid = {
        'svm_clf__C': [0.001, 0.01, 0.1, 1, 10, 100], # 10^-3 to 10^2
        'svm_clf__gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10] # 10^-4 to 10^1
    }
    print("\nParameter Grid for GridSearchCV:")
    print(param_grid)

    # --- GridSearchCV Execution ---
    print("\nRunning GridSearchCV (5-fold CV, F1 Macro scoring)...")
    # n_jobs=-1 uses all available CPU cores
    grid_search = GridSearchCV(rbf_pipeline, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1) # Added verbose
    grid_search.fit(X_train, y_train)
    print("GridSearchCV complete.")

    # --- Best Parameters Reporting & Saving ---
    print("\n--- GridSearchCV Results ---")
    print(f"Best Parameters Found: {grid_search.best_params_}")
    print(f"Best Cross-Validation F1 Macro Score: {grid_search.best_score_:.4f}")

    # Save best parameters to a JSON file
    params_file = 'best_rbf_params.json'
    print(f"Saving best parameters to {params_file}...")
    with open(params_file, 'w') as f:
        json.dump(grid_search.best_params_, f, indent=4)
    print("Parameters saved.")

    # --- Final Model Evaluation on Test Set ---
    print("\n--- Tuned RBF SVM Performance (on Test Set) ---")
    # Get the best estimator (pipeline) trained on the full training set (refit=True by default)
    best_model = grid_search.best_estimator_

    # Predict on the test set
    y_pred_tuned = best_model.predict(X_test)

    # Evaluate
    report_tuned = classification_report(y_test, y_pred_tuned, target_names=['Failure (-1)', 'Success (1)'])
    print("Classification Report (Tuned RBF SVM):")
    print(report_tuned)

    cm_tuned = confusion_matrix(y_test, y_pred_tuned)
    print("\nConfusion Matrix (Tuned RBF SVM):")
    print(cm_tuned)
    print(f"(Row=True, Col=Predicted: TN={cm_tuned[0,0]}, FP={cm_tuned[0,1]}, FN={cm_tuned[1,0]}, TP={cm_tuned[1,1]})")

    # --- Visualization of Best Tuned Model ---
    print("\nVisualizing the best tuned RBF SVM model...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    plot_title = (f"Tuned RBF SVM: Test Set Results\n" 
                  f"Best Params: C={grid_search.best_params_['svm_clf__C']}, gamma={grid_search.best_params_['svm_clf__gamma']:.4f}\n" 
                  f"CV F1 Macro: {grid_search.best_score_:.3f}, Test F1 Macro: {classification_report(y_test, y_pred_tuned, output_dict=True)['macro avg']['f1-score']:.3f}")
    plot_rbf_boundary(best_model, X_train, y_train, X_test, y_test,
                      plot_title, ax)
    plt.show()

    print("\n--- End of Tuned RBF SVM Evaluation (D-1) ---") 