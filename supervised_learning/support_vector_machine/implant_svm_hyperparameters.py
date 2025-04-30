import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs, make_circles # To generate synthetic data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix

# --- Dental Implant SVM Example ---
# This script demonstrates Support Vector Machine (SVM) concepts using a
# simulated dental implant dataset.
# We'll predict implant success (1) or failure (-1) based on two features:
# 1. Bone-Implant Contact (BIC) percentage (%)
# 2. Implant Stability Quotient (ISQ)

# --- Dental Implant SVM - Baseline Model Evaluation ---
# This script establishes a baseline performance for an SVM classifier
# on simulated dental implant data (BIC vs ISQ predicting Success/Failure).
# Key steps based on ML best practices:
# 1. Data Splitting: Separate data into training and testing sets.
# 2. Feature Scaling: Use StandardScaler as BIC and ISQ have different ranges.
# 3. Pipeline: Combine scaling and SVM into a single workflow object.
# 4. Baseline Model: Train a simple linear SVM.
# 5. Evaluation: Assess performance on the unseen test set using standard metrics.

# --- SVM Core Concepts Recap (from your image) ---
# 1. Inputs & Outputs:
#    - Input: Feature vector (here: [BIC, ISQ])
#    - Output: Class label (here: 1 for success, -1 for failure)
# 2. Training Loop:
#    - Give SVM the labeled data points ([BIC, ISQ] pairs with known success/failure).
#    - The SVM optimizer adjusts the separating line/plane (defined by parameters w and b)
#      to maximize the margin (the "street width" between the classes)
#      while keeping misclassifications minimal. The C parameter controls this trade-off.
# 3. Prediction:
#    - For a new implant's [BIC, ISQ], plug it into the learned line equation: sign(w*x - b).
#    - The sign (+ or -) determines the predicted class (success or failure).
# 4. Hyperparameters:
#    - C (Penalty): Controls the cost of misclassification.
#        - Low C: Wider margin, tolerates some misclassifications (can underfit).
#        - High C: Narrower margin, tries hard to classify all points correctly (can overfit).
#    - Kernel: Defines the shape of the separating boundary.
#        - 'linear': A straight line/plane.
#        - 'rbf' (Radial Basis Function): A more complex, curved boundary (good for non-linear data).
#    - gamma (for RBF kernel): Influences how far the effect of a single training point reaches.
#        - Low gamma: Broader influence, smoother boundary (can underfit).
#        - High gamma: Localized influence, complex boundary that closely follows data (can overfit). 

def plot_svm_boundary(clf_pipeline, X_train, y_train, X_test, y_test, title, ax):
    """Helper function to plot test data points, decision boundary, margins, and support vectors learned from training data."""
    # Ensure the classifier in the pipeline is fitted
    if not hasattr(clf_pipeline.named_steps['svc'], 'support_vectors_'):
         print("Classifier not fitted yet.")
         # Try fitting here if needed, or handle error
         # For now, just return to avoid error during plot generation if called prematurely
         # A better approach is to ensure it's called *after* fitting.
         # clf_pipeline.fit(X_train, y_train) # Avoid fitting inside plotting function ideally
         return

    # Get the scaler and SVM from the pipeline
    scaler = clf_pipeline.named_steps['standardscaler']
    svm_clf = clf_pipeline.named_steps['svc']

    # Create a mesh grid based on the scaled training data range
    # Important: Transform grid points using the *fitted* scaler for prediction
    X_combined = np.vstack((X_train, X_test))
    xlim = X_combined[:, 0].min() - 1, X_combined[:, 0].max() + 1
    ylim = X_combined[:, 1].min() - 1, X_combined[:, 1].max() + 1

    xx = np.linspace(xlim[0], xlim[1], 50)
    yy = np.linspace(ylim[0], ylim[1], 50)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    # Scale the grid points using the pipeline's scaler
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
    # Create a legend for the scatter plot classes
    handles, labels = scatter.legend_elements()
    # Assuming labels are -1 (Failure) and 1 (Success)
    # Find which original label corresponds to which generated handle
    # This mapping might need adjustment based on `make_blobs` output
    # Let's use a simpler legend for now
    # legend1 = ax.legend(handles=handles, labels=['Failure', 'Success'], title="Test Classes", loc='lower right')
    # ax.add_artist(legend1)

    ax.set_xlabel('Bone-Implant Contact (BIC %)')
    ax.set_ylabel('Implant Stability Quotient (ISQ)')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# --- Data Generation (Simulated) ---
# Using make_blobs to create linearly separable data with some overlap, as before.
# Replace this with your actual data loading when available.
centers = [[55, 70], [35, 55]] # Successful (1) vs. Failed (-1)
X, y = make_blobs(n_samples=100, centers=centers, cluster_std=10, # Increased std dev slightly
                  random_state=42)
y[y == 0] = -1 # Convert labels to -1, 1

# --- Train/Test Split ---
# Split data into 80% for training and 20% for testing.
# stratify=y ensures both train and test sets have proportional class representation.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    random_state=42, stratify=y)

print(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")

# --- Baseline Model Pipeline ---
# Create a pipeline that first scales the data then applies a linear SVM.
# Using C=1 as a standard baseline penalty value.
baseline_model = make_pipeline(
    StandardScaler(),
    svm.SVC(kernel='linear', C=1, random_state=42)
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

# Classification Report (Precision, Recall, F1-Score)
# target_names helps label the output
report = classification_report(y_test, y_pred, target_names=['Failure (-1)', 'Success (1)'])
print("Classification Report:")
print(report)

# Confusion Matrix
# Rows: True Class, Columns: Predicted Class
# [[True Negative, False Positive], [False Negative, True Positive]]
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print(f"(Row=True, Col=Predicted: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]})")

# --- Visualization (Optional but helpful) ---
# Plot the test set results and the decision boundary learned from the training set
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
plot_svm_boundary(baseline_model, X_train, y_train, X_test, y_test,
                  'Baseline Linear SVM: Test Set Results & Decision Boundary', ax)
plt.show()

print("\n--- End of Baseline Evaluation (D-0) ---") 