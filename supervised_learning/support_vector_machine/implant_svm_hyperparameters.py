import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs, make_circles # To generate synthetic data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

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

# --- Visualize Results --- 
# ... (Code for plotting baseline results remains here) ...


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
# We'll use the same training data (X_train, y_train) from the baseline section.

print("\n--- Visualizing Effects of C and Gamma (RBF Kernel) ---")

# Define illustrative values for C and Gamma
C_low = 0.1
C_high = 100.0
gamma_low = 0.1  # Gamma values depend heavily on data scaling
gamma_high = 10.0 # These values are chosen relative to each other for illustration on scaled data

# Create the four pipelines with different C and gamma
pipeline_low_c_low_gamma = Pipeline([("scaler", StandardScaler()), ("svm_clf", svm.SVC(kernel="rbf", C=C_low, gamma=gamma_low, probability=True))]) # Added probability=True for potential future use, doesn't harm here
pipeline_high_c_low_gamma = Pipeline([("scaler", StandardScaler()), ("svm_clf", svm.SVC(kernel="rbf", C=C_high, gamma=gamma_low, probability=True))])
pipeline_low_c_high_gamma = Pipeline([("scaler", StandardScaler()), ("svm_clf", svm.SVC(kernel="rbf", C=C_low, gamma=gamma_high, probability=True))])
pipeline_high_c_high_gamma = Pipeline([("scaler", StandardScaler()), ("svm_clf", svm.SVC(kernel="rbf", C=C_high, gamma=gamma_high, probability=True))])

# Fit the pipelines
pipeline_low_c_low_gamma.fit(X_train, y_train)
pipeline_high_c_low_gamma.fit(X_train, y_train)
pipeline_low_c_high_gamma.fit(X_train, y_train)
pipeline_high_c_high_gamma.fit(X_train, y_train)

# Create a 2x2 plot
fig, axes = plt.subplots(2, 2, figsize=(16, 14)) # Keep slightly larger height
fig.suptitle('Effect of C and Gamma on RBF SVM Decision Boundary (Training Data)\n(Red Dots = Success (+1), Blue Dots = Failure (-1))', fontsize=16)

# Define shared properties for text boxes
text_props = dict(boxstyle='round', facecolor='wheat', alpha=0.4) # Slightly increased alpha

# Plotting function (modified for detailed text)
def plot_rbf_demo_boundary(clf, X, y, axes, title, explanation):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # Use finer mesh for smoother contours
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), 
                         np.arange(y_min, y_max, 0.05))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Use contourf for filled contours, contour for lines if needed
    cs = axes.contourf(xx, yy, Z, cmap='RdBu', alpha=0.2) 
        
    # Scatter plot for training data
    scatter = axes.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='k', s=30)
    
    axes.set_xlabel('BIC (% - Scaled)') 
    axes.set_ylabel('ISQ (Scaled)')
    axes.set_title(title, fontsize=11) # Slightly smaller title
    axes.tick_params(axis='both', which='both', bottom=False, top=False,
                     left=False, right=False, labelbottom=False, labelleft=False)
    axes.set_xlim(xx.min(), xx.max())
    axes.set_ylim(yy.min(), yy.max())
    # Add explanation text below plot - adjusted y position and fontsize
    axes.text(0.5, -0.15, explanation, # Lowered y-position further below axes
              ha='center', va='top', transform=axes.transAxes, 
              fontsize=8.5, bbox=text_props) # Reduced fontsize slightly

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
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, hspace=0.4, wspace=0.3) # Added hspace/wspace
# plt.tight_layout(rect=[0, 0.05, 1, 0.92]) # tight_layout can sometimes interfere with manual adjustments
plt.show()

print("\nObserve how:")
print("- Low Gamma creates smoother boundaries (top row).")
print("- High Gamma creates more complex, localized boundaries (bottom row).")
print("- Low C allows more misclassifications for a potentially wider margin (left column).")
print("- High C tries harder to classify points correctly, potentially narrowing margin (right column).")
print("- The combination (e.g., High C / High Gamma) can lead to overfitting shapes.")

# --- End of Baseline Evaluation (D-0) --- 