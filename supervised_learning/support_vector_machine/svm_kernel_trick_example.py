import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline # Using Pipeline for clarity
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix

# --- SVM Kernel Trick Demonstration ---
# Goal: Understand how SVMs handle non-linearly separable data using kernels.

# --- Explanations ---
# Feature Vector: A list of numerical features describing a single data point.
#                 For example, if we are analyzing dental implants based on Bone-Implant
#                 Contact (BIC %) and Implant Stability Quotient (ISQ):
#                 - A single implant with BIC=65% and ISQ=72 would have the
#                   feature vector x = [65, 72].
#                 - The entire dataset X (capital X) would be a matrix where each
#                   row corresponds to one implant's feature vector:
#                   X = [[65, 72],  <- implant 1 (x_1 or X[0])
#                        [58, 60],  <- implant 2 (x_2 or X[1])
#                        [70, 75],  <- implant 3 (x_3 or X[2])
#                        ...       ]
#                 SVMs take these feature vectors as input.
# Kernel Trick: A mathematical shortcut. SVMs need dot products between data points
#               in a high-dimensional feature space (created by a mapping phi(x)).
#               Calculating phi(x) directly can be computationally expensive or impossible
#               (e.g., if the space is infinite-dimensional).
#               Kernels K(x, z) compute this dot product phi(x) . phi(z) efficiently
#               using only the original vectors x and z.
# RBF Kernel (Radial Basis Function Kernel): A popular kernel function that implicitly
#              maps data to an infinite-dimensional space. It measures similarity based
#              on distance; points closer together are more similar. Creates complex,
#              non-linear boundaries. Think of it as placing Gaussian "bumps" around
#              support vectors. The 'gamma' parameter controls the width of these bumps.

# Key Concept Recap (The "Magic" Explained):
# 1. SVMs are inherently LINEAR separators. They find the best flat hyperplane
#    (a line in 2D, a plane in 3D, etc.) in the space they operate in.
# 2. The Kernel Trick: Instead of actually creating potentially huge new feature
#    vectors (mapping data to a higher dimension, phi(x)), kernels compute the
#    *dot products* between these high-dimensional vectors *efficiently*,
#    directly from the original data points: K(x, z) = <phi(x), phi(z)>.
# 3. Non-linear Mapping (phi): The magic lies in the *implicit* non-linear
#    transformation phi suggested by the kernel. For example:
#      - Linear Kernel: phi(x) is just x. The space doesn't change. Boundary is linear.
#      - Polynomial Kernel: phi(x) includes polynomial terms (x1, x2, x1*x2, x1^2, ...).
#        The linear boundary in this polynomial feature space looks curved in the original space.
#      - RBF Kernel: phi(x) maps to an *infinite-dimensional* space using Gaussian functions.
#        A linear boundary in this infinite space can create highly complex, curvy boundaries
#        in the original space.
# 4. Result: The SVM *always* finds a linear boundary in the (potentially high-dimensional)
#    feature space defined by the kernel. When we look at the "shadow" of this boundary
#    back in our original low-dimensional input space, it appears non-linear if a
#    non-linear kernel (like RBF or polynomial) was used. 

def plot_decision_boundary(clf, X, y, axes, title):
    """Plots the decision boundary for a classifier (clf) on data (X, y)."""
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Predict classifications for each point on the grid
    # Important: If clf is a pipeline, it handles scaling internally!
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary (contour) - Using string name for cmap
    axes.contourf(xx, yy, Z, cmap='RdYlBu', alpha=0.3)

    # Plot the data points - Using string name for cmap
    axes.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='k')

    axes.set_xlabel('Feature 1')
    axes.set_ylabel('Feature 2')
    axes.set_title(title)
    axes.tick_params(axis='both', which='both', bottom=False, top=False,
                     left=False, right=False, labelbottom=False, labelleft=False) 

# --- Data Generation --- 
# Create a dataset that is NOT linearly separable
# make_moons is perfect for this.
# X here is a matrix where each row is a FEATURE VECTOR for one data point (moon).
# y is a vector containing the class label (0 or 1) for each point.
X, y = make_moons(n_samples=200, noise=0.15, random_state=42)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Generated {len(X)} samples. Split into {len(X_train)} training and {len(X_test)} testing.")

# Visualize the raw data
plt.figure(figsize=(6, 6))
# Using string name for cmap
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu', edgecolors='k', label='Training Data')
# Using string name for cmap
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdYlBu', edgecolors='grey', marker='s', label='Test Data')
plt.title('Moon-Shaped Dataset (Non-Linear)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show() 

# --- Attempt 1: Linear SVM (Expected Failure) ---
print("\n--- Training Linear SVM ---")

# Create a pipeline: StandardScaler -> Linear SVM
# StandardScaler is important even if the kernel is linear, 
# as the linear SVM algorithm itself is sensitive to feature scales.
linear_svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", svm.SVC(kernel="linear", C=1)) # Using C=1 as a default
])

# Train the pipeline on the training data
linear_svm_pipeline.fit(X_train, y_train)

# Predict on the test data
y_pred_linear = linear_svm_pipeline.predict(X_test)

# Evaluate
accuracy_linear = accuracy_score(y_test, y_pred_linear)
cm_linear = confusion_matrix(y_test, y_pred_linear)

print(f"Linear SVM Test Accuracy: {accuracy_linear:.4f}")
print("Linear SVM Confusion Matrix:")
print(cm_linear)
print("Observation: Accuracy is likely low, as a straight line cannot separate the moons.") 

# --- Attempt 2: RBF Kernel SVM (with Hyperparameter Tuning) ---
print("\n--- Training RBF SVM with GridSearchCV ---")

# Create a pipeline: StandardScaler -> RBF SVM
rbf_svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    # Use the RBF kernel. This allows the SVM to find a non-linear boundary
    # by implicitly mapping the data to a higher (infinite) dimensional space
    # and finding a linear separator there. The kernel trick makes this efficient.
    ("svm_clf", svm.SVC(kernel="rbf")) 
])

# Define the grid of parameters to search
# These ranges are common starting points
param_grid = {
    # C: Penalty for misclassification (Soft Margin parameter).
    'svm_clf__C': [0.1, 1, 10, 100],        
    # gamma: Defines the influence of single training examples. Low values mean
    #        'far' influence (smoother boundary), high values mean 'close' influence
    #        (can lead to overfitting, more complex boundary).
    #        Crucial for the RBF kernel.
    'svm_clf__gamma': [0.01, 0.1, 1, 10]     
}

# Create the GridSearchCV object
# cv=5 means 5-fold cross-validation will be used to evaluate each parameter combo
# n_jobs=-1 uses all available CPU cores for faster searching
grid_search = GridSearchCV(rbf_svm_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit GridSearchCV on the training data
# This performs the search and finds the best parameters
grid_search.fit(X_train, y_train)

# Get the best pipeline found by GridSearchCV
best_rbf_svm_pipeline = grid_search.best_estimator_

print(f"Best parameters found by GridSearchCV: {grid_search.best_params_}")
print(f"Best cross-validation accuracy score: {grid_search.best_score_:.4f}")

# Predict on the test data using the *best* pipeline
y_pred_rbf = best_rbf_svm_pipeline.predict(X_test)

# Evaluate the best RBF model on the test set
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
cm_rbf = confusion_matrix(y_test, y_pred_rbf)

print(f"\nBest RBF SVM Test Accuracy: {accuracy_rbf:.4f}")
print("Best RBF SVM Confusion Matrix:")
print(cm_rbf)
print("Observation: Accuracy should be much higher now! The RBF kernel allows the SVM")
print("to find a non-linear boundary that fits the moon shapes.") 

# --- Visualize Results --- 
fig, axes = plt.subplots(1, 2, figsize=(16, 7)) # Slightly taller figure for text

# Plot Linear SVM boundary
plot_decision_boundary(linear_svm_pipeline, X_test, y_test, axes[0],
                       f'Linear SVM (Accuracy: {accuracy_linear:.2f})')
axes[0].text(0.5, -0.1, # x, y position relative to axes (0,0 is bottom left, 1,1 is top right)
             'A straight line struggles\nto separate the moons.', 
             ha='center', va='top', transform=axes[0].transAxes, fontsize=10)

# Plot RBF SVM boundary
plot_decision_boundary(best_rbf_svm_pipeline, X_test, y_test, axes[1],
                       f'RBF SVM (Accuracy: {accuracy_rbf:.2f})\nBest Params: {grid_search.best_params_}')
axes[1].text(0.5, -0.1, # x, y position relative to axes
             f'The RBF kernel finds a curved\nboundary, fitting the data much better.\n(Best C={grid_search.best_params_["svm_clf__C"]:.1f}, gamma={grid_search.best_params_["svm_clf__gamma"]:.1f})',
             ha='center', va='top', transform=axes[1].transAxes, fontsize=10)


fig.suptitle('SVM Kernel Comparison on Moon Dataset', fontsize=16)
plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust bottom margin for text
plt.show()

print("\nFinal Observation: Compare the plots. The linear SVM draws a straight line,")
print("failing to separate the moons. The RBF SVM, thanks to the kernel trick,")
print("creates a curved boundary that effectively separates the classes, achieving")
print("much higher accuracy. The \"magic\" is the implicit high-dimensional mapping!") 