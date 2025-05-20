"""
Demonstrates the difference between Model-Based and Instance-Based learning algorithms
using synthetic data and conceptual dental examples.

Based on Chapter 2 concepts of 'The Hundred-Page Machine Learning Book'.

Covers:
- Model-Based Learning (e.g., Logistic Regression)
  - Learns an explicit model and its parameters.
  - Uses the model for predictions.
- Instance-Based Learning (e.g., k-Nearest Neighbors - k-NN)
  - Memorizes training instances.
  - Makes predictions by comparing new instances to stored ones.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

print("--- Model-Based vs. Instance-Based Learning Demonstration ---")

# --- 1. Generate Synthetic Dataset ---
# Using make_moons for a non-linearly separable dataset to highlight differences
X, y = make_moons(n_samples=200, noise=0.25, random_state=42)

# Standardize features
X = StandardScaler().fit_transform(X)

# Split into training and test sets (though for k-NN, 'training' is just storing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Plotting function for decision boundaries
def plot_decision_boundary(ax, clf, X, y, title, resolution=0.02):
    """Plots the decision boundary of a classifier."""
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    ax.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())

    # Plot all samples
    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                   alpha=0.8, c=colors[idx],
                   marker=markers[idx], label=f'Class {cl}', edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--')

# --- 2. Model-Based Learning: Logistic Regression ---
print("\n--- Model-Based Learning: Logistic Regression ---")
print("Logistic Regression learns a linear boundary (in the space of linear combinations of features)")
print("to separate classes. It explicitly builds a model by finding optimal parameters (coefficients).")

# Initialize and train Logistic Regression model
log_reg = LogisticRegression(solver='liblinear', random_state=42)
log_reg.fit(X_train, y_train)

# --- 3. Instance-Based Learning: k-Nearest Neighbors (k-NN) ---
print("\n--- Instance-Based Learning: k-Nearest Neighbors (k-NN) ---")
print("k-NN does not learn an explicit model. It memorizes training instances.")
print("Predictions are made by finding the 'k' nearest neighbors to a new point")
print("and taking a majority vote of their classes.")

# Initialize and "train" k-NN classifiers (k=1, k=5, k=15)
knn_1 = KNeighborsClassifier(n_neighbors=1)
knn_1.fit(X_train, y_train) # For k-NN, fit() just stores X_train, y_train

knn_5 = KNeighborsClassifier(n_neighbors=5)
knn_5.fit(X_train, y_train)

knn_15 = KNeighborsClassifier(n_neighbors=15)
knn_15.fit(X_train, y_train)

# --- 4. Visualization and Comparison ---
print("\n--- Visualizing Decision Boundaries ---")

fig, axes = plt.subplots(1, 4, figsize=(24, 5))

# Plot dataset
for idx, cl in enumerate(np.unique(y_train)):
    axes[0].scatter(x=X_train[y_train == cl, 0], y=X_train[y_train == cl, 1],
                   alpha=0.8, c=ListedColormap(('red', 'blue'))(idx), # Use map directly for 2 classes
                   marker=('o', 's')[idx], label=f'Class {cl} (Train)', edgecolor='black')
axes[0].set_title('Synthetic Dataset (Moons)')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].legend(loc='upper left')
axes[0].grid(True, linestyle='--')

# Plot Logistic Regression decision boundary
plot_decision_boundary(axes[1], log_reg, X_train, y_train, 'Logistic Regression (Model-Based)')

# Plot k-NN (k=1) decision boundary
plot_decision_boundary(axes[2], knn_1, X_train, y_train, 'k-NN (k=1, Instance-Based)')

# Plot k-NN (k=15) decision boundary
plot_decision_boundary(axes[3], knn_5, X_train, y_train, 'k-NN (k=5, Instance-Based)') # Corrected to knn_5

plt.tight_layout()
print("Displaying plots...")
plt.show()

print("\nObservations:")
print("- Logistic Regression (Model-Based) learns a relatively simple (linear) decision boundary.")
print("- k-NN (Instance-Based) decision boundary can be much more complex and adapt to the local structure of the data.")
print("- k-NN with k=1 can be very sensitive to noise (overfitting), leading to a jagged boundary.")
print("- k-NN with a larger k (e.g., k=5 or k=15) produces a smoother boundary, potentially generalizing better.")
print("- Model-based learners like Logistic Regression aim to find a general rule.")
print("- Instance-based learners like k-NN make decisions based on proximity to stored examples.")

# --- 5. Conceptual Dental Examples (Printed) ---
print("\n--- Conceptual Dental Examples ---")
print("\nModel-Based Example (e.g., Periodontitis Risk Prediction):")
print("  - Algorithm: Train a Logistic Regression model.")
print("  - Training: The model learns weights (parameters) for risk factors like age, plaque index, smoking status from a dataset of patient records.")
print("  - Model: A formula, e.g., log_odds(Periodontitis) = w0 + w1*age + w2*plaque + w3*smoking_status.")
print("  - Prediction: For a new patient, plug their features into the learned formula to get their risk probability.")
print("  - Key: An explicit model (the formula) is built.")

print("\nInstance-Based Example (e.g., Suggesting Treatment Based on Similar Cases):")
print("  - Algorithm: Use k-Nearest Neighbors (k-NN).")
print("  - Training (Memorization): Store a database of past patient cases, including their diagnostic profiles (features) and the treatments they received and outcomes.")
print("  - Prediction: For a new patient, find the 'k' most similar past patients based on their diagnostic profile.")
print("              Suggest treatments or predict outcomes based on what was common/successful for those 'k' neighbors.")
print("  - Key: Relies directly on stored instances and a similarity measure, no explicit global model is learned.")
print("-"*70) 