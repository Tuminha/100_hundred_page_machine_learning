"""
Demonstrates the conceptual difference between Shallow and Deep Learning algorithms
using a synthetic dataset and simple model implementations from scikit-learn.

Based on Chapter 2 concepts of 'The Hundred-Page Machine Learning Book'.

Covers:
- Shallow Learning (e.g., Logistic Regression)
  - Simpler architecture, often relies on engineered features.
- Deep Learning (represented by a Multi-Layer Perceptron - MLP)
  - Deeper architecture with hidden layers, capable of learning more complex patterns and features.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_circles, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

print("--- Shallow vs. Deep Learning Demonstration (Conceptual) ---")

# --- 1. Generate Synthetic Dataset ---
# Using make_circles, which is non-linearly separable and challenging for simple linear models.
X, y = make_circles(n_samples=300, noise=0.15, factor=0.5, random_state=42)

# Standardize features
X = StandardScaler().fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Plotting function for decision boundaries (reused from previous script, slightly adapted)
def plot_decision_boundary(ax, clf, X_plot, y_plot, title, resolution=0.02):
    """Plots the decision boundary of a classifier on given X_plot, y_plot data."""
    markers = ('o', 's') # Circles and squares for two classes
    colors = ('red', 'blue')
    cmap = ListedColormap(colors)

    # Plot the decision surface
    x1_min, x1_max = X_plot[:, 0].min() - 0.5, X_plot[:, 0].max() + 0.5
    x2_min, x2_max = X_plot[:, 1].min() - 0.5, X_plot[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.array([xx1.ravel(), xx2.ravel()]).T)
    elif hasattr(clf, "predict_proba"):
        Z = clf.predict_proba(np.array([xx1.ravel(), xx2.ravel()]).T)[:, 1]
    else:
        Z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    
    Z = Z.reshape(xx1.shape)
    ax.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap, levels=np.linspace(Z.min(), Z.max(), 3))
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y_plot)):
        ax.scatter(x=X_plot[y_plot == cl, 0], y=X_plot[y_plot == cl, 1],
                   alpha=0.8, c=colors[idx],
                   marker=markers[idx], label=f'Class {cl}', edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--')

# --- 2. Shallow Learning Example: Logistic Regression ---
print("\n--- Shallow Learning: Logistic Regression ---")
print("Logistic Regression is a shallow model. It tries to find a linear separation.")
log_reg_shallow = LogisticRegression(solver='liblinear', random_state=42)
log_reg_shallow.fit(X_train, y_train)

# --- 3. "Deep" Learning Example (Conceptual): Multi-Layer Perceptron (MLP) ---
print("\n--- \"Deep\" Learning (Conceptual): Multi-Layer Perceptron (MLP) ---")
print("An MLP with hidden layers can learn more complex, non-linear decision boundaries.")
print("This is a simple MLP, true deep learning models are often much larger and more complex.")
mlp_deep = MLPClassifier(hidden_layer_sizes=(10, 5), activation='relu', solver='adam',
                         max_iter=1000, random_state=42, early_stopping=False)
mlp_deep.fit(X_train, y_train)

# --- 4. Visualization and Comparison (Separate Plots) ---
print("\n--- Visualizing Decision Boundaries (Separate Plots) ---")

# Plot 1: Original Dataset
fig1, ax1 = plt.subplots(figsize=(7, 6))
for idx, cl in enumerate(np.unique(y)):
    ax1.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
               alpha=0.8, c=ListedColormap(('red', 'blue'))(idx),
               marker=('o', 's')[idx], label=f'Class {cl}', edgecolor='black')
ax1.set_title('Synthetic Dataset (Circles)')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle='--')
plt.tight_layout()
print("Displaying Plot 1: Original Dataset...")
plt.show()

# Plot 2: Logistic Regression (Shallow) decision boundary
fig2, ax2 = plt.subplots(figsize=(7, 6))
plot_decision_boundary(ax2, log_reg_shallow, X_train, y_train, 'Logistic Regression (Shallow)')
plt.tight_layout()
print("Displaying Plot 2: Logistic Regression Decision Boundary...")
plt.show()

# Plot 3: MLP (Deep) decision boundary
fig3, ax3 = plt.subplots(figsize=(7, 6))
plot_decision_boundary(ax3, mlp_deep, X_train, y_train, 'MLP Classifier (Simple Deep Model)')
plt.tight_layout()
print("Displaying Plot 3: MLP Decision Boundary...")
plt.show()

print("\nObservations:")
print("- The 'circles' dataset is not linearly separable.")
print("- Logistic Regression (shallow model) struggles to find a good boundary, resulting in poor separation.")
print("- The MLP (even a simple one representing deep learning) can learn a non-linear boundary and separate the classes much better.")
print("- This illustrates how deeper architectures can capture more complex patterns.")

# --- 5. Conceptual Dental Examples (Printed) ---
print("\n--- Conceptual Dental Examples ---")
print("\nShallow Learning Example (e.g., Predicting Implant Stability Based on Key Indicators):")
print("  - Model: Logistic Regression or Support Vector Machine (SVM).")
print("  - Features: Manually selected, well-understood clinical parameters like initial ISQ, insertion Torque, bone density (e.g., Hounsfield Units from CBCT), and perhaps BIC (Bone-Implant Contact) if measurable.")
print("  - Learning: Model learns weights for these specific features to predict implant success/failure or a stability category.")
print("  - Simplicity: Relatively interpretable, faster to train, good if the relationship between these key indicators and outcome is fairly direct or can be captured with careful feature engineering.")

print("\nDeep Learning Example (e.g., Advanced Implant Outcome Prediction or Radiographic Analysis):")
print("  - Model: Convolutional Neural Network (CNN) for radiographs, or a Recurrent Neural Network (RNN/LSTM) for time-series data (e.g., ISQ changes over time).")
print("  - Features:")
print("    - For CNNs: Raw pixel data from dental X-rays or 3D CBCT scans (to detect subtle bone changes, inflammation patterns around implants). The CNN learns relevant visual features automatically.")
print("    - For RNNs/LSTMs: Time-series data of ISQ, torque, BIC, patient-reported outcomes, etc., collected at multiple follow-ups. The model learns temporal patterns.")
print("  - Learning: The network automatically learns hierarchical features. For instance, a CNN might learn edges -> simple textures -> complex patterns indicative of osseointegration quality or potential issues.")
print("  - Complexity: Can model very complex, non-linear relationships and discover novel predictive patterns from raw data. Requires large datasets and significant computational power (GPUs). Less directly interpretable but can achieve higher accuracy for complex prediction tasks.")
print("-"*70) 