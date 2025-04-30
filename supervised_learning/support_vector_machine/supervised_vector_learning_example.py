import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Creating an example of supervised vector learning. The goal of this learning algorith is to leverage the dataset and find the optimal values w and b for parameters w and b. 
# Once the learning algorithm identifies this these optimal values, the model f(x) is defined as f(x) = sign(w*x - b*)

# Imagine blue dots (cats) and green dots (dogs) on graph paper
# Draw a straight line that separates them. Many lines work; we want the one that stays farthest from the nearest dots on both sides—the "fatest sidewalk" between teams 
# Only the edge players matter. The dots that kiss the sidewalk's borders are the support vectors; everyone else is irrelevant to the final line.
# Lets identify the support vectors with different colors and labels and also the side of the line that separates the two classes.

# Why care? A wide sidewalk means new dots can wiggle a bit yet still land on the correct side—better generalisation.
# Clear explains Bent sidewalks (kernels): If the dots are hopelessly mixed in 2-D, SVM secretly lifts them into a higher-dimensional space where a straight wall does exist, then projects the decision back down. 
# That magic elevator ride is the kernel trick.

# Key insight: SVM isn't about memorising all dots; it's about finding the smallest crew of boundary dots that maximise breathing room between classes.

# Lets create two examples of a wide sidewalk and a narrow sidewalk and the explanation of how the SVM algorithm works.

# Example 1: Wide sidewalk
# Create a dataset with two features and two classes (e.g., Feature 1 = Fluffiness, Feature 2 = Grumpiness)
# Class 1: Cats (represented by y=1)
# Class -1: Dogs (represented by y=-1)
X_wide = np.array([[1, 2], [2, 3], [3, 3], [6, 5], [7, 6], [8, 7]]) # Slightly adjusted for better visualization
y_wide = np.array([1, 1, 1, -1, -1, -1])

# Train the SVM classifier
# We use a linear kernel because the data seems linearly separable
clf_wide = svm.SVC(kernel='linear', C=1000) # High C value enforces a stricter separation
clf_wide.fit(X_wide, y_wide)

# Plot the dataset and the SVM decision boundary
plt.figure(figsize=(8, 6))
plt.scatter(X_wide[:, 0], X_wide[:, 1], c=y_wide, s=50, cmap='viridis', label='Data Points (Cats/Dogs)')

# Create a mesh grid to plot the decision boundary and margins
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf_wide.decision_function(xy).reshape(XX.shape)

# Plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--']) # Dashed for margins, solid for boundary

# Highlight support vectors
ax.scatter(clf_wide.support_vectors_[:, 0], clf_wide.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k', label='Support Vectors')

plt.xlabel('Feature 1 (e.g., Fluffiness)')
plt.ylabel('Feature 2 (e.g., Grumpiness)')
plt.title('SVM with Wide Margin (Linear Kernel)')
plt.legend()
plt.show()

# Example 2: Narrow sidewalk
# Create a dataset with two features and two classes
X_narrow = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]) # Points closer together
y_narrow = np.array([1, 1, 1, -1, -1, -1])

# Train the SVM classifier
clf_narrow = svm.SVC(kernel='linear', C=1000)
clf_narrow.fit(X_narrow, y_narrow)

# Plot the dataset
plt.figure(figsize=(8, 6))
plt.scatter(X_narrow[:, 0], X_narrow[:, 1], c=y_narrow, s=50, cmap='viridis', label='Data Points (Cats/Dogs)')

# Create a mesh grid to plot the decision boundary and margins
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf_narrow.decision_function(xy).reshape(XX.shape)

# Plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--']) # Dashed for margins, solid for boundary

# Highlight support vectors
ax.scatter(clf_narrow.support_vectors_[:, 0], clf_narrow.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k', label='Support Vectors')


plt.xlabel('Feature 1 (e.g., Fluffiness)')
plt.ylabel('Feature 2 (e.g., Grumpiness)')
plt.title('SVM with Narrow Margin (Linear Kernel)')
plt.legend()
plt.show()




