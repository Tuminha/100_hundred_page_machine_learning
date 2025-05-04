"""
Demonstrates fundamental data structures used in Machine Learning based on
Chapter 2 concepts of 'The Hundred-Page Machine Learning Book'.

Includes conceptual explanations, Python/NumPy representations, and visualizations for:
- Scalar (0D)
- Vector (1D)
- Matrix (2D)
- Tensor (3D+)
"""

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Scalar --- 
print("--- 1. Scalar (0D Tensor) ---")
# Concept: A single numerical value.
# Math Notation: Lowercase italic (e.g., x, a, λ)
# Dimensions: 0

# Python Representation:
scalar_value = 72.5 # e.g., a single ISQ score
print(f"Scalar Value: {scalar_value}")
print(f"Type: {type(scalar_value)}")

# Using NumPy (can create 0D arrays, though less common than Python scalars)
scalar_np = np.array(72.5)
print(f"NumPy Scalar (0D Array) Value: {scalar_np}")
print(f"NumPy Scalar Shape: {scalar_np.shape}") # Output: () - empty tuple for 0D
print(f"NumPy Scalar Dimensions (ndim): {scalar_np.ndim}") # Output: 0
# No direct visualization needed for a single number.
print("-"*25)

# --- 2. Vector --- 
print("\n--- 2. Vector (1D Tensor) ---")
# Concept: An ordered list/array of numbers.
# Often represents features of a single data point or parameters of a model.
# Math Notation: Lowercase bold (e.g., x, w) or arrow (vec{x}). Elements x_i.
# Dimensions: 1 (has length)

# Python/NumPy Representation:
vector_data = np.array([65.5, 70, 35]) # e.g., [BIC, ISQ, Torque] for one implant
print(f"Vector Data: {vector_data}")
print(f"Vector Shape: {vector_data.shape}") # Output: (3,) - length 3
print(f"Vector Dimensions (ndim): {vector_data.ndim}") # Output: 1

# Visualization (as points on a line)
fig_vec, ax_vec = plt.subplots(figsize=(8, 2))
# Plot points on y=0 line, use values as x-coords (or just indices)
ax_vec.scatter(np.arange(len(vector_data)), np.zeros(len(vector_data)), c=vector_data, cmap='viridis', s=100)
ax_vec.plot(np.arange(len(vector_data)), np.zeros(len(vector_data)), 'k--', alpha=0.3) # Line for reference
ax_vec.set_xticks(np.arange(len(vector_data)))
ax_vec.set_xticklabels([f'Feature {i+1}\n({val})' for i, val in enumerate(vector_data)])
ax_vec.set_yticks([]) # No y-axis needed
ax_vec.set_title("Vector Visualization (1D Array - e.g., Implant Features)")
plt.tight_layout()
print("Displaying Vector plot...")
plt.show()
print("-"*25)

# --- 3. Matrix --- 
print("\n--- 3. Matrix (2D Tensor) ---")
# Concept: A rectangular grid of numbers (rows and columns).
# Often represents a dataset (rows=samples, columns=features) or model parameters (e.g., weights).
# Math Notation: Uppercase bold (e.g., X, W). Elements X_ij.
# Dimensions: 2 (rows, columns)

# Python/NumPy Representation:
matrix_data = np.array([
    [65.5, 70, 35], # Implant 1 [BIC, ISQ, Torque]
    [70.1, 75, 40], # Implant 2
    [62.0, 68, 30], # Implant 3
    [75.0, 78, 45]  # Implant 4
])
print(f"Matrix Data:\n{matrix_data}")
print(f"Matrix Shape: {matrix_data.shape}") # Output: (4, 3) - 4 rows, 3 columns
print(f"Matrix Dimensions (ndim): {matrix_data.ndim}") # Output: 2

# Visualization (as a heatmap/grid)
fig_mat, ax_mat = plt.subplots(figsize=(5, 5))
im = ax_mat.imshow(matrix_data, cmap='viridis', aspect='auto')

# Add text annotations for values
for i in range(matrix_data.shape[0]):
    for j in range(matrix_data.shape[1]):
        text = ax_mat.text(j, i, matrix_data[i, j], ha="center", va="center", color="w" if matrix_data[i, j] < 70 else "k")

ax_mat.set_xticks(np.arange(matrix_data.shape[1]))
ax_mat.set_yticks(np.arange(matrix_data.shape[0]))
ax_mat.set_xticklabels(['BIC', 'ISQ', 'Torque']) # Example feature labels
ax_mat.set_yticklabels([f'Implant {i+1}' for i in range(matrix_data.shape[0])])
ax_mat.set_title("Matrix Visualization (2D Array - e.g., Dataset)")
fig_mat.colorbar(im, ax=ax_mat, label='Feature Value')
plt.tight_layout()
print("Displaying Matrix plot...")
plt.show()
print("-"*25)

# --- 4. Tensor (Rank 3+) --- 
print("\n--- 4. Tensor (3D+ Tensor) ---")
# Concept: Generalization to N dimensions.
# Often used for images (height, width, channels), video (time, height, width, channels), batches of data.
# Math Notation: Uppercase bold (e.g., X, T). Elements X_ijk...
# Dimensions: 3 or more

# Python/NumPy Representation (Rank 3 Tensor)
# Example: Batch of 2 matrices (like 2 patient datasets, or 2 channels of a simple image)
tensor_data = np.array([
    [[1, 2, 3],  # Sample 1, Feature Group 1 (or Channel 1)
     [4, 5, 6]],
    [[7, 8, 9],  # Sample 1, Feature Group 2 (or Channel 2)
     [10, 11, 12]],

    [[13, 14, 15], # Sample 2, Feature Group 1 (or Channel 1)
     [16, 17, 18]],
    [[19, 20, 21], # Sample 2, Feature Group 2 (or Channel 2)
     [22, 23, 24]]
])
# Let's reshape for a common tensor structure: (samples, rows_per_sample, cols_per_sample)
# Or (batch_size, height, width) for images
tensor_data = tensor_data.reshape((4, 2, 3)) # 4 samples, each is a 2x3 matrix
# Alternative: tensor_data = np.array([tensor_data[0:2], tensor_data[2:4]]) # 2 samples, each with 2 matrices of size 2x3

print(f"Tensor Data (Shape {tensor_data.shape}):\n{tensor_data}")
print(f"Tensor Dimensions (ndim): {tensor_data.ndim}") # Output: 3

# Visualization (as slices/matrices)
num_slices = tensor_data.shape[0] # Visualize slices along the first dimension (samples)
fig_tensor, axes_tensor = plt.subplots(1, num_slices, figsize=(5 * num_slices, 4))
fig_tensor.suptitle(f"Tensor Visualization (Rank {tensor_data.ndim} Array as Slices)", fontsize=14)

for i in range(num_slices):
    ax = axes_tensor[i]
    slice_data = tensor_data[i, :, :] # Get the i-th matrix slice
    im = ax.imshow(slice_data, cmap='viridis', aspect='auto')
    # Add text annotations
    for row in range(slice_data.shape[0]):
        for col in range(slice_data.shape[1]):
            ax.text(col, row, slice_data[row, col], ha="center", va="center", color="w" if slice_data[row, col] < 15 else "k")
    ax.set_title(f"Slice {i} (e.g., Sample {i+1})")
    ax.set_xlabel("Cols")
    ax.set_ylabel("Rows")
    fig_tensor.colorbar(im, ax=ax, label='Value', shrink=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
print("Displaying Tensor plot (showing 2D slices)...")
plt.show()
print("-"*25)

print("\nSummary:")
print("Scalar (0D) -> Vector (1D) -> Matrix (2D) -> Tensor (ND)")
print("NumPy handles these structures efficiently.") 