"""
Demonstrates Python representations (primarily using NumPy) for mathematical
notation concepts covered in Chapter 2 of 'The Hundred-Page Machine Learning Book'.

Includes examples for:
- Scalars
- Vectors (creation, accessing elements)
- Matrices (creation, accessing elements, shape)
- Sigma Notation (Summation using np.sum)
- Pi Notation (Product using np.prod)
- Basic visualization of matrix data
"""

import numpy as np
# Import matplotlib for visualization
import matplotlib.pyplot as plt

print("--- Scalars ---")
# Mathematical Notation: Typically lowercase italic letters (e.g., a, x, λ)
# Concept: A single number.

# General Example: A temperature reading
temperature = 25.3
print(f"General Example (Scalar - Temperature): {temperature}")

# Implant Dentistry Example: A single ISQ measurement or BIC value
implant_isq = 72
implant_bic = 68.5
print(f"Dental Example (Scalar - ISQ): {implant_isq}")
print(f"Dental Example (Scalar - BIC): {implant_bic}\n")


print("--- Vectors ---")
# Mathematical Notation: Lowercase bold (x) or arrow (vec{x}). Elements x_i or x(i).
# Concept: An ordered list of numbers (features of a single observation).

# General Example: Shopping list quantities [apples, bread, eggs]
shopping_list_py = [3, 1, 6] # Using Python list
shopping_list_np = np.array([3, 1, 6]) # Using NumPy array (preferred for ML)
print(f"General Example (Vector - Python List): {shopping_list_py}")
print(f"General Example (Vector - NumPy Array): {shopping_list_np}")

# Accessing elements (0-based indexing in Python/NumPy)
# Math: x_1 (first element), x_3 (third element)
# Python/NumPy: array[0], array[2]
print(f" Accessing first element (NumPy): {shopping_list_np[0]}")
print(f" Accessing third element (NumPy): {shopping_list_np[2]}")

# Implant Dentistry Example: Features for one implant [BIC, ISQ, Torque]
implant_features = np.array([65.5, 70, 35]) # Example: BIC=65.5, ISQ=70, Torque=35
print(f"Dental Example (Vector - Implant Features): {implant_features}")
print(f" Dental Example - ISQ value (2nd element): {implant_features[1]}\n")


print("--- Matrices ---")
# Mathematical Notation: Uppercase bold (X, W). Elements X_ij or X(i, j).
# Concept: A rectangular grid of numbers (rows=observations, columns=features).

# General Example: Hours worked [Person x Day]
# Person 1: Mon=8, Tue=7, Wed=8
# Person 2: Mon=6, Tue=8, Wed=7
hours_worked = np.array([
    [8, 7, 8], # Row 0 (Person 1)
    [6, 8, 7]  # Row 1 (Person 2)
])
print(f"General Example (Matrix - Hours Worked):\n{hours_worked}")

# Shape of the matrix (rows, columns)
print(f" Shape of matrix: {hours_worked.shape}") # Output: (2, 3) -> 2 rows, 3 columns

# Accessing elements (row_index, column_index) - 0-based
# Math: X_1,2 (row 1, col 2) -> Person 1, Tue
# NumPy: matrix[0, 1]
print(f" Accessing Person 1, Tue hours: {hours_worked[0, 1]}") # Output: 7

# Implant Dentistry Example: Data for multiple implants
# Rows = Implants, Columns = Features (BIC, ISQ, Torque)
implant_data_matrix = np.array([
    [65.5, 70, 35], # Implant 1 (Row 0)
    [70.1, 75, 40], # Implant 2 (Row 1)
    [62.0, 68, 30]  # Implant 3 (Row 2)
])
print(f"\nDental Example (Matrix - Implant Dataset):\n{implant_data_matrix}")
print(f" Shape of dataset: {implant_data_matrix.shape}") # Output: (3, 3) -> 3 implants, 3 features
print(f" Accessing Implant 2's ISQ value: {implant_data_matrix[1, 1]}\n") # Output: 75

# Visualize the first two features (BIC vs ISQ) using a scatter plot
bic_values = implant_data_matrix[:, 0] # Column 0 = BIC
isq_values_matrix = implant_data_matrix[:, 1] # Column 1 = ISQ

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(bic_values, isq_values_matrix, marker='o', s=100, label='Implants')
ax.set_xlabel("BIC (%)")
ax.set_ylabel("ISQ")
ax.set_title("Dental Example: Implant Data (BIC vs ISQ)")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)
print("Displaying plot for Matrix Dental Example (BIC vs ISQ)...")
plt.show() # Display the plot


print("\n--- Sigma Notation (Summation Σ) ---")
# Mathematical Notation: Σ_{i=m}^{n} x_i
# Concept: Summing up a sequence of numbers.

# General Example: Sum of the first 5 integers: Σ_{i=1}^{5} i
values_to_sum_gen = np.array([1, 2, 3, 4, 5])
sum_gen = np.sum(values_to_sum_gen)
# Or directly using np.arange: sum_gen = np.sum(np.arange(1, 6))
print(f"General Example (Sum of 1 to 5): {sum_gen}") # Output: 15

# Implant Dentistry Example: Sum of ISQ values for the 3 implants: Σ_{i=1}^{3} ISQ_i
isq_values_sum = implant_data_matrix[:, 1] # Get the ISQ column (index 1)
total_isq = np.sum(isq_values_sum)
print(f"Dental Example (ISQ values): {isq_values_sum}")
print(f"Dental Example (Sum of ISQ values): {total_isq}\n") # Output: 70+75+68 = 213


print("--- Pi Notation (Product Π) ---")
# Mathematical Notation: Π_{i=m}^{n} x_i
# Concept: Multiplying a sequence of numbers.

# General Example: Product of the first 4 integers: Π_{i=1}^{4} i (4!)
values_to_multiply_gen = np.array([1, 2, 3, 4])
product_gen = np.prod(values_to_multiply_gen)
# Or directly using np.arange: product_gen = np.prod(np.arange(1, 5))
print(f"General Example (Product of 1 to 4): {product_gen}") # Output: 24

# Implant Dentistry Example: Joint probability of independent success events
# Assume 3 implants with independent success probabilities: p1=0.9, p2=0.95, p3=0.88
# Probability all succeed = p1 * p2 * p3 = Π_{i=1}^{3} p_i
success_probs = np.array([0.9, 0.95, 0.88])
joint_success_prob = np.prod(success_probs)
print(f"Dental Example (Individual success probs): {success_probs}")
print(f"Dental Example (Joint success probability): {joint_success_prob:.4f}\n") # Output: ~0.7524


print("\n--- Vector/Matrix Operations (Demonstrated Briefly) ---")
# These will be covered more if specific algorithms require them.
# See NumPy documentation for details on vector addition, dot products, etc.
vec_a = np.array([1, 2])
vec_b = np.array([3, 4])
print(f"Vector A: {vec_a}")
print(f"Vector B: {vec_b}")
print(f"Vector Addition (A + B): {vec_a + vec_b}")
print(f"Vector Dot Product (A . B): {np.dot(vec_a, vec_b)}") 