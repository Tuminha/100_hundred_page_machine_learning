"""
Demonstrates common vector operations based on Chapter 2 concepts of
'The Hundred-Page Machine Learning Book'.

Includes explanations, NumPy implementations, and visualizations for:
- Vector Addition / Subtraction
- Scalar Multiplication
- Dot Product
- Hadamard (Element-wise) Product
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Define Example Vectors (2D for easy visualization) ---
# Think of these as simplified representations, e.g.:
# vec_a could represent [Avg_BIC_Change, Avg_ISQ_Change] after Treatment A
# vec_b could represent [Avg_BIC_Change, Avg_ISQ_Change] after Treatment B

vec_a = np.array([2, 3])
vec_b = np.array([4, -1])
scalar_c = 1.5

print(f"Example Vector A: {vec_a}")
print(f"Example Vector B: {vec_b}")
print(f"Example Scalar C: {scalar_c}")
print("-"*40)

# --- Helper function for plotting vectors ---
def plot_vectors(vectors, colors, labels, title, ax):
    """Plots multiple 2D vectors as arrows from the origin."""
    ax.axhline(0, color='grey', lw=0.5)
    ax.axvline(0, color='grey', lw=0.5)
    max_val = 0
    for i, vec in enumerate(vectors):
        # quiver([origin_x], [origin_y], [vector_x], [vector_y], ...)
        ax.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', scale=1, color=colors[i], label=labels[i])
        max_val = max(max_val, abs(vec[0]), abs(vec[1]))

    # Set plot limits dynamically
    limit = max_val * 1.2 + 1 # Add some padding
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_xlabel("Feature 1 (e.g., BIC Change)")
    ax.set_ylabel("Feature 2 (e.g., ISQ Change)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--')
    ax.set_aspect('equal', adjustable='box') # Ensure axes are equally scaled

# --- 1. Vector Addition & Subtraction --- 
print("\n--- 1. Vector Addition / Subtraction ---")
vec_add = vec_a + vec_b
vec_sub = vec_a - vec_b
print(f"A + B = {vec_a} + {vec_b} = {vec_add}")
print(f"A - B = {vec_a} - {vec_b} = {vec_sub}")

# Visualization
fig_add_sub, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
plot_vectors([vec_a, vec_b, vec_add], ['blue', 'green', 'red'], ['A', 'B', 'A+B'], "Vector Addition", ax1)
plot_vectors([vec_a, vec_b, vec_sub], ['blue', 'green', 'purple'], ['A', 'B', 'A-B'], "Vector Subtraction", ax2)
# Note: Subtraction A-B is like adding A + (-B)
vec_neg_b = -vec_b
ax2.quiver(0, 0, vec_neg_b[0], vec_neg_b[1], angles='xy', scale_units='xy', scale=1, color='orange', label='-B')
ax2.legend()
fig_add_sub.tight_layout()
print("Displaying Vector Addition/Subtraction plots...")
plt.show()
print("-"*40)

# --- 2. Scalar Multiplication --- 
print("\n--- 2. Scalar Multiplication ---")
vec_scaled = scalar_c * vec_a
print(f"C * A = {scalar_c} * {vec_a} = {vec_scaled}")

# Visualization
fig_scalar, ax_scalar = plt.subplots(figsize=(6, 6))
plot_vectors([vec_a, vec_scaled], ['blue', 'red'], [f'A = {vec_a}', f'C*A = {vec_scaled}'], "Scalar Multiplication", ax_scalar)
print("Displaying Scalar Multiplication plot...")
plt.show()
print("-"*40)

# --- 3. Dot Product (Scalar Product) --- 
print("\n--- 3. Dot Product (Scalar Product) ---")
# Concept: Calculates a scalar value representing projection/similarity.
# Formula: Σ (a_i * b_i)
dot_product = np.dot(vec_a, vec_b)
# Alternative: dot_product = vec_a @ vec_b
print(f"A · B = ({vec_a[0]}*{vec_b[0]}) + ({vec_a[1]}*{vec_b[1]}) = {dot_product}")
print(f"Result is a SCALAR: {dot_product}")

# Visualization (Show input vectors, annotate with result)
fig_dot, ax_dot = plt.subplots(figsize=(6, 6))
plot_vectors([vec_a, vec_b], ['blue', 'green'], [f'A = {vec_a}', f'B = {vec_b}'], "Dot Product Input Vectors", ax_dot)
# Add dot product value as text
dot_text = f'A · B = {dot_product}'
ax_dot.text(0.05, 0.95, dot_text, transform=ax_dot.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
print("Displaying Dot Product input vectors plot...")
plt.show()
print("-"*40)

# --- 4. Hadamard (Element-wise) Product --- 
print("\n--- 4. Hadamard (Element-wise) Product ---")
# Concept: Multiplies corresponding elements, results in a vector of the same dimension.
# Notation: a ∘ b or a ⊙ b
hadamard_product = vec_a * vec_b # NumPy's * performs element-wise multiplication
print(f"A ∘ B = [{vec_a[0]}*{vec_b[0]}, {vec_a[1]}*{vec_b[1]}] = {hadamard_product}")
print(f"Result is a VECTOR: {hadamard_product}")

# Visualization
fig_hadamard, ax_hadamard = plt.subplots(figsize=(6, 6))
plot_vectors([vec_a, vec_b, hadamard_product], ['blue', 'green', 'red'],
             [f'A = {vec_a}', f'B = {vec_b}', f'A ∘ B = {hadamard_product}'],
             "Hadamard (Element-wise) Product", ax_hadamard)
print("Displaying Hadamard Product plot...")
plt.show()
print("-"*40) 