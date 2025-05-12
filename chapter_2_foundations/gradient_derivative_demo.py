"""
Demonstrates the concepts of Derivative and Gradient, with visualizations and dental examples,
based on Chapter 2 concepts of 'The Hundred-Page Machine Learning Book'.

- Derivative: Measures the instantaneous rate of change of a function (1D).
- Gradient: Generalizes the derivative to multivariate functions (vector of partial derivatives).
- Both are foundational for optimization (e.g., gradient descent in ML).

Visualizes:
- Derivative as slope of tangent (1D)
- Gradient as vector (2D)
- Gradient descent path (2D)
"""

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Derivative: Slope of a Function at a Point ---
print("--- 1. Derivative: Slope of a Function at a Point ---")
# Example function: f(x) = x^2 - 4x + 6 (parabola)
def f(x):
    return x**2 - 4*x + 6

def df(x):
    return 2*x - 4

x_vals = np.linspace(-2, 6, 200)
y_vals = f(x_vals)

# Choose a point to visualize the tangent
x0 = 2.0
slope = df(x0)
y0 = f(x0)

# Tangent line at x0: y = f(x0) + slope*(x - x0)
tangent_line = y0 + slope * (x_vals - x0)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_vals, y_vals, label='f(x) = x² - 4x + 6')
ax.plot(x_vals, tangent_line, '--', color='orange', label=f'Tangent at x={x0} (slope={slope:.2f})')
ax.scatter([x0], [y0], color='red', zorder=5, label=f'Point (x₀={x0}, f(x₀)={y0:.2f})')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Derivative as Slope of Tangent')
ax.legend()
ax.grid(True, linestyle='--')
print(f"At x={x0}, f'(x)={slope:.2f} (slope of tangent)")
plt.show()
print("-"*50)

# --- 2. Gradient: Vector of Partial Derivatives (2D) ---
print("\n--- 2. Gradient: Vector of Partial Derivatives (2D) ---")
# Example: f(x, y) = (x-2)^2 + (y-3)^2 (minimum at (2,3))
def f2(x, y):
    return (x-2)**2 + (y-3)**2

def grad_f2(x, y):
    return np.array([2*(x-2), 2*(y-3)])

# Create a grid for visualization
xg, yg = np.meshgrid(np.linspace(-1, 5, 20), np.linspace(0, 6, 20))
zg = f2(xg, yg)

# Compute gradients at grid points (sparse for clarity)
skip = 3
xg_s, yg_s = xg[::skip, ::skip], yg[::skip, ::skip]
U = 2*(xg_s-2)
V = 2*(yg_s-3)

fig2, ax2 = plt.subplots(figsize=(7, 7))
contour = ax2.contour(xg, yg, zg, levels=20, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)
ax2.quiver(xg_s, yg_s, U, V, color='red', angles='xy', scale_units='xy', scale=8, label='Gradient Vectors')
ax2.scatter([2], [3], color='blue', s=80, label='Minimum (2,3)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Gradient as Vector Field (2D)')
ax2.legend()
ax2.grid(True, linestyle='--')
print("Gradient at (2,3) is [0,0] (minimum)")
plt.show()
print("-"*50)

# --- 3. Gradient Descent Path (2D Example) ---
print("\n--- 3. Gradient Descent Path (2D Example) ---")
# Start at an initial point, iteratively move against the gradient
def gradient_descent_2d(f, grad_f, start, lr=0.1, n_iter=20):
    path = [np.array(start)]
    point = np.array(start, dtype=float)
    for _ in range(n_iter):
        grad = grad_f(point[0], point[1])
        point = point - lr * grad
        path.append(point.copy())
    return np.array(path)

start_point = [5.0, 0.0]
path = gradient_descent_2d(f2, grad_f2, start_point, lr=0.2, n_iter=15)

fig3, ax3 = plt.subplots(figsize=(7, 7))
contour = ax3.contour(xg, yg, zg, levels=20, cmap='viridis')
ax3.clabel(contour, inline=True, fontsize=8)
ax3.quiver(xg_s, yg_s, U, V, color='red', angles='xy', scale_units='xy', scale=8, alpha=0.5)
ax3.plot(path[:,0], path[:,1], 'o-', color='magenta', label='Gradient Descent Path')
ax3.scatter([start_point[0]], [start_point[1]], color='orange', s=80, label='Start')
ax3.scatter([2], [3], color='blue', s=80, label='Minimum (2,3)')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title('Gradient Descent Path on 2D Function')
ax3.legend()
ax3.grid(True, linestyle='--')
print(f"Gradient descent path: {path}")
plt.show()
print("-"*50)

print("\n--- Dental Example: ISQ/BIC Optimization ---")
print("In dental implantology, optimizing parameters (e.g., ISQ, BIC) can be framed as minimizing a loss function.")
print("Gradient descent is used to iteratively adjust parameters to improve outcomes, just as shown above.")

# --- 4. Gradient Sensitivity: Elongated Bowl Example ---
print("\n--- 4. Gradient Sensitivity: Elongated Bowl Example ---")
# Function: f(x, y) = (x-2)^2 + 0.1*(y-3)^2
# Much steeper in x than y; gradient in y is small

def f3(x, y):
    return (x-2)**2 + 0.1*(y-3)**2

def grad_f3(x, y):
    return np.array([2*(x-2), 0.2*(y-3)])

xg3, yg3 = np.meshgrid(np.linspace(-1, 5, 20), np.linspace(0, 6, 20))
zg3 = f3(xg3, yg3)

skip = 3
xg3_s, yg3_s = xg3[::skip, ::skip], yg3[::skip, ::skip]
U3 = 2*(xg3_s-2)
V3 = 0.2*(yg3_s-3)

fig4, ax4 = plt.subplots(figsize=(7, 7))
contour3 = ax4.contour(xg3, yg3, zg3, levels=20, cmap='plasma')
ax4.clabel(contour3, inline=True, fontsize=8)
qv = ax4.quiver(xg3_s, yg3_s, U3, V3, color='darkgreen', angles='xy', scale_units='xy', scale=8, label='Gradient Vectors')
ax4.scatter([2], [3], color='blue', s=80, label='Minimum (2,3)')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_title('Gradient Sensitivity: Elongated Bowl (Feature Importance)')
ax4.legend()
ax4.grid(True, linestyle='--')
# Annotate a point where gradient in y is small
annot_x, annot_y = 3.5, 3.0
ax4.annotate('Gradient mostly in x-direction\n(y has little effect)',
             xy=(annot_x, annot_y), xytext=(annot_x+0.5, annot_y+1.5),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
             fontsize=11, color='black', bbox=dict(facecolor='white', alpha=0.7))
print("Notice how the gradient vectors point mostly along x; changing y barely affects f(x, y). Useful for feature importance intuition!")
plt.show()
print("-"*50) 