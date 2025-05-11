"""
Demonstrates the Assignment Operator (:= or ← in pseudocode, = in Python)
and its effect in iterative processes, based on Chapter 2 concepts of
'The Hundred-Page Machine Learning Book'.

Visualizes:
- Cumulative sum using iterative assignment.
- Simplified 1D "gradient descent" like parameter update.
"""

import numpy as np
import matplotlib.pyplot as plt

print("--- Assignment Operator (:= or ← in pseudocode, = in Python) ---")
# Concept: Assigns or updates the value of a variable.
# Pseudocode:  variable := new_value  OR  variable ← new_value
# Python:     variable = new_value

# Basic Assignment:
# Pseudocode: a := 5
# Python:
a = 5
print(f"Basic assignment: a = {a}")

# Pseudocode: b := a + 3
# Python:
b = a + 3
print(f"Assignment using an expression: b = a + 3 = {b}")
print("-"*50)


# --- 1. Iterative Assignment: Cumulative Sum --- 
print("\n--- 1. Visualizing Iterative Assignment: Cumulative ISQ Change ---")
# Example: Tracking total ISQ change over several implant checkups.

isq_changes_observed = [5, -2, 8, 3, -1, 6] # Positive for increase, negative for decrease
cumulative_isq_change = 0 # Pseudocode: total_isq_change := 0
cumulative_history = [cumulative_isq_change] # To store history for plotting

print(f"Initial cumulative_isq_change: {cumulative_isq_change}")
print(f"Observed ISQ changes at checkups: {isq_changes_observed}")

for i, change in enumerate(isq_changes_observed):
    # Pseudocode: cumulative_isq_change := cumulative_isq_change + change
    cumulative_isq_change = cumulative_isq_change + change
    cumulative_history.append(cumulative_isq_change)
    print(f"After checkup {i+1} (change={change:+}), cumulative_isq_change = {cumulative_isq_change}")

# Visualization
fig_cum, ax_cum = plt.subplots(figsize=(8, 5))
ax_cum.plot(range(len(cumulative_history)), cumulative_history, marker='o', linestyle='-', label='Cumulative ISQ Change')
ax_cum.set_xlabel("Number of Checkups (Assignments)")
ax_cum.set_ylabel("Total ISQ Change")
ax_cum.set_title("Effect of Iterative Assignment: Cumulative Sum")
ax_cum.set_xticks(range(len(cumulative_history)))
ax_cum.grid(True, linestyle='--')
ax_cum.legend()
print("Displaying Cumulative ISQ Change plot...")
plt.show()
print("-"*50)


# --- 2. Iterative Assignment: Simplified 1D "Gradient Descent" --- 
print("\n--- 2. Visualizing Iterative Assignment: Simplified Parameter Update ---")
# Goal: Find 'w' that minimizes a simple function f(w) = (w - 5)^2
# The minimum is at w = 5, where f(w) = 0.
# Gradient (derivative) of f(w) is f'(w) = 2*(w - 5).
# Update rule (pseudocode): w := w - learning_rate * f'(w)

def f_simple(w):
    return (w - 5)**2

def grad_f_simple(w):
    return 2 * (w - 5)

# Hyperparameters for the optimization process
learning_rate = 0.1
num_iterations = 25

# Initial assignment
w_current = 0.0 # Pseudocode: w := 0.0

w_history = [w_current]
loss_history = [f_simple(w_current)]

print(f"Optimizing f(w) = (w - 5)^2 using iterative assignment (simplified gradient descent).")
print(f"Initial w: {w_current:.2f}, Initial f(w): {f_simple(w_current):.2f}")

for i in range(num_iterations):
    gradient = grad_f_simple(w_current)
    # Iterative assignment using the update rule:
    # Pseudocode: w_current := w_current - learning_rate * gradient
    w_current = w_current - learning_rate * gradient
    
    w_history.append(w_current)
    current_loss = f_simple(w_current)
    loss_history.append(current_loss)
    if (i + 1) % 5 == 0 or i == 0:
        print(f"Iter {i+1:2d}: w ≈ {w_current:.3f}, f(w) ≈ {current_loss:.3f}, grad ≈ {gradient:.3f}")

# Visualization
fig_gd, (ax_w, ax_loss) = plt.subplots(1, 2, figsize=(14, 6))
fig_gd.suptitle("Effect of Iterative Assignment: Simplified 1D Optimization", fontsize=16)

# Plot evolution of w
ax_w.plot(w_history, marker='.', linestyle='-', label='Value of w over iterations')
ax_w.axhline(5, color='red', linestyle='--', label='Target w (minimum of f(w))')
ax_w.set_xlabel("Iteration (Assignment Step)")
ax_w.set_ylabel("Parameter w")
ax_w.set_title("Parameter w Converging to Optimum")
ax_w.legend()
ax_w.grid(True)

# Plot evolution of loss f(w)
ax_loss.plot(loss_history, marker='.', linestyle='-', color='green', label='Loss f(w) over iterations')
ax_loss.axhline(0, color='red', linestyle='--', label='Minimum Loss')
ax_loss.set_xlabel("Iteration (Assignment Step)")
ax_loss.set_ylabel("Loss f(w) = (w-5)²")
ax_loss.set_title("Loss f(w) Decreasing to Minimum")
ax_loss.legend()
ax_loss.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
print("Displaying 1D Optimization plots...")
plt.show()
print("-"*50)

print("Key takeaway: The assignment operator (`=`) in Python facilitates the step-by-step updates")
print("that are central to how many machine learning algorithms learn and optimize.") 