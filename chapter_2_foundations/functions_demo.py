"""
Demonstrates the concept of Functions in mathematics and machine learning,
based on Chapter 2 concepts of 'The Hundred-Page Machine Learning Book'.

A function is a rule that maps inputs from a domain to a single, well-defined
output in a codomain.

Visualizes:
- A simple mathematical function (e.g., polynomial)
- An activation function (Sigmoid)
- A loss function (Squared Error for a fixed true value)
"""

import numpy as np
import matplotlib.pyplot as plt

print("--- Concept of a Function ---")
print("A function f: A → B maps inputs from set A (Domain) to outputs in set B (Codomain).")
print("For each input x ∈ A, there is exactly one output y = f(x) ∈ B.")
print("-"*40)

# --- 1. Simple Mathematical Function --- 
print("\n--- 1. Example: A Simple Polynomial Function --- ")
# f(x) = 0.5x^2 - 2x + 1
def simple_polynomial(x):
    return 0.5 * x**2 - 2 * x + 1

x_values_poly = np.linspace(-5, 7, 100)
y_values_poly = simple_polynomial(x_values_poly)

print(f"Example: f(x) = 0.5x^2 - 2x + 1")
print(f"f(0) = {simple_polynomial(0)}")
print(f"f(2) = {simple_polynomial(2)}")
print(f"f(5) = {simple_polynomial(5)}")

fig_poly, ax_poly = plt.subplots(figsize=(8, 5))
ax_poly.plot(x_values_poly, y_values_poly, label='f(x) = 0.5x² - 2x + 1')
ax_poly.set_xlabel("Input (x)")
ax_poly.set_ylabel("Output f(x)")
ax_poly.set_title("Simple Mathematical Function (Polynomial)")
ax_poly.grid(True, linestyle='--')
ax_poly.axhline(0, color='black', lw=0.5)
ax_poly.axvline(0, color='black', lw=0.5)
ax_poly.legend()
print("Displaying Polynomial function plot...")
plt.show()
print("-"*40)


# --- 2. Activation Function (Sigmoid) --- 
print("\n--- 2. Example: Sigmoid Activation Function (Common in ML) ---")
# σ(z) = 1 / (1 + e^-z)
# Domain: ℝ (all real numbers)
# Range: (0, 1) (outputs a value between 0 and 1)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z_values = np.linspace(-10, 10, 200)
sigmoid_values = sigmoid(z_values)

print(f"Sigmoid function: σ(z) = 1 / (1 + e^-z)")
print(f"σ(0) = {sigmoid(0):.2f}")
print(f"σ(5) = {sigmoid(5):.3f}")
print(f"σ(-5) = {sigmoid(-5):.3f}")

fig_sig, ax_sig = plt.subplots(figsize=(8, 5))
ax_sig.plot(z_values, sigmoid_values, label='σ(z) = 1 / (1 + e⁻ᶻ)', color='green')
ax_sig.set_xlabel("Input (z - e.g., weighted sum of features + bias)")
ax_sig.set_ylabel("Output σ(z) (e.g., probability)")
ax_sig.set_title("Activation Function: Sigmoid")
ax_sig.grid(True, linestyle='--')
ax_sig.axhline(0, color='black', lw=0.5)
ax_sig.axhline(0.5, color='grey', lw=0.5, linestyle=':')
ax_sig.axhline(1, color='black', lw=0.5)
ax_sig.axvline(0, color='black', lw=0.5)
ax_sig.legend()
print("Displaying Sigmoid function plot...")
plt.show()
print("-"*40)


# --- 3. Loss Function (Squared Error) --- 
print("\n--- 3. Example: Squared Error Loss Function (for a fixed true value) ---")
# L(y_true, ŷ_pred) = (y_true - ŷ_pred)²
# Here, we fix y_true and see how loss changes with ŷ_pred.

y_true_fixed = 70 # Example: Actual ISQ value

def squared_error_loss(y_true, y_pred):
    return (y_true - y_pred)**2

# Predictions (ŷ) around the true value
y_pred_values = np.linspace(y_true_fixed - 20, y_true_fixed + 20, 200)
loss_values = squared_error_loss(y_true_fixed, y_pred_values)

print(f"Squared Error Loss: L(y_true, ŷ_pred) = (y_true - ŷ_pred)²")
print(f"Assuming y_true = {y_true_fixed}")
print(f"If ŷ_pred = {y_true_fixed}, Loss = {squared_error_loss(y_true_fixed, y_true_fixed)}")
print(f"If ŷ_pred = {y_true_fixed + 10}, Loss = {squared_error_loss(y_true_fixed, y_true_fixed + 10)}")
print(f"If ŷ_pred = {y_true_fixed - 5}, Loss = {squared_error_loss(y_true_fixed, y_true_fixed - 5)}")

fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
ax_loss.plot(y_pred_values, loss_values, label=f'L(ŷ) = ({y_true_fixed} - ŷ)²', color='red')
ax_loss.set_xlabel("Predicted Value (ŷ_pred)")
ax_loss.set_ylabel("Loss L(y_true, ŷ_pred)")
ax_loss.set_title(f"Loss Function: Squared Error (with y_true = {y_true_fixed})")
ax_loss.scatter([y_true_fixed], [0], color='blue', s=100, zorder=5, label=f'True Value ({y_true_fixed}) -> Min Loss')
ax_loss.grid(True, linestyle='--')
ax_loss.axvline(y_true_fixed, color='blue', lw=0.5, linestyle=':')
ax_loss.legend()
print("Displaying Squared Error Loss function plot...")
plt.show()
print("-"*40)


print("\n--- 4. ML Model as a Complex Function (Conceptual) ---")
print("A trained Machine Learning model itself is a function.")
print("Input: Feature vector (e.g., [BIC, ISQ, Torque] for an implant)")
print("Rule: The complex mapping learned by the algorithm (e.g., SVM, Neural Network)")
print("Output: Prediction (e.g., 'Success'/'Failure', or a predicted ISQ score)")
print("Example: f_SVM([65, 70, 30]) --> 'Success'")
print("Example: f_Regression([65, 70, 30]) --> 72.8 (predicted final ISQ)")
print("No single plot captures the entirety of a complex ML model function easily,")
print("but its behavior can be understood by how it maps inputs to outputs.") 