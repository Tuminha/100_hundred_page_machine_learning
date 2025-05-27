"""
Introduction to Logistic Regression - Concepts and Formulas.

Based on Chapter 3: Fundamental Algorithms (Logistic Regression)
from 'The Hundred-Page Machine Learning Book' by Andriy Burkov,
and personal study notes.

This script covers the initial concepts of Logistic Regression:
- What is Logistic Regression (Classification, Sigmoid function for probability)
- Data and Notation (Binary target variable)
- The Model Formulas (z-score, Sigmoid, Binary Cross-Entropy Loss)

This script is a work in progress and reflects the initial sections of
the learning material on Logistic Regression.
"""

import numpy as np
import matplotlib.pyplot as plt

print("--- Chapter 3: Logistic Regression - Introduction & Concepts ---")

# --- 1. What is Logistic Regression? 🎯 ---
print("\n--- 1. What is Logistic Regression? ---")
print("   - General Idea: Logistic Regression is used for binary classification problems.")
print("     It predicts the probability that an input belongs to a particular class (e.g., Class 1 vs. Class 0).")

print("\n   - Core Concept:")
print("     1. Calculate a score (z-score or log-odds) using a linear combination similar to linear regression:")
print("        z = w · x + b  (where w are weights, x are features, b is bias)")
print("     2. Transform this score into a probability (between 0 and 1) using the Sigmoid function (also called logistic function):")
print("        σ(z) = 1 / (1 + e^(-z))")
print("     3. This probability is then used to classify the input. A common threshold is 0.5:")
print("        - If σ(z) > 0.5, predict Class 1.")
print("        - If σ(z) <= 0.5, predict Class 0.")

print("\n   --- Dental Example Teaser (Implant Success/Failure) ---")
print("   - Goal: Classify if a dental implant is likely to succeed (1) or fail (0) based on features.")
print("   - Features (x): e.g., Torque, ISQ value at placement.")
print("   - Training (to be detailed later): The model learns optimal weights (w) and bias (b) by minimizing a loss function")
print("     called Binary Cross-Entropy, often using an optimization algorithm like Gradient Descent.")

print("\n   - Example Prediction (after hypothetical training):")
# Adjusted weights and bias for more illustrative z-scores on a typical sigmoid plot
w_torque_example = 0.1
w_ISQ_example = 0.15
b_example_logistic = -7 # Adjusted bias
print(f"     Adjusted Hypothetical Learned Parameters: w_torque = {w_torque_example}, w_ISQ = {w_ISQ_example}, b = {b_example_logistic}")

# Dental Example 1: Expected High Probability
Torque_ex1 = 40
ISQ_ex1 = 70
z_score_ex1 = (Torque_ex1 * w_torque_example) + (ISQ_ex1 * w_ISQ_example) + b_example_logistic
prob_ex1 = 1 / (1 + np.exp(-z_score_ex1))
print(f"\n   Dental Example 1 (High P): Torque={Torque_ex1}, ISQ={ISQ_ex1}")
print(f"     z1 = ({Torque_ex1}*{w_torque_example}) + ({ISQ_ex1}*{w_ISQ_example}) + ({b_example_logistic}) = {z_score_ex1:.2f}")
print(f"     σ(z1) = {prob_ex1:.4f} -> Likely Success")

# Dental Example 2: Expected Low Probability
Torque_ex2 = 15
ISQ_ex2 = 30
z_score_ex2 = (Torque_ex2 * w_torque_example) + (ISQ_ex2 * w_ISQ_example) + b_example_logistic
prob_ex2 = 1 / (1 + np.exp(-z_score_ex2))
print(f"\n   Dental Example 2 (Low P): Torque={Torque_ex2}, ISQ={ISQ_ex2}")
print(f"     z2 = ({Torque_ex2}*{w_torque_example}) + ({ISQ_ex2}*{w_ISQ_example}) + ({b_example_logistic}) = {z_score_ex2:.2f}")
print(f"     σ(z2) = {prob_ex2:.4f} -> Likely Failure")

# Dental Example 3: Expected Mid-Range Probability
Torque_ex3 = 30
ISQ_ex3 = 40 # Adjusted for z near 0
z_score_ex3 = (Torque_ex3 * w_torque_example) + (ISQ_ex3 * w_ISQ_example) + b_example_logistic
prob_ex3 = 1 / (1 + np.exp(-z_score_ex3))
print(f"\n   Dental Example 3 (Mid P): Torque={Torque_ex3}, ISQ={ISQ_ex3}")
print(f"     z3 = ({Torque_ex3}*{w_torque_example}) + ({ISQ_ex3}*{w_ISQ_example}) + ({b_example_logistic}) = {z_score_ex3:.2f}")
print(f"     σ(z3) = {prob_ex3:.4f} -> Uncertain (near threshold)")


# --- 2. The Building Blocks: Data and Notation 🧱 ---
print("\n\n--- 2. The Building Blocks: Data and Notation ---")
print("   - Dataset Notation: {(xi, yi)}")
print("     - xi: Feature vector for the i-th sample.")
print("     - yi: Binary categorical label for the i-th sample, typically 0 or 1.")
print("       (e.g., 0 for 'Failure', 1 for 'Success'; or 0 for 'Not Risky', 1 for 'Risky').")
print("   - The model learns to predict the probability pi that yi = 1.")
print("   - Example Data Point: {(x_torquei = 35, x_ISQi = 70), yi = 1}")
print("     This represents an implant with Torque=35, ISQ=70, which was a 'Success' (label 1).")

# --- 3. The Model: The Logistic Regression Formulas ⚙️ ---
print("\n\n--- 3. The Model: The Logistic Regression Formulas ---")
print("   --- Formulas for Prediction (on new/unseen data) ---")
print("   1. Calculate the z-score (linear combination or weighted sum):")
print("      z = w · x + b = w1*x1 + w2*x2 + ... + wn*xn + b")
print("      This is the same initial step as in Linear Regression.")

print("\n   2. Apply the Sigmoid Function (σ) to the z-score:")
print("      σ(z) = 1 / (1 + e^(-z))  or  1 / (1 + exp(-z))")
print("      - Output (σ(z)) is a probability value between 0 and 1.")
print("      - This indicates the model's estimated likelihood of the input belonging to Class 1.")

print("\n   3. Make a Classification based on a threshold (commonly 0.5):")
print("      - If σ(z) > 0.5, predict Class 1.")
print("      - If σ(z) <= 0.5, predict Class 0.")

print("\n   --- Loss Function for Training: Binary Cross-Entropy (BCE) Loss ---")
print("   - Purpose: Measures how good the model's predictions (probabilities) are compared to actual binary labels (0 or 1). Used during training to find optimal w and b.")
print("   - Formula for a single observation (y, p), where y is the true label (0 or 1) and p is the predicted probability σ(z) for Class 1:")
print("     L(y, p) = - [y * log(p) + (1 - y) * log(1-p)]")
print("     (log refers to the natural logarithm)")

print("\n   - Breakdown of BCE Loss:")
print("     - If the true label y = 1:")
print("       The loss simplifies to L(1, p) = -log(p).")
print("       - If model predicts a high probability p for Class 1 (e.g., p ≈ 1), then log(p) ≈ 0, so Loss is low (good).")
print("       - If model predicts a low probability p for Class 1 (e.g., p ≈ 0), then log(p) is a large negative number, so Loss is high (poor).")

print("     - If the true label y = 0:")
print("       The loss simplifies to L(0, p) = -log(1-p).")
print("       (1-p is the predicted probability for Class 0)")
print("       - If model predicts a low probability p for Class 1 (e.g., p ≈ 0, so 1-p ≈ 1), then log(1-p) ≈ 0, so Loss is low (good).")
print("       - If model predicts a high probability p for Class 1 (e.g., p ≈ 1, so 1-p ≈ 0), then log(1-p) is a large negative number, so Loss is high (poor).")

print("\n   - Overall BCE behavior summary:")
print("     - True y=1, Predicted p is high (close to 1) -> Loss is low (Good model for this sample)." )
print("     - True y=1, Predicted p is low (close to 0)  -> Loss is high (Poor model for this sample)." )
print("     - True y=0, Predicted p is high (close to 1) -> Loss is high (Poor model for this sample)." )
print("     - True y=0, Predicted p is low (close to 0)  -> Loss is low (Good model for this sample)." )
print("   The total loss over the dataset is typically the average of these individual losses.")


# --- 3.1 Visualizing the Sigmoid Function ---
print("\n\n--- 3.1 Visualizing the Sigmoid Function ---")
print("The Sigmoid function is key to logistic regression, as it maps any real-valued number (z-score)")
print("into a probability between 0 and 1.")

# Define the sigmoid function for plotting
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Generate z values for plotting (ensure range covers example z-scores)
z_min = min(-8, z_score_ex1, z_score_ex2, z_score_ex3) -1 # Ensure a bit of padding
z_max = max(8, z_score_ex1, z_score_ex2, z_score_ex3) +1
z_values_plot = np.linspace(z_min, z_max, 200) 
sigmoid_values_plot = sigmoid(z_values_plot)

plt.figure(figsize=(10, 7)) # Slightly larger figure
plt.plot(z_values_plot, sigmoid_values_plot, label="σ(z) = 1 / (1 + e^(-z))", color="green")

# Plotting the dental examples on the sigmoid curve
plt.scatter([z_score_ex1], [prob_ex1], color='blue', s=100, zorder=5, label=f'Ex1 (High P: z={z_score_ex1:.2f}, P={prob_ex1:.2f})')
plt.scatter([z_score_ex2], [prob_ex2], color='red', s=100, zorder=5, label=f'Ex2 (Low P: z={z_score_ex2:.2f}, P={prob_ex2:.2f})')
plt.scatter([z_score_ex3], [prob_ex3], color='purple', s=100, zorder=5, label=f'Ex3 (Mid P: z={z_score_ex3:.2f}, P={prob_ex3:.2f})')

plt.title("The Sigmoid (Logistic) Function with Dental Examples")
plt.xlabel("z (log-odds, linear combination w·x + b)")
plt.ylabel("σ(z) (Probability)")
plt.grid(True, linestyle='--')
plt.axhline(y=0.5, color='red', linestyle='--', label="Threshold = 0.5")
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.axhline(y=1, color='black', linestyle='-', linewidth=0.5)
plt.yticks(np.arange(0, 1.1, 0.1)) # Y-axis ticks from 0 to 1 in 0.1 increments
plt.legend()

# Save the plot
sigmoid_plot_save_path = "plots/chapter_3/logistic_regression/sigmoid_function_plot.png"
plt.savefig(sigmoid_plot_save_path, dpi=300, bbox_inches='tight')
print(f"   Sigmoid function plot saved to: {sigmoid_plot_save_path}")
print("   Displaying Sigmoid function plot...")
plt.show()

# --- 3.2 Visualizing Binary Cross-Entropy (BCE) Loss ---
print("\n\n--- 3.2 Visualizing Binary Cross-Entropy (BCE) Loss ---")
print("BCE Loss quantifies the error for binary classification tasks.")

# Probabilities for plotting loss (avoiding log(0))
p_plot = np.linspace(0.001, 0.999, 200)

# Case 1: True label y = 1. Loss = -log(p)
loss_y_is_1 = -np.log(p_plot)
plt.figure(figsize=(8, 6))
plt.plot(p_plot, loss_y_is_1, color='dodgerblue', label="Loss when True Label y = 1 (Loss = -log(p))")
plt.title("Binary Cross-Entropy Loss (when True Label y = 1)")
plt.xlabel("Predicted Probability p for Class 1, σ(z)")
plt.ylabel("Loss = -log(p)")
plt.grid(True, linestyle='--')
plt.legend()
bce1_plot_save_path = "plots/chapter_3/logistic_regression/bce_loss_y_equals_1.png"
plt.savefig(bce1_plot_save_path, dpi=300, bbox_inches='tight')
print(f"   BCE Loss (y=1) plot saved to: {bce1_plot_save_path}")
print("   Displaying BCE Loss (y=1) plot...")
plt.show()

# Case 2: True label y = 0. Loss = -log(1-p)
loss_y_is_0 = -np.log(1 - p_plot)
plt.figure(figsize=(8, 6))
plt.plot(p_plot, loss_y_is_0, color='orangered', label="Loss when True Label y = 0 (Loss = -log(1-p))")
plt.title("Binary Cross-Entropy Loss (when True Label y = 0)")
plt.xlabel("Predicted Probability p for Class 1, σ(z)")
plt.ylabel("Loss = -log(1-p)")
plt.grid(True, linestyle='--')
plt.legend()
bce0_plot_save_path = "plots/chapter_3/logistic_regression/bce_loss_y_equals_0.png"
plt.savefig(bce0_plot_save_path, dpi=300, bbox_inches='tight')
print(f"   BCE Loss (y=0) plot saved to: {bce0_plot_save_path}")
print("   Displaying BCE Loss (y=0) plot...")
plt.show()

print("\n--- End of Logistic Regression Introduction (Sections 1-3 & Visualizations) ---")
print("   Further sections (Training, Prediction details, Considerations, etc.) will be added later.")
print("-" * 70) 