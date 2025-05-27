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
print("     Suppose after training, we found: w_torque = 0.4, w_ISQ = 1.2, and b = -2.")
print("     New implant data: Torque = 35 Nm, ISQ = 75.")

# Given values for the example
w_torque_example = 0.4
w_ISQ_example = 1.2
b_example_logistic = -2

Torque_new_logistic = 35
ISQ_new_logistic = 75

# 1. Calculate the z-score (linear combination)
z_score_example = (Torque_new_logistic * w_torque_example) + (ISQ_new_logistic * w_ISQ_example) + b_example_logistic
print(f"     1. Calculate z-score: z = (Torque * w_torque) + (ISQ * w_ISQ) + b")
print(f"        z = ({Torque_new_logistic} * {w_torque_example}) + ({ISQ_new_logistic} * {w_ISQ_example}) + ({b_example_logistic})")
print(f"        z = {Torque_new_logistic * w_torque_example} + {ISQ_new_logistic * w_ISQ_example} + ({b_example_logistic})")
print(f"        z = {z_score_example}")

# 2. Apply the Sigmoid function
# σ(z) = 1 / (1 + e^(-z))
probability_success = 1 / (1 + np.exp(-z_score_example))
print(f"\n     2. Apply Sigmoid function: σ(z) = 1 / (1 + e^(-z))")
print(f"        σ({z_score_example}) = 1 / (1 + e^(-{z_score_example}))")
print(f"        Since e^(-102) is extremely close to 0 (approx {np.exp(-z_score_example):.10e})...") # Show the small value
print(f"        σ({z_score_example}) ≈ 1 / (1 + 0)")
print(f"        σ({z_score_example}) ≈ {probability_success:.4f}")

print("\n     3. Interpretation:")
print(f"        The predicted probability of success is approximately {probability_success:.4f} (or {probability_success*100:.2f}%)." )
if probability_success > 0.5:
    print("        Since this probability > 0.5, the model predicts 'Success' (Class 1) for this implant.")
else:
    print("        Since this probability <= 0.5, the model predicts 'Failure' (Class 0) for this implant.")


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


print("\n--- End of Logistic Regression Introduction (Sections 1-3) ---")
print("   Further sections (Training, Prediction details, Considerations, etc.) will be added later.")
print("-" * 70) 