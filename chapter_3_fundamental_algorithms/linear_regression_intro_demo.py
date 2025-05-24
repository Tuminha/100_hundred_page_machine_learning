"""
Introduction to Linear Regression - Concepts and Notation.

Based on Chapter 3: Fundamental Algorithms (Linear Regression)
from 'The Hundred-Page Machine Learning Book' by Andriy Burkov,
and personal study notes.

This script covers the initial concepts of Linear Regression:
- What is Linear Regression?
- Data and Notation (Dataset, Feature Vectors)
- The Model: Linear Regression Equation and Objective Function (MSE)

This script is a work in progress and reflects the initial sections of
the learning material.
"""

import numpy as np
import matplotlib.pyplot as plt

print("--- Chapter 3: Linear Regression - Introduction & Concepts ---")

# --- 1. What is Linear Regression? 🎯 ---
print("\n--- 1. What is Linear Regression? ---")
print("   - General Idea: Linear Regression predicts a continuous numerical value (Y) from input features (X).")
print("     The goal is to model Y as a linear function of X: Y <- ƒ_w,b(X).")
print("   - Core Concept: It finds the best 'straight-line' (for 1 feature) or 'flat-plane/hyperplane' (for multiple features)")
print("     relationship that minimizes the distances between the line/plane and the actual data points.")
print("   - Robustness: Linear regression rarely overfits, meaning it often generalizes well to unseen data,")
print("     making it a robust choice for many applications.")
print("   - Note on Polynomial Regression: While linear regression focuses on straight lines/planes, polynomial regression")
print("     can model more complex, curved relationships but has a higher risk of overfitting.")

print("\n   --- Dental Application Teaser ---")
print("   - Example: Predicting the success rate of dental implants (a continuous numerical score).")
print("   - Input Features (Xi): A vector of D dimensions, e.g., D=3 for (Implant Surface Sa value, ISQ, BIC).")
print("     - Xi = (Surface Sa_i, BIC_i, ISQ_i)")
print("   - Output (Yi): The numerical value we want to predict (e.g., implant success score).")
print("   - Dataset representation: A set of pairs {(X_i, Y_i)}, where X_i is the feature vector and Y_i is the target value.")
print("     - This can be written as {(X_i,1, X_i,2, X_i,3), Y_i} for D=3.")
print("     - X_i,1 = Surface Sa_i")
print("     - X_i,2 = BIC_i")
print("     - X_i,3 = ISQ_i")

# --- 2. The Building Blocks: Data and Notation 🧱 ---
print("\n--- 2. The Building Blocks: Data and Notation ---")
print("   - Dataset Notation: Typically, the dataset is represented as:")
print("     - X: An N x D matrix, where N is the number of samples and D is the number of features.")
print("       Each row is a feature vector X_i.")
print("     - Y: An N x 1 vector (or an array of N elements), where each element Y_i is the target value for X_i.")

print("\n   --- Example of X matrix (N=2 samples, D=3 features) ---")
# Features: [Surface Sa, BIC, ISQ]
X_example = np.array([
    [1.75, 45, 68],  # Sample 1
    [1.64, 39, 67]   # Sample 2
])
print("   X = ")
print(X_example)
print("     - Row 1 (X_1): Surface Sa=1.75, BIC=45, ISQ=68")
print("     - Row 2 (X_2): Surface Sa=1.64, BIC=39, ISQ=67")

print("\n   --- Example of Y vector (N=2 samples) ---")
# Target: Implant Success Score (e.g., on a scale of 0-100)
Y_example = np.array([
    [75],  # Target for Sample 1
    [72]   # Target for Sample 2
])
print("   Y = ")
print(Y_example)
print("     - Y_1 = 75 (for X_1)")
print("     - Y_2 = 72 (for X_2)")

print("\n   --- D-dimensional Feature Vector Examples ---")
print("   - D=1: {(Torque_i, Y_i)}")
print("     - X_i = (Torque_i)")
print("   - D=2: {(Torque_i, BIC_i), Y_i}")
print("     - X_i = (Torque_i, BIC_i)")
print("   - D=3: {(Torque_i, BIC_i, ISQ_i), Y_i}")
print("     - X_i = (Torque_i, BIC_i, ISQ_i)")

# --- 2.1. Visualizing a Simple Linear Relationship (1D Example) ---
print("\n--- 2.1. Visualizing a Simple Linear Relationship (1D Example) ---")
print("   Linear regression aims to find a line that best fits the data points.")
print("   Let's visualize some synthetic 1D data (e.g., Study Hours vs. Exam Score).")

# Generate synthetic data for plotting
np.random.seed(42) # for reproducibility
X_plot_study_hours = np.linspace(1, 10, 30)  # 30 data points for study hours from 1 to 10
# Assume a base score, a positive impact of study hours, and some random noise
Y_plot_exam_scores = 20 + 7 * X_plot_study_hours + np.random.normal(0, 8, 30)
Y_plot_exam_scores = np.clip(Y_plot_exam_scores, 0, 100) # Clip scores to be between 0 and 100

# Create the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(X_plot_study_hours, Y_plot_exam_scores, color='blue', label='Actual Exam Scores')

# Add a hypothetical "best-fit" line (not calculated, just for illustration)
# We'll pick two points to define a plausible line for this data.
# Let's say it passes near (1, 25) and (10, 90)
m_hypothetical = (90 - 25) / (10 - 1)
b_hypothetical = 25 - m_hypothetical * 1
Y_hypothetical_line = m_hypothetical * X_plot_study_hours + b_hypothetical
plt.plot(X_plot_study_hours, Y_hypothetical_line, color='red', linestyle='--', label='Hypothetical Best-Fit Line (Illustrative)')

plt.title('Illustrative Example: Study Hours vs. Exam Score')
plt.xlabel('Study Hours (X)')
plt.ylabel('Exam Score (Y)')
plt.legend()
plt.grid(True, linestyle='--')
print("   Displaying scatter plot of synthetic data (Study Hours vs. Exam Score)...")
print("   The blue dots are individual data points (students).")
print("   The red dashed line is a *hypothetical* line that linear regression would try to find")
print("   to best represent the relationship between study hours and exam scores.")
plt.show()

# --- 3. The Model: The Linear Regression Equation ⚙️ ---
print("\n--- 3. The Model: The Linear Regression Equation ---")
print("   - Formula (vector notation for one sample X_i):")
print("     ƒ_w,b(X_i) = w · X_i + b")
print("     where:")
print("       - ƒ_w,b(X_i) is the predicted value (Y_hat_i).")
print("       - w is the vector of weights (D x 1), one weight for each feature.")
print("       - X_i is the feature vector of a single sample (D x 1 or 1 x D, depends on convention; here D elements).")
print("       - '·' denotes the dot product (sum of element-wise products).")
print("       - b is the bias term (a scalar), also known as the intercept.")
print("   - The weights (w) and bias (b) are learned from the training data.")

print("\n   --- Objective Function (Loss Function): Mean Squared Error (MSE) ---")
print("   - Purpose: To measure how well the model's predictions match the actual values.")
print("     The goal of training is to find w and b that MINIMIZE this loss function.")
print("   - Formula:")
print("     MSE = (1/N) * Σ_i (ƒ_w,b(X_i) - Y_i)^2  (summation from i=1 to N)")
print("     Alternatively: MSE = (1/N) * Σ_i (Y_hat_i - Y_i)^2")
print("   - Why squared difference?")
print("     - Emphasizes larger errors more significantly (e.g., an error of 4 (2^2) is much larger than an error of 2).")
print("     - Results in a convex loss function, which is easier to optimize (has a single global minimum).")
print("     - Mathematically convenient for derivation (e.g., when using gradient descent).")
print("   - How it's used in training:")
print("     - The model iterates over the dataset (or batches of it) multiple times (epochs).")
print("     - In each iteration, it calculates predictions, computes the MSE loss.")
print("     - It then adjusts w and b in a direction that reduces the MSE (e.g., using Gradient Descent).")
print("     - This continues until the loss converges to a minimum, or a set number of iterations is reached.")
print("     - The final w and b are the learned parameters of the model.")

print("\n   --- Breakdown of the Objective Function (MSE) ---")
print("   - (ƒ_w,b(X_i) - Y_i): The error (or residual) for a single prediction.")
print("   - (ƒ_w,b(X_i) - Y_i)^2: The squared error for a single prediction.")
print("   - Σ_i (...): The sum of squared errors over all N samples in the dataset.")
print("   - (1/N) * Σ_i (...): The average of the squared errors (Mean Squared Error).")

print("\n   --- Full Expanded Form with Dental Example ---")
print("   - Let Y = 'Predicted Implant Score' (continuous, e.g., on a scale of 0-100).")
print("   - Features for X_i: X_i1 = HU (Hounsfield Units), X_i2 = Torque, X_i3 = BIC.")
print("   - The linear regression equation for one implant (sample i):")
print("     Predicted_Implant_Score_i = (w_Hu * X_i,Hu) + (w_Torque * X_i,Torque) + (w_BIC * X_i,BIC) + b")

print("\n   --- Hypothetical Weights and Bias (as if 'learned' from training) ---")
w_Hu_example = 0.05 # Adjusted for more realistic contribution given typical HU values
w_Torque_example = 0.8 # Adjusted
w_BIC_example = 0.5    # Adjusted
b_example = 10       # Adjusted bias
print(f"     - w_Hu = {w_Hu_example} (weight for Hounsfield Units)")
print(f"     - w_Torque = {w_Torque_example} (weight for Torque)")
print(f"     - w_BIC = {w_BIC_example} (weight for Bone-Implant Contact)")
print(f"     - b = {b_example} (bias term)")

print("\n   --- Mock Dental Data and Predictions ---")
# Mock data: [HU, Torque, BIC]
mock_dental_data = np.array([
    [650, 34, 40],  # Sample 1
    [450, 25, 30],  # Sample 2
    [800, 45, 55],  # Sample 3
    [500, 30, 35]   # Sample 4
])
# Corresponding mock "actual" implant scores (for conceptual comparison later)
mock_actual_scores = np.array([78, 55, 95, 68])

print("   Let's apply these hypothetical weights to some mock patient data:")
for i in range(mock_dental_data.shape[0]):
    hu_val = mock_dental_data[i, 0]
    torque_val = mock_dental_data[i, 1]
    bic_val = mock_dental_data[i, 2]

    # Calculate individual contributions
    contribution_hu = w_Hu_example * hu_val
    contribution_torque = w_Torque_example * torque_val
    contribution_bic = w_BIC_example * bic_val

    # Calculate predicted score
    predicted_score = contribution_hu + contribution_torque + contribution_bic + b_example

    print(f"\n   - Patient Sample {i+1}: HU={hu_val}, Torque={torque_val} Nm, BIC={bic_val}%")
    print(f"     - Contribution from HU (w_Hu * HU):       {w_Hu_example:.2f} * {hu_val} = {contribution_hu:.2f}")
    print(f"     - Contribution from Torque (w_Torque * Torque): {w_Torque_example:.2f} * {torque_val} = {contribution_torque:.2f}")
    print(f"     - Contribution from BIC (w_BIC * BIC):       {w_BIC_example:.2f} * {bic_val} = {contribution_bic:.2f}")
    print(f"     - Bias term (b):                                {b_example:.2f}")
    print(f"     - Predicted Implant Score:                    {contribution_hu:.2f} + {contribution_torque:.2f} + {contribution_bic:.2f} + {b_example:.2f} = {predicted_score:.2f}")
    print(f"     (Conceptual: If actual score was {mock_actual_scores[i]}, the error for this sample would be {predicted_score - mock_actual_scores[i]:.2f})")

print("\n   --- Visualizing Mock Dental Data & Predictions ---")

# Calculate all predicted scores for plotting
all_predicted_scores = []
for i in range(mock_dental_data.shape[0]):
    hu_val = mock_dental_data[i, 0]
    torque_val = mock_dental_data[i, 1]
    bic_val = mock_dental_data[i, 2]
    predicted_score = (w_Hu_example * hu_val) + \
                      (w_Torque_example * torque_val) + \
                      (w_BIC_example * bic_val) + \
                      b_example
    all_predicted_scores.append(predicted_score)
all_predicted_scores = np.array(all_predicted_scores)

# 1. Individual Feature vs. Mock Actual Score Plots
feature_names = ['HU (Hounsfield Units)', 'Torque (Nm)', 'BIC (%)']

for i in range(mock_dental_data.shape[1]):
    plt.figure(figsize=(8, 6))
    plt.scatter(mock_dental_data[:, i], mock_actual_scores, color='teal', label='Mock Actual Scores')
    
    # Add a simple hypothetical trend line (not a calculated regression line)
    # This is purely illustrative of what linear regression might find.
    # It takes the min/max of the feature and tries to draw a line through the cloud.
    x_vals_feature = mock_dental_data[:, i]
    y_vals_actual = mock_actual_scores
    
    # Simple trend: connect line between (min_x, corresponding_y_or_min_y) and (max_x, corresponding_y_or_max_y)
    # This is a very rough heuristic for illustration only.
    if len(x_vals_feature) > 1:
        min_x_idx = np.argmin(x_vals_feature)
        max_x_idx = np.argmax(x_vals_feature)
        
        # Ensure min_x and max_x are different to avoid division by zero if all x are same
        if x_vals_feature[min_x_idx] != x_vals_feature[max_x_idx]:
            # For simplicity, let's just use the y-values at the min and max x points if they seem reasonable,
            # or estimate based on overall trend if points are noisy.
            # A robust way to draw an illustrative line without actual regression is tricky.
            # Here, we'll just use the y-values corresponding to min/max x for simplicity.
            y1 = y_vals_actual[min_x_idx]
            y2 = y_vals_actual[max_x_idx]
            m = (y2 - y1) / (x_vals_feature[max_x_idx] - x_vals_feature[min_x_idx])
            b_line = y1 - m * x_vals_feature[min_x_idx]
            trend_line_y = m * x_vals_feature + b_line
            plt.plot(x_vals_feature, trend_line_y, color='orange', linestyle='--', label='Hypothetical Trend Line')
        else: # all x values are the same, just plot points
            pass 

    plt.title(f'{feature_names[i]} vs. Mock Actual Implant Score')
    plt.xlabel(f'{feature_names[i]}')
    plt.ylabel('Mock Actual Implant Score')
    plt.legend()
    plt.grid(True, linestyle='--')
    print(f"   Displaying plot: {feature_names[i]} vs. Mock Actual Implant Score...")
    plt.show()

# 2. Predicted Scores vs. Mock Actual Scores Plot
plt.figure(figsize=(8, 6))
plt.scatter(mock_actual_scores, all_predicted_scores, color='purple', label='Predicted vs. Actual')
plt.plot([min(mock_actual_scores.min(), all_predicted_scores.min()), max(mock_actual_scores.max(), all_predicted_scores.max())], 
         [min(mock_actual_scores.min(), all_predicted_scores.min()), max(mock_actual_scores.max(), all_predicted_scores.max())], 
         color='red', linestyle='--', label='Perfect Prediction Line (y=x)')
plt.title('Predicted Implant Scores vs. Mock Actual Implant Scores')
plt.xlabel('Mock Actual Implant Scores')
plt.ylabel('Predicted Implant Scores (Hypothetical Weights)')
plt.legend()
plt.grid(True, linestyle='--')
print("   Displaying plot: Predicted Scores vs. Mock Actual Scores...")
plt.show()

# 3. Feature Contribution Bar Chart (for the first mock patient)
patient_idx_for_bar = 0
patient_data = mock_dental_data[patient_idx_for_bar]
hu_val = patient_data[0]
torque_val = patient_data[1]
bic_val = patient_data[2]

contributions = {
    'HU Contribution': w_Hu_example * hu_val,
    'Torque Contribution': w_Torque_example * torque_val,
    'BIC Contribution': w_BIC_example * bic_val,
    'Bias (b)': b_example
}

final_prediction = sum(contributions.values())

labels = list(contributions.keys())
values = list(contributions.values())

plt.figure(figsize=(10, 7))
colors = ['skyblue', 'lightgreen', 'salmon', 'gold']
bars = plt.bar(labels, values, color=colors)

# Add values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom' if yval >=0 else 'top') # Adjust va for negative values if they were possible

plt.ylabel('Contribution to Predicted Score')
plt.title(f'Breakdown of Predicted Implant Score for Patient {patient_idx_for_bar + 1}\n(HU={hu_val}, Torque={torque_val}, BIC={bic_val}) -> Predicted: {final_prediction:.2f}')
plt.xticks(rotation=15, ha="right")
plt.tight_layout() # Adjust layout to make room for rotated x-axis labels
print(f"   Displaying plot: Feature Contribution Bar Chart for Patient {patient_idx_for_bar + 1}...")
plt.show()

print("\n   --- Conceptual 'Model Accuracy' ---")
print("   If our model's predicted scores (calculated above) were consistently close to the *actual* implant scores")
print("   (like our 'mock_actual_scores'), we'd say our hypothetical weights and bias form a 'good' or 'accurate' model.")
print("   The goal of training is to find the weights and bias that minimize these differences (errors) across all training data.")

print("\n   --- CRITICAL: Interpreting Weights & The Need for Feature Scaling ---")
print("   Looking at our hypothetical weights: w_Hu={w_Hu_example}, w_Torque={w_Torque_example}, w_BIC={w_BIC_example}.")
print("   It might be tempting to say 'Torque (w={w_Torque_example}) is the most important feature because its weight is the largest.'")
print("   However, this direct comparison is MISLEADING if features are on different scales!")

print("\n   Illustrative Example of Scaling Issue for Weight Interpretation:")
print("   Imagine two features contributing to an outcome:")
print("     - Feature A: 'Normalized Lab Value', typically ranges from 0 to 1.")
print("     - Feature B: 'Raw Measurement Count', typically ranges from 0 to 1000.")

weight_A = 0.8  # Seems large
weight_B = 0.01 # Seems small

value_A = 0.5   # A typical value for Feature A
value_B = 600   # A typical value for Feature B (on its original scale)

contribution_A = weight_A * value_A
contribution_B = weight_B * value_B

print(f"   Suppose after 'training', we get weights: w_A = {weight_A}, w_B = {weight_B}")
print(f"   Now, let's see their contribution for typical values: X_A = {value_A}, X_B = {value_B}")
print(f"     - Contribution of Feature A = {weight_A} * {value_A} = {contribution_A}")
print(f"     - Contribution of Feature B = {weight_B} * {value_B} = {contribution_B}")
print("   In this case, Feature B (Raw Measurement Count) has a LARGER impact on the prediction ({contribution_B}) ")
print("   than Feature A ({contribution_A}), DESPITE Feature B having a much smaller weight.")
print("   This is because Feature B's values are numerically much larger.")
print("   CONCLUSION: To fairly compare weights to determine feature importance, features MUST be scaled to a similar range")
print("   (e.g., using Standardization or Normalization) BEFORE training the model. Once scaled, larger absolute weights generally indicate greater importance.")


print("\n--- End of Linear Regression Introduction (Sections 1-3) ---")
print("   Further sections (Training, Prediction, Considerations, etc.) will be added later.")
print("-" * 70) 