# Chapter 3: Fundamental Algorithms - Linear Regression

This section of the repository explores fundamental machine learning algorithms, starting with Linear Regression, based on Chapter 3 of "The Hundred-Page Machine Learning Book" by Andriy Burkov and supplemented with practical examples and visualizations.

## 1. Introduction to Linear Regression (`linear_regression_intro_demo.py`)

This script serves as an initial exploration of Linear Regression, focusing on core concepts, notation, the model equation, and the Mean Squared Error (MSE) objective function. It heavily utilizes print statements for explanations and `matplotlib` for visualizations.

### Key Concepts Covered:

*   **What is Linear Regression?**
    *   Predicts a continuous numerical value (Y) from input features (X).
    *   Models Y as a linear function: `Y <- ƒ_w,b(X)`.
    *   Aims to find the best "straight-line" or "flat-plane/hyperplane" that minimizes distances to data points.
    *   Generally robust against overfitting.
    *   Includes a dental application teaser: predicting implant success scores from features like Surface Sa, ISQ, and BIC.

*   **Data and Notation:**
    *   **Dataset:** Represented by an N x D matrix `X` (N samples, D features) and an N x 1 vector `Y` (target values).
    *   **Feature Vector `X_i`**: A single row in `X`, representing one sample's features.
    *   Examples are provided for `X` and `Y` matrices using mock dental data.

*   **The Model: Linear Regression Equation:**
    *   Formula for one sample `X_i`: `ƒ_w,b(X_i) = w · X_i + b`
        *   `w`: Vector of weights (one per feature).
        *   `b`: Bias term (scalar intercept).
        *   `w` and `b` are learned during training.
    *   **Objective Function (Loss Function): Mean SquaredError (MSE)**
        *   Formula: `MSE = (1/N) * Σ_i (ƒ_w,b(X_i) - Y_i)^2`
        *   Measures the average squared difference between predicted and actual values.
        *   Training aims to find `w` and `b` that minimize MSE.
        *   Reasons for using squared difference (emphasizes large errors, convex, mathematically convenient) are discussed.

*   **Detailed Dental Example (Predicting Implant Score):**
    *   Features: Hounsfield Units (HU), Torque, Bone-Implant Contact (BIC).
    *   Hypothetical weights and bias are used to demonstrate prediction.
    *   **Mock Dental Data:** Four patient samples with HU, Torque, BIC values, and corresponding mock "actual" implant scores.
    *   **Prediction Calculation:** For each mock patient, the script calculates:
        *   The individual contribution of each weighted feature (e.g., `w_Hu * HU_value`).
        *   The final predicted implant score.
    *   **Weight Interpretation & Feature Scaling:**
        *   A critical discussion on why direct comparison of raw weight magnitudes can be misleading if features are not on similar scales.
        *   An illustrative example with "Feature A" (small scale) and "Feature B" (large scale) demonstrates how a feature with a smaller weight can have a larger impact if its numerical values are much larger. This highlights the necessity of feature scaling (e.g., Standardization) before training if one wishes to interpret weights as indicators of feature importance.

### Visualizations in `linear_regression_intro_demo.py`:

The script includes several plots to aid understanding:

1.  **Illustrative Example: Study Hours vs. Exam Score:**
    *   **Purpose:** To provide a simple, intuitive 1D visualization of what linear regression tries to achieve.
    *   **Content:** A scatter plot of synthetic data (Study Hours vs. Exam Score) showing a positive linear trend. A hypothetical "best-fit" line is overlaid to illustrate the target of the regression.
    *   *(Image Reference: ../../plots/chapter_3/linear_regression/study_hours_vs_exam_score.png)*

2.  **Individual Dental Features vs. Mock Actual Implant Score (3 Plots):**
    *   **Purpose:** To visually inspect the relationship between each individual dental feature (HU, Torque, BIC) from the mock data and the mock actual implant scores.
    *   **Content:** Three separate scatter plots:
        *   HU vs. Mock Actual Score
        *   Torque vs. Mock Actual Score
        *   BIC vs. Mock Actual Score
    *   Each plot includes a hypothetical trend line to suggest the linear relationship the model might try to capture for that feature in isolation.
    *   *(Image References: ../../plots/chapter_3/linear_regression/hu_vs_actual.png, ../../plots/chapter_3/linear_regression/torque_vs_actual.png, ../../plots/chapter_3/linear_regression/bic_vs_actual.png)*

3.  **Predicted Implant Scores vs. Mock Actual Implant Scores:**
    *   **Purpose:** To visually assess how well the predictions (made using the *hypothetical* weights) align with the mock actual scores.
    *   **Content:** A scatter plot with mock actual scores on the x-axis and the predicted scores on the y-axis. A 45-degree (y=x) "Perfect Prediction Line" is included as a reference. Deviations from this line represent prediction errors.
    *   *(Image Reference: ../../plots/chapter_3/linear_regression/predicted_vs_actual.png)*

4.  **Breakdown of Predicted Implant Score (Bar Chart):**
    *   **Purpose:** To clearly show how individual feature contributions and the bias term add up to form the final predicted score for a specific patient.
    *   **Content:** A bar chart for the first mock dental patient, where each bar represents the magnitude of:
        *   HU Contribution (`w_Hu * HU_value`)
        *   Torque Contribution (`w_Torque * Torque_value`)
        *   BIC Contribution (`w_BIC * BIC_value`)
        *   Bias (`b`)
    *   The numerical value of each component is displayed on its bar.
    *   *(Image Reference: ../../plots/chapter_3/linear_regression/contribution_breakdown_patient1.png)*

This script and its explanations serve as the foundational step for understanding Linear Regression before moving on to topics like training algorithms (e.g., Gradient Descent, Normal Equation), model evaluation, and more advanced considerations.

*(You can replace the `plots/*.png` placeholders with actual paths if you decide to save the plots generated by the script, or remove these lines if you prefer not to reference specific image files in the README.)* 