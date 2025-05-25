"""
Introduction to Linear Regression - Concepts and Notation.

Based on Chapter 3: Fundamental Algorithms (Linear Regression)
from 'The Hundred-Page Machine Learning Book' by Andriy Burkov,
and personal study notes.

This script covers the initial concepts of Linear Regression:
- What is Linear Regression?
- Data and Notation (Dataset, Feature Vectors)
- The Model: Linear Regression Equation and Objective Function (MSE)
- Conceptual Prediction with "Learned" Parameters (Illustrative)

It includes various print explanations and visualizations to illustrate these concepts,
including a general 1D example and detailed dental examples.

This script is a work in progress and reflects the initial sections of
the learning material, plus a conceptual look at prediction post-training.
"""

import numpy as np

print("\n--- End of Linear Regression Introduction (Sections 1-3) ---")
print("   Further sections (Training, Prediction, Considerations, etc.) will be added later.")
print("-" * 70)

# --- 3.1 Conceptual Prediction with "Learned" Parameters (Illustrative) ---
print("\n\n--- 3.1 Conceptual Prediction with \"Learned\" Parameters (Post-Training Illustration) ---")
print("Once a model is trained, it provides learned weights (w_learned) and a bias (b_learned).")
print("These are then used to make predictions on new, unseen data.")

print("\n   --- Dental Example: Predicting Marginal Bone Loss (MBL) --- ")
print("   Assume a model was trained to predict MBL using HU, Torque, and ISQ at placement.")
print("   Hypothetical learned parameters after training:")

w_torque_learned_mbl = 0.2
w_hu_learned_mbl = 0.3
w_isq_learned_mbl = 0.4
b_learned_mbl = 12

print(f"     - w_torque_learned = {w_torque_learned_mbl}")
print(f"     - w_hu_learned = {w_hu_learned_mbl}")
print(f"     - w_isq_learned = {w_isq_learned_mbl}")
print(f"     - b_learned = {b_learned_mbl}")

print("\n   New implant scenario data:")
Torque_new_mbl = 30
HU_new_mbl = 450
ISQ_new_mbl = 80

print(f"     - Torque_new = {Torque_new_mbl} Nm")
print(f"     - HU_new = {HU_new_mbl}")
print(f"     - ISQ_new = {ISQ_new_mbl}")

# Prediction calculation
MBL_predicted = (w_torque_learned_mbl * Torque_new_mbl) + \
                (w_hu_learned_mbl * HU_new_mbl) + \
                (w_isq_learned_mbl * ISQ_new_mbl) + \
                b_learned_mbl

print("\n   Prediction calculation:")
print(f"     MBL_predicted = ({w_torque_learned_mbl} * {Torque_new_mbl}) + ({w_hu_learned_mbl} * {HU_new_mbl}) + ({w_isq_learned_mbl} * {ISQ_new_mbl}) + {b_learned_mbl}")
print(f"                   = {(w_torque_learned_mbl * Torque_new_mbl)} + {(w_hu_learned_mbl * HU_new_mbl)} + {(w_isq_learned_mbl * ISQ_new_mbl)} + {b_learned_mbl}")
print(f"                   = {MBL_predicted}")
print("   Interpretation: This predicted value of 185 might represent 1.85 mm of MBL if the output was scaled by 100.")

print("-" * 70) 