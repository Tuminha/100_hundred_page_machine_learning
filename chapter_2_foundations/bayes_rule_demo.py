"""
Demonstrates Bayes' Rule with examples, based on Chapter 2 concepts of
'The Hundred-Page Machine Learning Book'.

Covers:
- Conceptual explanation of Bayes' Rule
- Classic disease test example
- Dental example: Diagnosing Apical Periodontitis
"""

import numpy as np
import matplotlib.pyplot as plt
# Note: scipy.stats might not be needed if only direct calculations are performed.
# We'll keep it for now in case of future enhancements but it's not used by current Bayes code.

print("\n\n--- Bayes' Rule: Updating Beliefs with Evidence ---")
print("Bayes' Rule is a fundamental theorem in probability that describes how to update the probability")
print("of a hypothesis based on new evidence.")
print("Formula: P(A|B) = [P(B|A) * P(A)] / P(B)")
print("Where:")
print("  - P(A|B): Posterior probability (probability of A given B is true)")
print("  - P(B|A): Likelihood (probability of B given A is true)")
print("  - P(A):   Prior probability (initial belief in A)")
print("  - P(B):   Marginal likelihood / Evidence (total probability of B occurring)")
print("\nIntuition: Bayes' Rule allows us to systematically combine our prior knowledge with new data")
print("to arrive at a more informed, posterior belief.")
print("The denominator P(B) can be expanded as: P(B) = P(B|A)P(A) + P(B|not A)P(not A)")
print("-" * 70)

print("\n--- Bayes' Rule: Classic Example (Disease Test) ---")

# Define Prior Probabilities
P_Disease = 0.01  # Prevalence of the disease (Prior for having the disease)
P_No_Disease = 1 - P_Disease

print(f"P(Disease) = {P_Disease:.2f} (Prior probability of having the disease)")
print(f"P(No Disease) = {P_No_Disease:.2f}")

# Define Likelihoods (Test Accuracy)
# P(Test Positive | Disease) - True Positive Rate (Sensitivity)
P_Pos_given_Disease = 0.95
# P(Test Positive | No Disease) - False Positive Rate
P_Pos_given_No_Disease = 0.05

print(f"P(Test Positive | Disease) = {P_Pos_given_Disease:.2f} (Likelihood: Test is positive if person has disease - True Positive Rate)")
print(f"P(Test Negative | Disease) = {1-P_Pos_given_Disease:.2f}")
print(f"P(Test Positive | No Disease) = {P_Pos_given_No_Disease:.2f} (Likelihood: Test is positive if person does NOT have disease - False Positive Rate)")
print(f"P(Test Negative | No Disease) = {1-P_Pos_given_No_Disease:.2f} (True Negative Rate)")

# Calculate P(B) = P(Test Positive) - The Marginal Likelihood / Evidence
# P(Test Positive) = P(Test Positive | Disease) * P(Disease) + P(Test Positive | No Disease) * P(No Disease)
P_Test_Positive = (P_Pos_given_Disease * P_Disease) + (P_Pos_given_No_Disease * P_No_Disease)

print(f"\nCalculating P(Test Positive) (Marginal Likelihood/Evidence):")
print(f"P(Test Positive) = P(Test Positive | Disease) * P(Disease) + P(Test Positive | No Disease) * P(No Disease)")
print(f"P(Test Positive) = ({P_Pos_given_Disease:.2f} * {P_Disease:.2f}) + ({P_Pos_given_No_Disease:.2f} * {P_No_Disease:.2f})")
print(f"P(Test Positive) = {P_Pos_given_Disease * P_Disease:.4f} + {P_Pos_given_No_Disease * P_No_Disease:.4f}")
print(f"P(Test Positive) = {P_Test_Positive:.4f}")

# Apply Bayes' Rule to find P(Disease | Test Positive) - The Posterior Probability
# P(Disease | Test Positive) = [P(Test Positive | Disease) * P(Disease)] / P(Test Positive)
P_Disease_given_Pos = (P_Pos_given_Disease * P_Disease) / P_Test_Positive

print(f"\nApplying Bayes' Rule to find P(Disease | Test Positive):")
print(f"P(Disease | Test Positive) = [P(Test Positive | Disease) * P(Disease)] / P(Test Positive)")
print(f"P(Disease | Test Positive) = [{P_Pos_given_Disease:.2f} * {P_Disease:.2f}] / {P_Test_Positive:.4f}")
print(f"P(Disease | Test Positive) = {P_Pos_given_Disease * P_Disease:.4f} / {P_Test_Positive:.4f}")
print(f"P(Disease | Test Positive) = {P_Disease_given_Pos:.4f}")

print(f"\nResult: If a person tests positive, the probability they actually have the disease is {P_Disease_given_Pos:.2%}.")
print("Notice how the posterior probability ({P_Disease_given_Pos:.2%}) is much lower than the test's true positive rate ({P_Pos_given_Disease:.0%}).")
print("This is because the disease is rare (low prior P(Disease)). This is a common counter-intuitive result!")
print("-" * 70)

print("\n--- Bayes' Rule: Dental Example (Apical Periodontitis Diagnosis) ---")
print("Scenario: A patient presents with persistent pain after root canal treatment.")
print("We want to calculate the probability they have Apical Periodontitis (AP) given this symptom.")

# Let A = Patient has Apical Periodontitis (AP)
# Let B = Patient reports persistent pain after root canal

# Define Prior Probabilities
# P(AP): Prior probability of having Apical Periodontitis after a root canal (general incidence)
P_AP = 0.15
P_No_AP = 1 - P_AP

print(f"\nP(AP) = {P_AP:.2f} (Prior probability of having Apical Periodontitis)")
print(f"P(No AP) = {P_No_AP:.2f}")

# Define Likelihoods
# P(Pain | AP): Likelihood of pain if AP is present
P_Pain_given_AP = 0.80
# P(Pain | No AP): Likelihood of pain even if AP is NOT present (e.g., other causes)
P_Pain_given_No_AP = 0.10

print(f"P(Pain | AP) = {P_Pain_given_AP:.2f} (Likelihood: Pain occurs if AP is present)")
print(f"P(Pain | No AP) = {P_Pain_given_No_AP:.2f} (Likelihood: Pain occurs even if AP is not present)")

# Calculate P(B) = P(Pain) - The Marginal Likelihood / Evidence
# P(Pain) = P(Pain | AP) * P(AP) + P(Pain | No AP) * P(No AP)
P_Pain = (P_Pain_given_AP * P_AP) + (P_Pain_given_No_AP * P_No_AP)

print(f"\nCalculating P(Pain) (Marginal Likelihood/Evidence for pain):")
print(f"P(Pain) = P(Pain | AP) * P(AP) + P(Pain | No AP) * P(No AP)")
print(f"P(Pain) = ({P_Pain_given_AP:.2f} * {P_AP:.2f}) + ({P_Pain_given_No_AP:.2f} * {P_No_AP:.2f})")
print(f"P(Pain) = {P_Pain_given_AP * P_AP:.4f} + {P_Pain_given_No_AP * P_No_AP:.4f}")
print(f"P(Pain) = {P_Pain:.4f}")

# Apply Bayes' Rule to find P(AP | Pain) - The Posterior Probability
# P(AP | Pain) = [P(Pain | AP) * P(AP)] / P(Pain)
P_AP_given_Pain = (P_Pain_given_AP * P_AP) / P_Pain

print(f"\nApplying Bayes' Rule to find P(AP | Pain):")
print(f"P(AP | Pain) = [P(Pain | AP) * P(AP)] / P(Pain)")
print(f"P(AP | Pain) = [{P_Pain_given_AP:.2f} * {P_AP:.2f}] / {P_Pain:.4f}")
print(f"P(AP | Pain) = {P_Pain_given_AP * P_AP:.4f} / {P_Pain:.4f}")
print(f"P(AP | Pain) = {P_AP_given_Pain:.4f}")

print(f"\nResult: If a patient reports persistent pain after a root canal, the probability they have Apical Periodontitis is {P_AP_given_Pain:.2%}.")
print(f"The initial belief (prior) of having AP was {P_AP:.2%}. After observing pain, this belief increased to {P_AP_given_Pain:.2%}.")
print("This shows how evidence (pain) updates our probability assessment.")

# Simple visualization of Prior vs Posterior
labels = ['Prior P(AP)', 'Posterior P(AP|Pain)']
probabilities = [P_AP, P_AP_given_Pain]

fig_bayes_dental, ax_bayes_dental = plt.subplots(figsize=(6, 4))
bar_colors = ['skyblue', 'salmon']
ax_bayes_dental.bar(labels, probabilities, color=bar_colors, edgecolor='black')
ax_bayes_dental.set_ylabel('Probability')
ax_bayes_dental.set_title('Prior vs. Posterior Probability of Apical Periodontitis')
ax_bayes_dental.set_ylim(0, 1)
for i, prob in enumerate(probabilities):
    ax_bayes_dental.text(i, prob + 0.02, f"{prob:.2%}", ha='center', fontweight='bold')

print("\nDisplaying plot for Prior vs Posterior probability (Dental Example)...")
plt.tight_layout()
plt.show()
print("-" * 70) 