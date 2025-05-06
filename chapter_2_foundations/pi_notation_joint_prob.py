"""
Demonstrates the use of Pi (Π) notation, particularly for calculating
the joint probability of multiple independent events, based on
Chapter 2 concepts of 'The Hundred-Page Machine Learning Book'.

Joint Probability P(A1 ∩ A2 ∩ ... ∩ An) = Π_{i=1}^{N} P(Ai) for independent events.
Where:
- Π represents product.
- i is the index from 1 to N (number of events).
- P(Ai) is the probability of the i-th independent event.
"""

import numpy as np
import matplotlib.pyplot as plt

print("--- Pi Notation (Π) Introduction ---")
# Concept: A compact way to represent the product of a sequence of numbers.
# Π_{i=m}^{n} expression(i)
# Multiply the results of 'expression(i)' as the index 'i' goes from 'm' to 'n'.

# General Example: Product of elements of a vector
# Π_{i=1}^{4} v_i where v = [1, 2, 3, 4]
v_general = np.array([1, 2, 3, 4])
product_v_general = np.prod(v_general) # NumPy's prod directly applies Π notation
print(f"General Example: Vector v = {v_general}")
print(f"Product using np.prod (Π v_i): {product_v_general}") # Output: 24 (1*2*3*4)
print("-"*35)

print("\n--- Pi Notation in Probability: Joint Probability of Independent Events ---")
# Context: Calculating the probability of several independent events all occurring.

# Implant Dentistry Example: Joint probability of N independent implant successes.
# Assume we have several implants, and each has an individual probability of success.
# These probabilities might be estimated based on factors like ISQ, BIC, patient health etc.
# For simplicity, let's assign some hypothetical probabilities:

implant_success_probs = np.array([
    0.95, # Implant 1 (e.g., high ISQ, high BIC)
    0.90, # Implant 2
    0.98, # Implant 3 (e.g., excellent bone quality)
    0.85, # Implant 4 (e.g., some minor risk factor)
    0.92  # Implant 5
])
num_implants = len(implant_success_probs)

print(f"Individual success probabilities for {num_implants} implants: {implant_success_probs}")

# Calculate the joint probability that ALL implants succeed
# P(all succeed) = P(Implant1 succeeds) * P(Implant2 succeeds) * ... * P(ImplantN succeeds)
# P(all succeed) = Π_{i=1}^{N} P(success_i)
joint_prob_all_succeed = np.prod(implant_success_probs)

print(f"Joint probability that all {num_implants} implants succeed (Π P(success_i)): {joint_prob_all_succeed:.4f}")

# 2. Visualize how joint probability decreases with more events
# Let's assume a constant high probability for each additional implant for simplicity here
constant_prob_per_implant = 0.90 # A generally good success rate
num_events_to_plot = 10
cumulative_joint_probs = []
current_joint_prob = 1.0

for i in range(1, num_events_to_plot + 1):
    current_joint_prob *= constant_prob_per_implant
    cumulative_joint_probs.append(current_joint_prob)

num_events_axis = np.arange(1, num_events_to_plot + 1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(num_events_axis, cumulative_joint_probs, marker='o', linestyle='-', color='dodgerblue')

for i, prob in enumerate(cumulative_joint_probs):
    ax.text(num_events_axis[i], prob + 0.02, f'{prob:.3f}', ha='center')

ax.set_xlabel("Number of Independent Implants Considered")
ax.set_ylabel(f"Joint Probability of ALL Succeeding (assuming P(success)={constant_prob_per_implant} for each)")
ax.set_title("Π Notation: Joint Probability Decreases with More Independent Events")
ax.set_xticks(num_events_axis)
ax.set_ylim(0, 1.1)
ax.grid(True, linestyle='--', alpha=0.6)

print("\nDisplaying plot showing decrease in joint probability...")
plt.show()

print("\nKey takeaway for joint probability with Π notation:")
print("- If individual event probabilities are < 1, the joint probability of ALL events occurring decreases as more events are added.")
print("- This is crucial in risk assessment (e.g., probability of zero failures in a batch) or system reliability.")

print("\n--- Conceptual Link to Likelihood Function (Advanced) ---")
print("The Likelihood function, L(θ | data) = Π f(data_i; θ), also uses Pi notation.")
print("It calculates the joint probability (or density) of observing the entire dataset,")
print("assuming data points are independent and come from a distribution f with parameters θ.")
print("Maximizing this likelihood (MLE) is a common way to estimate θ.")
print("A full coding example of MLE is more complex and involves specific distributions and optimization.") 