"""
Demonstrates concepts related to Random Variables, based on Chapter 2 concepts of
'The Hundred-Page Machine Learning Book'.

Covers:
- Random Variable (Discrete and Continuous)
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

print("--- Introduction to Random Variables ---")
print("A Random Variable (RV) maps outcomes of random phenomena to numerical values.")
print("1. Discrete RV: Takes a finite or countable number of distinct values (e.g., die roll).")
print("2. Continuous RV: Takes any value within a given range (e.g., height of a person).")
print("-"*70)


print("\n--- Deeper Dive: Discrete Random Variables ---")
print("Discrete RVs typically arise from COUNTING processes.")
print("They have distinct, separate values with 'gaps' in between.")
print("Example: Number of patient no-shows in a dental clinic in a week.")
print("Possible values could be 0, 1, 2, 3, ... (you can't have 1.5 no-shows).")

# Visualizing Discrete Random Variable example
# Let X be the number of no-shows. Possible values for our example: 0, 1, 2, 3, 4, 5
discrete_values = np.array([0, 1, 2, 3, 4, 5])
# For visualization, we just need the values themselves, not probabilities yet (that's PMF)

plt.figure(figsize=(8, 3))
# Plotting points on a number line. Using y=0 for all, and 'o' for markers.
plt.plot(discrete_values, np.zeros_like(discrete_values), 'o', ms=10, color='blue', label='Possible Values')
# Adding stems to make it a lollipop plot
plt.vlines(discrete_values, 0, 0.5, colors='blue', lw=2) # Arbitrary y-limit for stem visibility
plt.title('Visualizing Discrete Random Variable: Number of No-Shows')
plt.xlabel('Value of Random Variable X (Number of No-Shows)')
plt.yticks([]) # Hide y-axis as it's not meaningful here
plt.xticks(discrete_values)
plt.xlim(-0.5, 5.5)
plt.ylim(-0.1, 1) # Adjust y-limits to make stems visible and give some space
plt.legend()
plt.grid(True, axis='x', linestyle='--')
print("Displaying plot for Discrete Random Variable example...")
plt.show()
print("-"*70)


print("\n--- Deeper Dive: Continuous Random Variables ---")
print("Continuous RVs typically arise from MEASURING processes.")
print("They can take on ANY value within a specified interval (or intervals).")
print("There are no 'gaps' between possible values.")
print("Example: Implant Stability Quotient (ISQ) for a dental implant.")
print("Possible values could be 65.0, 65.1, 65.12, 65.123, ... within a range like 0 to 100.")

# Visualizing Continuous Random Variable example
# Let Y be the ISQ value. Example range [50, 85]
isq_min = 50
isq_max = 85

# For visualization, we show a continuous line segment
# and can also plot a few random points within this range

plt.figure(figsize=(8, 3))
# Plot the continuous range as a thick line
plt.hlines(0, isq_min, isq_max, colors='green', lw=10, label=f'Possible ISQ Range [{isq_min}-{isq_max}]')

# Optionally, add a few random points to illustrate density
np.random.seed(42) # for reproducibility
random_isq_values = np.random.uniform(isq_min, isq_max, 5)
plt.plot(random_isq_values, np.zeros_like(random_isq_values) + 0.01, '|', ms=20, color='darkgreen', label='Example Measured Values')

plt.title('Visualizing Continuous Random Variable: ISQ Values')
plt.xlabel('Value of Random Variable Y (ISQ Score)')
plt.yticks([]) # Hide y-axis
plt.xlim(isq_min - 5, isq_max + 5)
plt.ylim(-0.1, 0.5)
plt.legend(loc='upper center')
plt.grid(True, axis='x', linestyle='--')
print("Displaying plot for Continuous Random Variable example...")
plt.show()
print("-"*70)