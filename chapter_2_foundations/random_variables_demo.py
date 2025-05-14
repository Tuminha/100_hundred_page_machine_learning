"""
Demonstrates concepts related to Random Variables, based on Chapter 2 concepts of
'The Hundred-Page Machine Learning Book'.

Covers:
- Random Variable (Discrete and Continuous)
- Probability Mass Function (PMF) for discrete RVs
- Probability Density Function (PDF) for continuous RVs
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


print("\n--- Probability Mass Function (PMF) for Discrete RVs ---")
print("The PMF gives the probability that a discrete random variable is EXACTLY equal to some value.")
print("P(X=x) - 'What is the probability of outcome x occurring?'")
print("Key properties of PMF:")
print("1. P(X=x) >= 0 for all x (probabilities are non-negative).")
print("2. Sum of P(X=x) over all possible x is 1 (total probability is 1).")

# Example: PMF of a fair six-sided die roll
# X = outcome of a die roll
die_outcomes = np.array([1, 2, 3, 4, 5, 6])
# For a fair die, each outcome has a probability of 1/6
die_probabilities = np.array([1/6] * 6)

print(f"\nExample: Fair Six-Sided Die")
print(f"Possible outcomes (x): {die_outcomes}")
print(f"Probabilities P(X=x): {np.round(die_probabilities, 3)}")
print(f"Sum of probabilities: {np.sum(die_probabilities):.2f}") # Should be 1.0

plt.figure(figsize=(7, 5))
plt.bar(die_outcomes, die_probabilities, color='coral', edgecolor='black', width=0.7)
plt.title('PMF of a Fair Six-Sided Die Roll')
plt.xlabel('Outcome (x)')
plt.ylabel('Probability P(X=x)')
plt.xticks(die_outcomes)
plt.ylim(0, np.max(die_probabilities) * 1.2) # Adjust y-limit based on max probability
plt.grid(axis='y', linestyle='--')
for i, prob in enumerate(die_probabilities):
    plt.text(die_outcomes[i], prob + 0.01, f'{prob:.2f}', ha='center')
print("Displaying PMF plot for a fair die roll...")
plt.show()
print("-"*70)

# --- Probability Mass Function (PMF) with Dental Example ---
print("\n--- Probability Mass Function (PMF): Dental Example ---")
print("The PMF gives the probability of each possible value of a discrete random variable.")
print("Notation: P(X = x_i) is the probability that X equals x_i. The sum over all possible x_i is 1.")
print("\nDental Example: Number of successful implants out of 3 placed.")
print("Let X be the number of successful implants. Possible values: 0, 1, 2, 3.")
print("Suppose based on data, the probabilities are:")
print("P(0) = 0.02 (none succeed)\nP(1) = 0.098 (one succeeds)\nP(2) = 0.3 (two succeed)\nP(3) = 0.582 (all succeed)")

implant_outcomes = np.array([0, 1, 2, 3])
implant_probs = np.array([0.02, 0.098, 0.3, 0.582])

print(f"Sum of probabilities: {np.sum(implant_probs):.3f} (should be 1.0)")

plt.figure(figsize=(7, 5))
bars = plt.bar(implant_outcomes, implant_probs, color='slateblue', edgecolor='black', width=0.6)
plt.title('PMF: Number of Successful Implants out of 3')
plt.xlabel('Number of Successful Implants (X)')
plt.ylabel('Probability P(X = x)')
plt.xticks(implant_outcomes)
plt.ylim(0, max(implant_probs)*1.2)
plt.grid(axis='y', linestyle='--')
for i, prob in enumerate(implant_probs):
    plt.text(implant_outcomes[i], prob + 0.01, f'{prob:.3f}', ha='center', fontsize=10)
plt.tight_layout()
print("Displaying PMF plot for dental implant example...")
plt.show()
print("-"*70)

# --- Probability Density Function (PDF) for Continuous RVs ---
print("\n--- Probability Density Function (PDF) for Continuous RVs ---")
print("The PDF describes the relative likelihood of a continuous random variable falling within a range.")
print("Notation: f(x) is the PDF. Probability for an interval [a, b] is the area under the curve: P(a ≤ X ≤ b) = ∫[a, b] f(x) dx.")
print("For any exact value, P(X = x) = 0. Only intervals have nonzero probability.")

# Dental Example: ISQ values modeled as a normal distribution
print("\nDental Example: ISQ values after implant placement are often modeled as a normal distribution.")
print("Suppose mean ISQ = 70, std = 5. What is the probability that ISQ is between 65 and 75?")

from scipy.stats import norm
isq_mean = 70
isq_std = 5
x_vals = np.linspace(50, 90, 400)
pdf_vals = norm.pdf(x_vals, loc=isq_mean, scale=isq_std)

plt.figure(figsize=(8, 5))
plt.plot(x_vals, pdf_vals, color='darkgreen', lw=2, label=r'$f(x)$ (PDF)')
plt.title('PDF: ISQ Values after Implant Placement (Normal Distribution)')
plt.xlabel('ISQ Value (x)')
plt.ylabel('Probability Density $f(x)$')
plt.grid(True, linestyle='--')

# Shade area between 65 and 75
x_fill = np.linspace(65, 75, 200)
y_fill = norm.pdf(x_fill, loc=isq_mean, scale=isq_std)
plt.fill_between(x_fill, y_fill, color='limegreen', alpha=0.5, label=r'Area = $P(65 \leq X \leq 75)$')

# Calculate probability for interval
prob_65_75 = norm.cdf(75, loc=isq_mean, scale=isq_std) - norm.cdf(65, loc=isq_mean, scale=isq_std)
plt.legend()
plt.tight_layout()
print(f"Probability ISQ is between 65 and 75: {prob_65_75:.3f} (area under the curve)")
print("Displaying PDF plot for ISQ values with shaded probability interval...")
plt.show()
print("-"*70)

# --- Special Case: PDF > 1 (Narrow Normal Distribution) ---
print("\n--- Special Case: PDF Value Greater Than 1 ---")
print("The PDF can be greater than 1 for very narrow distributions, but the area under the curve (probability) is still ≤ 1.")
print("Example: ISQ values with mean = 70, std = 0.5 (very little variation)")

narrow_std = 0.5
x_narrow = np.linspace(68, 72, 400)
pdf_narrow = norm.pdf(x_narrow, loc=isq_mean, scale=narrow_std)

plt.figure(figsize=(8, 5))
plt.plot(x_narrow, pdf_narrow, color='crimson', lw=2, label=r'Narrow PDF ($\sigma=0.5$)')
plt.title('PDF Can Exceed 1: Narrow Normal Distribution')
plt.xlabel('ISQ Value (x)')
plt.ylabel('Probability Density $f(x)$')
plt.grid(True, linestyle='--')

# Annotate the peak
peak_x = isq_mean
peak_y = norm.pdf(peak_x, loc=isq_mean, scale=narrow_std)
plt.scatter([peak_x], [peak_y], color='black', zorder=5)
plt.text(peak_x, peak_y+0.2, f'Peak f({peak_x}) = {peak_y:.2f} (>1)', ha='center', fontsize=11, color='black')

# Shade a small interval to show area
x_fill_narrow = np.linspace(69.5, 70.5, 200)
y_fill_narrow = norm.pdf(x_fill_narrow, loc=isq_mean, scale=narrow_std)
plt.fill_between(x_fill_narrow, y_fill_narrow, color='orange', alpha=0.5, label='Area = Probability in [69.5, 70.5]')

# Calculate probability for this interval
prob_narrow = norm.cdf(70.5, loc=isq_mean, scale=narrow_std) - norm.cdf(69.5, loc=isq_mean, scale=narrow_std)
plt.legend()
plt.tight_layout()
print(f"Peak PDF value: {peak_y:.2f} (greater than 1)")
print(f"Probability ISQ in [69.5, 70.5]: {prob_narrow:.3f} (area under the curve, still < 1)")
print("Displaying narrow PDF plot to illustrate PDF > 1 confusion point...")
plt.show()
print("-"*70)