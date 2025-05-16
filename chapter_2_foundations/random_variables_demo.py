"""
Demonstrates concepts related to Random Variables, based on Chapter 2 concepts of
'The Hundred-Page Machine Learning Book'.

Covers:
- Random Variable (Discrete and Continuous)
- Probability Mass Function (PMF) for discrete RVs
- Probability Density Function (PDF) for continuous RVs
- Expected Value (Mean)
- Variance (σ²)
- Standard Deviation (σ)
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

# Calculate E[X] for the fair die example
# die_outcomes and die_probabilities are already defined
expected_value_die = np.sum(die_outcomes * die_probabilities)
print(f"\nExample: Fair Six-Sided Die")
print(f"Outcomes: {die_outcomes}, Probabilities: {np.round(die_probabilities, 3)}")
print(f"E[X_die] = (1*1/6) + (2*1/6) + (3*1/6) + (4*1/6) + (5*1/6) + (6*1/6) = {expected_value_die:.2f}")
print("This means if you roll a fair die many times, the average outcome will be close to 3.5.")
print("Notice E[X] doesn't have to be a value X can actually take (you can't roll a 3.5).")

# Plotting PMF with E[X] for Fair Die
plt.figure(figsize=(7, 5))
plt.bar(die_outcomes, die_probabilities, color='coral', edgecolor='black', width=0.7, label='P(X=x)')
plt.axvline(expected_value_die, color='red', linestyle='dashed', linewidth=2, label=f'E[X] = {expected_value_die:.2f}')
plt.title('PMF of Die Roll with Expected Value (E[X])')
plt.xlabel('Outcome (x)')
plt.ylabel('Probability P(X=x)')
plt.xticks(die_outcomes)
plt.ylim(0, np.max(die_probabilities) * 1.2)
plt.grid(axis='y', linestyle='--')
for i, prob in enumerate(die_probabilities):
    plt.text(die_outcomes[i], prob + 0.01, f'{prob:.2f}', ha='center')
plt.legend()
print("Displaying PMF plot for die roll with E[X] indicated...")
plt.show()

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

# Calculate E[X] for the dental implant success example
# implant_outcomes and implant_probs are already defined
expected_value_implants = np.sum(implant_outcomes * implant_probs)
print(f"\nExample: Number of Successful Implants (out of 3)")
print(f"Outcomes: {implant_outcomes}, Probabilities: {implant_probs}")
print(f"E[X_implants] = (0*{implant_probs[0]}) + (1*{implant_probs[1]}) + (2*{implant_probs[2]}) + (3*{implant_probs[3]}) = {expected_value_implants:.3f}")
print(f"On average, about {expected_value_implants:.3f} implants are expected to be successful per set of 3.")

# Plotting PMF with E[X] for Dental Implants
plt.figure(figsize=(7, 5))
plt.bar(implant_outcomes, implant_probs, color='slateblue', edgecolor='black', width=0.6, label='P(X=x)')
plt.axvline(expected_value_implants, color='red', linestyle='dashed', linewidth=2, label=f'E[X] = {expected_value_implants:.3f}')
plt.title('PMF of Implant Success with Expected Value (E[X])')
plt.xlabel('Number of Successful Implants (X)')
plt.ylabel('Probability P(X = x)')
plt.xticks(implant_outcomes)
plt.ylim(0, max(implant_probs)*1.2)
plt.grid(axis='y', linestyle='--')
for i, prob in enumerate(implant_probs):
    plt.text(implant_outcomes[i], prob + 0.01, f'{prob:.3f}', ha='center', fontsize=10)
plt.legend()
plt.tight_layout()
print("Displaying PMF plot for implant success with E[X] indicated...")
plt.show()

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

# E[X] for the ISQ normal distribution example
# For a Normal distribution N(μ, σ^2), the expected value E[X] is simply μ.
# isq_mean is already defined as 70
print(f"\nExample: ISQ Values (Normally Distributed)")
print(f"ISQ values are modeled as N(μ={isq_mean}, σ={isq_std}).")
print(f"E[X_ISQ] = μ = {isq_mean}")
print(f"The average expected ISQ value is {isq_mean}.")

# Plotting PDF with E[X] for ISQ Values
plt.figure(figsize=(8, 5))
plt.plot(x_vals, pdf_vals, color='darkgreen', lw=2, label=r'$f(x)$ (PDF)')
plt.axvline(isq_mean, color='blue', linestyle='dashed', linewidth=2, label=f'E[X] = $\mu$ = {isq_mean}')
plt.title('PDF of ISQ Values with Expected Value (E[X])')
plt.xlabel('ISQ Value (x)')
plt.ylabel('Probability Density $f(x)$')
plt.grid(True, linestyle='--')
plt.legend()
plt.tight_layout()
print("Displaying PDF plot for ISQ values with E[X] indicated...")
plt.show()

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

# --- Expected Value (Mean) ---
print("\n--- Expected Value (Mean, Average) E[X] or μ ---")
print("The Expected Value (E[X]) is the long-run average value of a random variable.")
print("It represents the 'center of mass' or balancing point of its probability distribution.")

print("\nFor a Discrete Random Variable X:")
print("E[X] = Σ [x * P(X=x)] for all possible values x.")
print("It's the sum of each value multiplied by its probability.")

print("\nFor a Continuous Random Variable X:")
print("E[X] = ∫ x * f(x) dx, integrated over all possible values of x.")
print("It's the integral of each value x multiplied by its probability density f(x).")

print("\nKey Confusion Point for Continuous E[X]:")
print("The integral for E[X] is generally from -∞ to +∞ (or the full support of the PDF).")
print("Calculating ∫[a,b] x*f(x)dx for a sub-interval [a,b] is NOT E[X], unless f(x) is 0 outside [a,b].")
print("It gives a 'conditional expectation' contribution from that interval but isn't the overall mean.")
print("For named distributions like Normal, Exponential, etc., E[X] is often a known parameter (e.g., μ for Normal).")
print("-"*70)

# E[X] for the fair die example
# die_outcomes and die_probabilities are defined in the PMF section
expected_value_die = np.sum(die_outcomes * die_probabilities)
print(f"\nExample: Fair Six-Sided Die")
print(f"Outcomes: {die_outcomes}, Probabilities: {np.round(die_probabilities, 3)}")
print(f"E[X_die] = (1*1/6) + (2*1/6) + (3*1/6) + (4*1/6) + (5*1/6) + (6*1/6) = {expected_value_die:.2f}")
print("This means if you roll a fair die many times, the average outcome will be close to 3.5.")
print("Notice E[X] doesn't have to be a value X can actually take (you can't roll a 3.5).")

# Plotting PMF with E[X] for Fair Die (within E[X] section)
plt.figure(figsize=(7, 5))
plt.bar(die_outcomes, die_probabilities, color='coral', edgecolor='black', width=0.7, label='P(X=x)')
plt.axvline(expected_value_die, color='red', linestyle='dashed', linewidth=2, label=f'E[X] = {expected_value_die:.2f}')
plt.title('E[X] on PMF of Die Roll')
plt.xlabel('Outcome (x)')
plt.ylabel('Probability P(X=x)')
plt.xticks(die_outcomes)
plt.ylim(0, np.max(die_probabilities) * 1.2)
plt.grid(axis='y', linestyle='--')
for i, prob in enumerate(die_probabilities):
    plt.text(die_outcomes[i], prob + 0.01, f'{prob:.2f}', ha='center')
plt.legend()
print("Displaying die roll PMF with E[X] (dedicated plot)...")
plt.show()

# E[X] for the dental implant success example
# implant_outcomes and implant_probs are defined in the PMF section
expected_value_implants = np.sum(implant_outcomes * implant_probs)
print(f"\nExample: Number of Successful Implants (out of 3)")
print(f"Outcomes: {implant_outcomes}, Probabilities: {implant_probs}")
print(f"E[X_implants] = (0*{implant_probs[0]}) + (1*{implant_probs[1]}) + (2*{implant_probs[2]}) + (3*{implant_probs[3]}) = {expected_value_implants:.3f}")
print(f"On average, about {expected_value_implants:.3f} implants are expected to be successful per set of 3.")

# Plotting PMF with E[X] for Dental Implants (within E[X] section)
plt.figure(figsize=(7, 5))
plt.bar(implant_outcomes, implant_probs, color='slateblue', edgecolor='black', width=0.6, label='P(X=x)')
plt.axvline(expected_value_implants, color='red', linestyle='dashed', linewidth=2, label=f'E[X] = {expected_value_implants:.3f}')
plt.title('E[X] on PMF of Implant Success')
plt.xlabel('Number of Successful Implants (X)')
plt.ylabel('Probability P(X = x)')
plt.xticks(implant_outcomes)
plt.ylim(0, max(implant_probs)*1.2)
plt.grid(axis='y', linestyle='--')
for i, prob in enumerate(implant_probs):
    plt.text(implant_outcomes[i], prob + 0.01, f'{prob:.3f}', ha='center', fontsize=10)
plt.legend()
plt.tight_layout()
print("Displaying implant PMF with E[X] (dedicated plot)...")
plt.show()

print("\nFor a Continuous Random Variable X:")
print("E[X] = ∫ x * f(x) dx, integrated over all possible values of x.")
print("It's the integral of each value x multiplied by its probability density f(x).")

# E[X] for the ISQ normal distribution example
# For a Normal distribution N(μ, σ^2), the expected value E[X] is simply μ.
# isq_mean, isq_std, x_vals, pdf_vals are defined in the PDF section
print(f"\nExample: ISQ Values (Normally Distributed)")
print(f"ISQ values are modeled as N(μ={isq_mean}, σ={isq_std}).")
print(f"E[X_ISQ] = μ = {isq_mean}")
print(f"The average expected ISQ value is {isq_mean}.")

# Plotting PDF with E[X] for ISQ Values (within E[X] section)
plt.figure(figsize=(8, 5))
plt.plot(x_vals, pdf_vals, color='darkgreen', lw=2, label=r'$f(x)$ (PDF)')
plt.axvline(isq_mean, color='blue', linestyle='dashed', linewidth=2, label=f'E[X] = $\mu$ = {isq_mean}')
plt.title('E[X] on PDF of ISQ Values')
plt.xlabel('ISQ Value (x)')
plt.ylabel('Probability Density $f(x)$')
plt.grid(True, linestyle='--')
# Re-shade area for context if desired, or keep it simple focusing on E[X]
x_fill_e_context = np.linspace(isq_mean - isq_std, isq_mean + isq_std, 200) # Example: +/- 1 std dev
y_fill_e_context = norm.pdf(x_fill_e_context, loc=isq_mean, scale=isq_std)
plt.fill_between(x_fill_e_context, y_fill_e_context, color='lightgreen', alpha=0.3, label=r'Context: $\mu \pm \sigma$')
plt.legend()
plt.tight_layout()
print("Displaying ISQ PDF with E[X] (dedicated plot)...")
plt.show()

print("\nKey Confusion Point for Continuous E[X]:")
print("The integral for E[X] is generally from -∞ to +∞ (or the full support of the PDF).")
print("Calculating ∫[a,b] x*f(x)dx for a sub-interval [a,b] is NOT E[X], unless f(x) is 0 outside [a,b].")
print("It gives a 'conditional expectation' contribution from that interval but isn't the overall mean.")
print("For named distributions like Normal, Exponential, etc., E[X] is often a known parameter (e.g., μ for Normal).")
print("-"*70)

# --- Variance (σ²) and Standard Deviation (σ) ---
print("\n--- Variance (σ²) and Standard Deviation (σ) ---")
print("Variance and Standard Deviation measure the SPREAD or DISPERSION of a random variable's distribution.")

print("\nVariance (Var(X) or σ²):")
print("Measures the average squared difference of a random variable from its Expected Value (Mean).")
print("Definitional Formula: Var(X) = E[(X - E[X])²] = E[(X - μ)²]")
print("A larger variance means values are, on average, further spread out from the mean.")
print("Computational Formula (often easier): Var(X) = E[X²] - (E[X])²")
print("Note: Variance is in SQUARED units of the random variable.")

print("\nStandard Deviation (SD(X) or σ):")
print("Is the square root of the Variance: σ = √Var(X).")
print("Provides a measure of spread in the SAME UNITS as the random variable, making it more interpretable.")
print("A smaller SD means values tend to be close to the mean; larger SD means more spread.")

# --- Calculations for Discrete Random Variables ---
print("\n--- Variance/SD for Discrete RVs ---")

# For Fair Six-Sided Die
# E[X_die] (expected_value_die) was calculated in the E[X] section as 3.5
# die_outcomes = [1, 2, 3, 4, 5, 6], die_probabilities = [1/6]*6
print("\nExample: Fair Six-Sided Die")
# First, calculate E[X²] = Σ [x² * P(X=x)]
e_x_squared_die = np.sum((die_outcomes**2) * die_probabilities)
print(f"E[X²_die] = (1²*1/6) + ... + (6²*1/6) = {e_x_squared_die:.3f}")
var_die = e_x_squared_die - (expected_value_die**2)
sd_die = np.sqrt(var_die)
print(f"Var(X_die) = E[X²_die] - (E[X_die])² = {e_x_squared_die:.3f} - ({expected_value_die:.2f})² = {var_die:.3f}")
print(f"SD(X_die) = √Var(X_die) = {sd_die:.3f}")
print(f"Interpretation: For a fair die, the outcomes vary from the mean of {expected_value_die:.2f} by approx. {sd_die:.3f} on average.")

# Plotting PMF with E[X] and SD for Fair Die
plt.figure(figsize=(8, 5))
plt.bar(die_outcomes, die_probabilities, color='coral', edgecolor='black', width=0.7, label='P(X=x)')
plt.axvline(expected_value_die, color='red', linestyle='dashed', linewidth=2, label=f'E[X] = {expected_value_die:.2f}')
# Show E[X] ± SD(X)
plt.axvline(expected_value_die - sd_die, color='darkorange', linestyle=':', linewidth=2, label=f'E[X] ± SD (approx. {expected_value_die-sd_die:.2f} to {expected_value_die+sd_die:.2f})')
plt.axvline(expected_value_die + sd_die, color='darkorange', linestyle=':', linewidth=2)
plt.title('Die Roll PMF with E[X] and Standard Deviation (σ)')
plt.xlabel('Outcome (x)')
plt.ylabel('Probability P(X=x)')
plt.xticks(die_outcomes)
plt.legend()
plt.grid(True, axis='y', linestyle='--')
print("Displaying die roll PMF with E[X] and ±SD marked...")
plt.show()

# For Dental Implant Success
# E[X_implants] (expected_value_implants) was 2.444
# implant_outcomes = [0, 1, 2, 3], implant_probs = [0.02, 0.098, 0.3, 0.582]
print("\nExample: Number of Successful Implants (out of 3)")
# First, calculate E[X²]
e_x_squared_implants = np.sum((implant_outcomes**2) * implant_probs)
print(f"E[X²_implants] = (0²*{implant_probs[0]}) + ... + (3²*{implant_probs[3]}) = {e_x_squared_implants:.3f}")
var_implants = e_x_squared_implants - (expected_value_implants**2)
sd_implants = np.sqrt(var_implants)
print(f"Var(X_implants) = {e_x_squared_implants:.3f} - ({expected_value_implants:.3f})² = {var_implants:.3f}")
print(f"SD(X_implants) = {sd_implants:.3f}")
print(f"Interpretation: For 3 implants, successful outcomes vary from mean of {expected_value_implants:.3f} by approx. {sd_implants:.3f} on average.")

# Plotting PMF with E[X] and SD for Dental Implants
plt.figure(figsize=(8, 5))
plt.bar(implant_outcomes, implant_probs, color='slateblue', edgecolor='black', width=0.6, label='P(X=x)')
plt.axvline(expected_value_implants, color='red', linestyle='dashed', linewidth=2, label=f'E[X] = {expected_value_implants:.3f}')
# Show E[X] ± SD(X)
plt.axvline(expected_value_implants - sd_implants, color='purple', linestyle=':', linewidth=2, label=f'E[X] ± SD (approx. {expected_value_implants-sd_implants:.3f} to {expected_value_implants+sd_implants:.3f})')
plt.axvline(expected_value_implants + sd_implants, color='purple', linestyle=':', linewidth=2)
plt.title('Implant Success PMF with E[X] and Standard Deviation (σ)')
plt.xlabel('Number of Successful Implants (X)')
plt.ylabel('Probability P(X=x)')
plt.xticks(implant_outcomes)
plt.legend()
plt.grid(True, axis='y', linestyle='--')
plt.tight_layout()
print("Displaying implant PMF with E[X] and ±SD marked...")
plt.show()

# --- Variance/SD for Continuous Random Variables ---
print("\n--- Variance/SD for Continuous RVs ---")

# For ISQ Values (Normal Distribution)
# isq_mean = 70, isq_std = 5 (from PDF section)
# For a Normal distribution N(μ, σ²), Var(X) = σ² and SD(X) = σ.
print("\nExample: ISQ Values (Normally Distributed N(μ=70, σ=5))")
var_isq = isq_std**2
sd_isq = isq_std # This is directly the parameter sigma
print(f"Var(X_ISQ) = σ² = ({isq_std})² = {var_isq}")
print(f"SD(X_ISQ) = σ = {sd_isq}")
print(f"Interpretation: ISQ values typically deviate from the mean of {isq_mean} by {sd_isq} ISQ units.")

# Plotting PDF with E[X] and SD regions for ISQ
# x_vals and pdf_vals are from the PDF section
plt.figure(figsize=(10, 6))
plt.plot(x_vals, pdf_vals, color='darkgreen', lw=2, label='PDF $f(x)$')
plt.axvline(isq_mean, color='blue', linestyle='dashed', linewidth=2, label=f'E[X] = $\mu$ = {isq_mean}')

# Shade E[X] ± 1*SD, E[X] ± 2*SD, E[X] ± 3*SD
plt.fill_between(x_vals, pdf_vals, where=(x_vals >= isq_mean - sd_isq) & (x_vals <= isq_mean + sd_isq), color='lightgreen', alpha=0.5, label=f'$\mu \pm 1\sigma$ ({norm.cdf(isq_mean + sd_isq, isq_mean, sd_isq) - norm.cdf(isq_mean - sd_isq, isq_mean, sd_isq):.1%})')
plt.fill_between(x_vals, pdf_vals, where=(x_vals >= isq_mean - 2*sd_isq) & (x_vals <= isq_mean + 2*sd_isq), color='yellowgreen', alpha=0.3, label=f'$\mu \pm 2\sigma$ ({norm.cdf(isq_mean + 2*sd_isq, isq_mean, sd_isq) - norm.cdf(isq_mean - 2*sd_isq, isq_mean, sd_isq):.1%})')
# For 3 sigma, adjust alpha if too dark or extend x_vals if needed
# plt.fill_between(x_vals, pdf_vals, where=(x_vals >= isq_mean - 3*sd_isq) & (x_vals <= isq_mean + 3*sd_isq), color='olivedrab', alpha=0.2, label='μ ± 3σ')

plt.title('ISQ PDF with E[X] and Standard Deviation (σ) Regions (Empirical Rule Context)')
plt.xlabel('ISQ Value (x)')
plt.ylabel('Probability Density $f(x)$')
plt.legend(fontsize=8)
plt.grid(True, linestyle='--')
plt.tight_layout()
print("Displaying ISQ PDF with E[X] and ±1σ, ±2σ regions marked...")
print("For a Normal distribution (bell curve):")
print(f"  ~68% of values fall within μ ± 1σ (approx. {isq_mean-sd_isq} to {isq_mean+sd_isq})")
print(f"  ~95% of values fall within μ ± 2σ (approx. {isq_mean-2*sd_isq} to {isq_mean+2*sd_isq})")
print(f"  ~99.7% of values fall within μ ± 3σ (approx. {isq_mean-3*sd_isq} to {isq_mean+3*sd_isq})")
plt.show()
print("-"*70)