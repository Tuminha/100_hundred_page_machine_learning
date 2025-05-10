"""
Demonstrates the concepts of Max and Arg Max based on Chapter 2 of
'The Hundred-Page Machine Learning Book'.

- Max: The maximum output value of a function over a given domain.
- Arg Max: The input value(s) from the domain that yield the maximum output.

Visualizes a sample function and annotates its max and arg max.
"""

import numpy as np
import matplotlib.pyplot as plt

print("--- Max and Arg Max Introduction ---")
print("Max f(x): The highest output value the function f achieves.")
print("Arg Max f(x): The input value x for which f(x) is maximized.")
print("-"*40)

# --- Define a Sample Function and Domain ---
# Let's use f(x) = -(x - 2)^2 + 10. This is a parabola opening downwards.
# Its theoretical maximum is 10, occurring at x = 2.

def sample_function(x):
    return -(x - 2)**2 + 10

# Define a discrete domain for x values for demonstration
# (In continuous cases, calculus or optimization algorithms find these)
x_domain = np.linspace(-2, 6, 100) # 100 points from -2 to 6

# Calculate function values (outputs) over the domain
y_values = sample_function(x_domain)

print(f"Sample function: f(x) = -(x - 2)^2 + 10")
print(f"Domain for x: from {x_domain.min():.1f} to {x_domain.max():.1f} with {len(x_domain)} points.")

# --- Calculate Max and Arg Max --- 

# Max f(x): The maximum value in y_values
max_y = np.max(y_values)

# Arg Max f(x): The x_value that corresponds to max_y
# np.argmax(y_values) gives the *index* of the maximum value in y_values
index_of_max_y = np.argmax(y_values)
arg_max_x = x_domain[index_of_max_y]

print(f"\nCalculated Max and Arg Max from the sampled data:")
print(f"Max f(x) (Maximum output value) ≈ {max_y:.4f}")
print(f"Arg Max f(x) (Input x that yields max output) ≈ {arg_max_x:.4f}")
print("(Theoretical Max = 10.0, Theoretical Arg Max = 2.0)")
print("Note: Numerical precision might lead to slight differences from theoretical values.")
print("-"*40)

# --- Visualization --- 

fig, ax = plt.subplots(figsize=(10, 6))

# Plot the function
ax.plot(x_domain, y_values, label='f(x) = -(x - 2)² + 10', color='dodgerblue')

# Highlight the maximum point
ax.scatter([arg_max_x], [max_y], color='red', s=100, zorder=5, label=f'Max Point (Arg Max, Max)')

# Annotate Arg Max on x-axis
ax.plot([arg_max_x, arg_max_x], [ax.get_ylim()[0], max_y], color='red', linestyle='--', alpha=0.7)
ax.text(arg_max_x, ax.get_ylim()[0] * 1.15 , f'Arg Max f(x) ≈ {arg_max_x:.2f}', color='red', ha='center')

# Annotate Max on y-axis
ax.plot([ax.get_xlim()[0], arg_max_x], [max_y, max_y], color='red', linestyle='--', alpha=0.7)
ax.text(ax.get_xlim()[0] * 0.9, max_y, f'Max f(x) ≈ {max_y:.2f}', color='red', va='center')

ax.set_xlabel("Input (x) - Domain")
ax.set_ylabel("Output f(x) - Values")
ax.set_title("Visualizing Max and Arg Max of a Function")
ax.legend(loc='lower center')
ax.grid(True, linestyle='--')
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)

print("\nDisplaying plot for Max and Arg Max...")
plt.show()

print("\n--- Example with a Discrete Set of Values (as in Tana notes) ---")
# Domain A = {-2, 1, 2, 4, 6}
# Function f(x) = -(x - 2)^2 + 3
discrete_domain_A = np.array([-2, 1, 2, 4, 6])

def f_discrete(x):
    return -(x - 2)**2 + 3

f_values_A = f_discrete(discrete_domain_A)

print(f"Discrete Domain A = {discrete_domain_A}")
print(f"Function f(x) = -(x - 2)^2 + 3")
print(f"Values f(A) = {f_values_A}")

max_f_A = np.max(f_values_A)
idx_max_f_A = np.argmax(f_values_A)
arg_max_f_A = discrete_domain_A[idx_max_f_A]

print(f"Max f(x) over A = {max_f_A}")        # Expected: 3
print(f"Arg Max f(x) over A = {arg_max_f_A}") # Expected: 2

# Visualization for Discrete Domain Example
fig_discrete, ax_discrete = plt.subplots(figsize=(8, 5))
bars = ax_discrete.bar(range(len(discrete_domain_A)), f_values_A, color='skyblue',
                       tick_label=[str(val) for val in discrete_domain_A])
ax_discrete.set_xlabel("Input x from Discrete Domain A")
ax_discrete.set_ylabel("Output f(x)")
ax_discrete.set_title("Max/Arg Max for f(x) = -(x-2)²+3 over Discrete Domain A")

# Highlight the max bar and annotate
bars[idx_max_f_A].set_color('salmon')
ax_discrete.text(idx_max_f_A, max_f_A + 0.2, f'Max f(x) = {max_f_A}\nArg Max f(x) = {arg_max_f_A}',
                 ha='center', color='black', bbox=dict(facecolor='wheat', alpha=0.5))
ax_discrete.grid(axis='y', linestyle='--')
print("Displaying plot for Discrete Domain Max/Arg Max...")
plt.show()
print("-"*40)

print("\n--- Max of a set vs Max of a function over a set ---")
# Using Dentistry Example: Torque values
torque_values_set_A = np.array([35, 47, 27, 38, 24])
print(f"Set of Torque Values (A) = {torque_values_set_A}")
max_in_set_A = np.max(torque_values_set_A)
idx_max_in_set_A = np.argmax(torque_values_set_A)
print(f"Max value IN the set A: {max_in_set_A} (This is max(A), at index {idx_max_in_set_A})")

# Applying function f(x) = x - 50 to the set A
def f_torque(x):
    return x - 50

torque_function_outputs = f_torque(torque_values_set_A)
print(f"Function f(x)=x-50 applied to A: {torque_function_outputs}")

max_output_from_function = np.max(torque_function_outputs)
idx_max_output = np.argmax(torque_function_outputs)
argmax_input_for_function = torque_values_set_A[idx_max_output]

print(f"Max f(x) for x in A: {max_output_from_function}")
print(f"Arg Max f(x) for x in A: {argmax_input_for_function} (input torque that gave max f(x))")

# Visualization for Max of Set vs Max of Function
fig_set_func, (ax_set, ax_func) = plt.subplots(1, 2, figsize=(12, 5))
fig_set_func.suptitle("Max of Set vs. Max of Function applied to Set Elements", fontsize=14)

# Plot 1: Max of the set A (Torque values)
bars_set = ax_set.bar(range(len(torque_values_set_A)), torque_values_set_A, color='lightgreen',
                      tick_label=[str(val) for val in torque_values_set_A])
ax_set.set_xlabel("Elements in Set A (Torque Values)")
ax_set.set_ylabel("Value")
ax_set.set_title("Original Set A of Torque Values")
bars_set[idx_max_in_set_A].set_color('forestgreen')
ax_set.text(idx_max_in_set_A, max_in_set_A + 2, f'Max(A) = {max_in_set_A}',
            ha='center', color='black', bbox=dict(facecolor='white', alpha=0.7))
ax_set.grid(axis='y', linestyle='--')

# Plot 2: Max of f(x) for x in A
bars_func = ax_func.bar(range(len(torque_values_set_A)), torque_function_outputs, color='lightcoral',
                        tick_label=[str(val) for val in torque_values_set_A])
ax_func.set_xlabel("Input x from Set A (Torque Values)")
ax_func.set_ylabel("Output f(x) = x - 50")
ax_func.set_title("f(x) = x - 50 applied to Elements of A")
bars_func[idx_max_output].set_color('crimson')
ax_func.text(idx_max_output, max_output_from_function + 2, 
             f'Max f(x) = {max_output_from_function}\n(Input x={argmax_input_for_function})',
             ha='center', color='black', bbox=dict(facecolor='white', alpha=0.7))
ax_func.grid(axis='y', linestyle='--')

plt.tight_layout(rect=[0, 0, 1, 0.96])
print("Displaying plots for Max of Set vs Max of Function...")
plt.show() 