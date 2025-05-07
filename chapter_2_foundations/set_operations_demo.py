"""
Demonstrates Set Operations based on Chapter 2 concepts of
'The Hundred-Page Machine Learning Book'.

Uses Python's built-in `set` data type and `matplotlib_venn` for visualization.

Covers:
- Membership (∈, ∉)
- Subset (⊆, ⊂)
- Union (∪)
- Intersection (∩)
- Difference (\)
- Cardinality (|A|)
- Venn Diagram Visualization
"""

# Note: No direct NumPy dependency needed here, Python sets are sufficient.
import matplotlib.pyplot as plt
# Import venn diagram functions - requires `pip install matplotlib-venn`
from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles

print("--- Defining Example Sets ---")
# General Numeric Sets
set_A = {1, 2, 3, 4}
set_B = {3, 4, 5, 6}
set_C = {1, 2}
print(f"Set A: {set_A}")
print(f"Set B: {set_B}")
print(f"Set C: {set_C} (A subset of A)")

# Dentistry Example Sets (Patient IDs)
# Patients receiving Implants
patients_implants = {'P001', 'P002', 'P003', 'P004', 'P005'}
# Patients receiving Crowns
patients_crowns = {'P003', 'P004', 'P006', 'P007'}
# Patients with Periodontal Treatment
patients_perio = {'P002', 'P005', 'P008'}

print(f"\nDental Set - Implant Patients: {patients_implants}")
print(f"Dental Set - Crown Patients: {patients_crowns}")
print(f"Dental Set - Perio Patients: {patients_perio}")
print("-"*35)

print("\n--- Set Operations --- ")

# 1. Membership (∈, ∉)
# Python: `in` and `not in` operators
print("\n1. Membership (∈, ∉):")
print(f"Is 3 in Set A? (3 ∈ A): {3 in set_A}") # True
print(f"Is 7 in Set A? (7 ∈ A): {7 in set_A}") # False
print(f"Is P001 an Implant Patient? ('P001' ∈ patients_implants): {'P001' in patients_implants}") # True
print(f"Is P001 a Crown Patient? ('P001' ∈ patients_crowns): {'P001' in patients_crowns}") # False

# 2. Subset (⊆, ⊂)
# Python: `<=` for subset (⊆), `<` for proper subset (⊂)
print("\n2. Subset (⊆, ⊂):")
print(f"Is Set C a subset of Set A? (C ⊆ A): {set_C <= set_A}") # True
print(f"Is Set A a subset of Set B? (A ⊆ B): {set_A <= set_B}") # False
print(f"Is Set C a proper subset of Set A? (C ⊂ A): {set_C < set_A}") # True (A has elements not in C)
print(f"Is Set A a proper subset of Set A? (A ⊂ A): {set_A < set_A}") # False

# 3. Union (∪)
# Python: `|` operator or `.union()` method
print("\n3. Union (∪): Elements in either set or both.")
print(f"A ∪ B: {set_A | set_B}")
print(f"Implants ∪ Crowns: {patients_implants.union(patients_crowns)}")

# 4. Intersection (∩)
# Python: `&` operator or `.intersection()` method
print("\n4. Intersection (∩): Elements common to both sets.")
print(f"A ∩ B: {set_A & set_B}")
print(f"Patients with Implants AND Crowns: {patients_implants.intersection(patients_crowns)}")

# 5. Difference (\)
# Python: `-` operator or `.difference()` method
print("\n5. Difference (\): Elements in the first set but not the second.")
print(f"A \ B (A - B): {set_A - set_B}") # Elements in A, not in B
print(f"B \ A (B - A): {set_B - set_A}") # Elements in B, not in A
print(f"Implant patients WITHOUT Crowns: {patients_implants.difference(patients_crowns)}")
print(f"Crown patients WITHOUT Implants: {patients_crowns.difference(patients_implants)}")

# 6. Cardinality (|A|)
# Python: `len()` function
print("\n6. Cardinality (|A|): Number of elements in the set.")
print(f"Number of elements in Set A (|A|): {len(set_A)}")
print(f"Number of Implant Patients: {len(patients_implants)}")
print(f"Number of patients with Implants OR Crowns (|Implants ∪ Crowns|): {len(patients_implants | patients_crowns)}")
print(f"Number of patients with Implants AND Crowns (|Implants ∩ Crowns|): {len(patients_implants & patients_crowns)}")
print("-"*35)

# --- Visualizations (Venn Diagrams) --- 
print("\n--- Visualizations (Venn Diagrams) ---")

# Example 1: Implants vs Crowns (2 Sets)
plt.figure(figsize=(7, 7))
venn2([patients_implants, patients_crowns], set_labels=('Implant Patients', 'Crown Patients'))
venn2_circles([patients_implants, patients_crowns], linestyle='dashed', linewidth=1, color='grey')
plt.title("Venn Diagram: Implant vs Crown Patients")
print("Displaying Venn Diagram (Implants vs Crowns)...")
plt.show()

# Example 2: Implants vs Crowns vs Perio (3 Sets)
plt.figure(figsize=(8, 8))
venn3([patients_implants, patients_crowns, patients_perio],
      set_labels=('Implants', 'Crowns', 'Perio Tx'))
venn3_circles([patients_implants, patients_crowns, patients_perio], linestyle='dashed', linewidth=1, color='grey')
plt.title("Venn Diagram: Implants vs Crowns vs Perio Treatment")
print("Displaying Venn Diagram (Implants vs Crowns vs Perio)...")
plt.show()

print("\nNote: The numbers in the Venn diagram sections represent the cardinality (count) of elements unique to that specific intersection or difference.") 