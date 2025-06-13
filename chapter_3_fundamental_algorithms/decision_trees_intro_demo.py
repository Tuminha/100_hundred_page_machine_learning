"""
Introduction to Decision Trees - Concepts and Structure.

Based on Chapter 3: Fundamental Algorithms (Decision Trees)
from 'The Hundred-Page Machine Learning Book' by Andriy Burkov,
and comprehensive study notes.

This script covers the core concepts of Decision Trees:
- What is Decision Tree Learning?
- Data and Notation (Dataset, Feature Vectors, Target Variables)
- The Model: Tree Structure & The "Best Question"
- Training: How a Tree is Built (Recursive Partitioning)
- Using the Model: Prediction Phase
- Important Considerations & Confusing Points
- Dental Applications & Examples
- Strengths & Weaknesses

Focus on dental applications throughout with rich visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import make_classification
import pandas as pd

print("--- Chapter 3: Decision Trees - Introduction & Concepts ---")

# --- 1. What is Decision Tree Learning? 🎯 ---
print("\n--- 1. What is Decision Tree Learning? ---")
print("   - Core Idea:")
print("     Decision trees are a shallow learning algorithm that, once trained on a dataset,")
print("     creates a sort of flowchart that helps in making decisions based on the features of the data.")
print("   - Classification Decision Tree:")
print("     The main goal is to classify a new observation into a category based on the input features.")
print("     For example, an implant that, after going through a set of conditions or a decision tree,")
print("     determines to which class it belongs, such as 'success' or 'failure.'")
print("   - Regression Decision Tree:")
print("     The model will predict a continuous output variable instead of a categorical one,")
print("     for example, predicting ISQ = 70.")

print("\n   - Key Concept:")
print("     Instead of a formula, the model uses a simple 'yes' or 'no' flowchart")
print("     to navigate through the various conditions and ultimately arrive at a decision.")
print("     Think of it as a flowchart with 20 questions that, depending on the answer")
print("     in each step, leads to a final decision about the classification or regression outcome.")

print("\n   - Dental Relevance:")
print("     From a dental perspective, decision trees can be very useful in assisting")
print("     clinicians to make decisions or predict outcomes based on observations made during procedures.")
print("     Examples include:")
print("     • Estimating likelihood of periodontal disease based on age, smoking habits, oral hygiene")
print("     • Predicting implant success based on torque, ISQ, bone density, patient factors")
print("     • Determining treatment acceptance based on patient demographics and treatment complexity")

print("\n--- 1.1. Simple Decision Tree Structure Visualization ---")
# Create a simple tree structure diagram
fig, ax = plt.subplots(figsize=(12, 8))

# Define positions for nodes
positions = {
    'root': (6, 7),
    'left1': (3, 5),
    'right1': (9, 5),
    'left2': (1.5, 3),
    'right2': (4.5, 3),
    'left3': (7.5, 3),
    'right3': (10.5, 3)
}

# Define node styles
def draw_node(ax, pos, text, node_type='internal', color='lightblue'):
    if node_type == 'leaf':
        color = 'lightgreen'
    elif node_type == 'root':
        color = 'lightcoral'
    
    box = FancyBboxPatch((pos[0]-0.8, pos[1]-0.4), 1.6, 0.8, 
                         boxstyle="round,pad=0.1", 
                         facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(box)
    ax.text(pos[0], pos[1], text, ha='center', va='center', fontsize=9, fontweight='bold')

# Draw nodes
draw_node(ax, positions['root'], 'ISQ > 70?', 'root')
draw_node(ax, positions['left1'], 'Smoker?', 'internal')
draw_node(ax, positions['right1'], 'Torque > 35?', 'internal')
draw_node(ax, positions['left2'], 'FAILURE', 'leaf')
draw_node(ax, positions['right2'], 'Age > 50?', 'internal')
draw_node(ax, positions['left3'], 'SUCCESS', 'leaf')
draw_node(ax, positions['right3'], 'SUCCESS', 'leaf')

# Draw additional leaf
draw_node(ax, (3, 1), 'FAILURE', 'leaf')
draw_node(ax, (6, 1), 'SUCCESS', 'leaf')

# Draw edges with labels
def draw_edge(ax, start, end, label, offset=0.2):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    mid_x, mid_y = (start[0] + end[0])/2, (start[1] + end[1])/2
    ax.text(mid_x + offset, mid_y, label, fontsize=8, ha='center', 
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

# Draw edges
draw_edge(ax, positions['root'], positions['left1'], 'No', -0.3)
draw_edge(ax, positions['root'], positions['right1'], 'Yes', 0.3)
draw_edge(ax, positions['left1'], positions['left2'], 'Yes', -0.3)
draw_edge(ax, positions['left1'], positions['right2'], 'No', 0.3)
draw_edge(ax, positions['right1'], positions['left3'], 'No', -0.3)
draw_edge(ax, positions['right1'], positions['right3'], 'Yes', 0.3)
draw_edge(ax, positions['right2'], (3, 1), 'Yes', -0.3)
draw_edge(ax, positions['right2'], (6, 1), 'No', 0.3)

ax.set_xlim(-1, 13)
ax.set_ylim(0, 8)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Decision Tree Structure for Dental Implant Success Prediction', 
             fontsize=14, fontweight='bold', pad=20)

# Add legend
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, facecolor='lightcoral', edgecolor='black', label='Root Node'),
    plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='black', label='Internal Node'),
    plt.Rectangle((0, 0), 1, 1, facecolor='lightgreen', edgecolor='black', label='Leaf Node (Decision)')
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))

plot_save_path = "../plots/chapter_3/decision_trees/tree_structure_example.png"
plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
print(f"   Tree structure diagram saved to: {plot_save_path}")
print("   Displaying simple decision tree structure...")
plt.show()

# --- 2. The Building Blocks: Data and Notation 🧱 ---
print("\n--- 2. The Building Blocks: Data and Notation ---")
print("   - Dataset Notation:")
print("     The dataset is similar to previous models: {xi, yi}")
print("     • xi: Input features (patient characteristics, clinical measurements)")
print("     • yi: Target variable (outcome we want to predict)")

print("\n   - Target Variable:")
print("     • Classification Decision Tree: yi ∈ {0, 1}")
print("       - '0' = Failure (e.g., implant failure, treatment rejection)")
print("       - '1' = Success (e.g., implant success, treatment acceptance)")
print("     • Regression Decision Tree: yi ∈ ℝ (continuous values)")
print("       - Examples: Predicted ISQ value, Expected torque, Bone density score")

print("\n   - Feature Vector:")
print("     xi is a D-dimensional vector where D = number of features")
print("     Each feature contributes to the model's decision-making process")
print("     Examples of dental features:")
print("     • Patient factors: Age, Gender, Smoking status, Medical history")
print("     • Clinical measurements: Torque, ISQ, Bone density (HU)")
print("     • Procedural factors: Implant type, Placement technique")

print("\n   --- Example Dental Dataset ---")
# Create mock dental data for examples
np.random.seed(42)
n_patients = 100

# Generate realistic dental implant data
ages = np.random.normal(55, 15, n_patients).astype(int)
ages = np.clip(ages, 25, 85)
smoking = np.random.choice([0, 1], n_patients, p=[0.7, 0.3])  # 30% smokers
torque = np.random.normal(35, 10, n_patients)
torque = np.clip(torque, 15, 60)
isq = np.random.normal(65, 12, n_patients)
isq = np.clip(isq, 40, 85)

# Create target variable with realistic relationships
success_prob = 0.3 + 0.4 * (isq > 70) + 0.2 * (torque > 30) - 0.3 * smoking - 0.1 * (ages > 65)
success_prob = np.clip(success_prob, 0.1, 0.9)
success = np.random.binomial(1, success_prob, n_patients)

# Create DataFrame for easy manipulation
dental_data = pd.DataFrame({
    'Age': ages,
    'Smoker': smoking,
    'Torque': torque,
    'ISQ': isq,
    'Success': success
})

print("   Sample of dental implant dataset:")
print("   Features: Age, Smoker (0=No, 1=Yes), Torque (Nm), ISQ")
print("   Target: Success (0=Failure, 1=Success)")
print("\n   First 5 patients:")
print(dental_data.head())

print(f"\n   Dataset summary:")
print(f"   • Total patients: {n_patients}")
print(f"   • Success rate: {success.mean():.1%}")
print(f"   • Average age: {ages.mean():.1f} years")
print(f"   • Smokers: {smoking.mean():.1%}")
print(f"   • Average torque: {torque.mean():.1f} Nm")
print(f"   • Average ISQ: {isq.mean():.1f}") 

# --- 3. The Model: Tree Structure & The "Best Question" ⚙️ ---
print("\n--- 3. The Model: Tree Structure & The 'Best Question' ---")
print("   - Structure of a tree:")
print("     • Root node: First question that splits the entire dataset based on a specific feature")
print("     • Internal node: Subsequent questions that further divide the dataset into smaller subsets")
print("     • Branches: 'Yes' or 'No' pathways emerging from each node")
print("     • Leaf node: Final decision that returns the prediction")

print("\n   - The best question: How is the split chosen?")
print("     The decision tree searches for the split that returns the lowest impurity.")
print("     It finds the feature and threshold that provides the best separation of data points.")
print("     We use two main methods to measure impurity:")

print("\n   --- Impurity Measures ---")
print("   1. Entropy (Information Theory):")
print("      H(S) = -[p₁·log₂(p₁) + p₀·log₂(p₀)]")
print("      where p₁ = proportion of class 1, p₀ = proportion of class 0")
print("   2. Gini Impurity:")
print("      Gini(S) = 1 - [p₁² + p₀²]")
print("      Alternative measure, computationally simpler than entropy")

print("\n   --- Example of Pure vs Impure Nodes ---")
print("   Let's calculate entropy for different scenarios:")

# Example calculations from the Tana notes
print("\n   Scenario 1: Perfect separation by Torque > 30")
print("   Group 1 (Torque > 30): 6 successes, 0 failures")
print("   • p₁ = 6/6 = 1.0, p₀ = 0/6 = 0.0")
print("   • H(S₊) = -[1.0·log₂(1.0) + 0.0·log₂(0.0)] = -[0 + 0] = 0")
print("   • This is a PERFECTLY PURE node (entropy = 0)")

print("\n   Group 2 (Torque ≤ 30): 0 successes, 4 failures")
print("   • p₁ = 0/4 = 0.0, p₀ = 4/4 = 1.0") 
print("   • H(S₋) = -[0.0·log₂(0.0) + 1.0·log₂(1.0)] = -[0 + 0] = 0")
print("   • This is also PERFECTLY PURE (entropy = 0)")

print("\n   Scenario 2: Mixed group (impure)")
print("   Mixed group: 3 successes, 3 failures")
print("   • p₁ = 3/6 = 0.5, p₀ = 3/6 = 0.5")
print("   • H(S) = -[0.5·log₂(0.5) + 0.5·log₂(0.5)] = -[0.5·(-1) + 0.5·(-1)] = 1.0")
print("   • This is MAXIMALLY IMPURE (entropy = 1.0)")

print("\n--- 3.1. Visualizing Impurity Measures ---")
# Create entropy and Gini impurity visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Calculate entropy and Gini for different proportions
p1_range = np.linspace(0.001, 0.999, 100)
entropy_vals = -p1_range * np.log2(p1_range) - (1-p1_range) * np.log2(1-p1_range)
gini_vals = 1 - (p1_range**2 + (1-p1_range)**2)

# Plot entropy
ax1.plot(p1_range, entropy_vals, 'b-', linewidth=3, label='Entropy')
ax1.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='Pure Node (H=0)')
ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Max Impurity (H=1)')
ax1.scatter([0.5], [1.0], color='red', s=100, zorder=5, label='Worst Case (p=0.5)')
ax1.scatter([1.0, 0.0], [0.0, 0.0], color='green', s=100, zorder=5, label='Best Cases (p=0,1)')
ax1.set_xlabel('Proportion of Success (p₁)')
ax1.set_ylabel('Entropy H(S)')
ax1.set_title('Entropy: Measure of Impurity')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot Gini
ax2.plot(p1_range, gini_vals, 'r-', linewidth=3, label='Gini Impurity')
ax2.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='Pure Node (Gini=0)')
ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Max Impurity (Gini=0.5)')
ax2.scatter([0.5], [0.5], color='red', s=100, zorder=5, label='Worst Case (p=0.5)')
ax2.scatter([1.0, 0.0], [0.0, 0.0], color='green', s=100, zorder=5, label='Best Cases (p=0,1)')
ax2.set_xlabel('Proportion of Success (p₁)')
ax2.set_ylabel('Gini Impurity')
ax2.set_title('Gini Impurity: Alternative Measure')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plot_save_path = "../plots/chapter_3/decision_trees/impurity_measures_comparison.png"
plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
print(f"   Impurity measures comparison saved to: {plot_save_path}")
print("   Displaying impurity measures comparison...")
plt.show()

print("\n--- 3.2. Information Gain Calculation ---")
print("   Information Gain = Entropy(parent) - Weighted_Average_Entropy(children)")
print("   The split with the HIGHEST information gain is chosen.")

# Calculate information gain for a sample split
def calculate_entropy(y):
    if len(y) == 0:
        return 0
    p1 = np.mean(y)
    if p1 == 0 or p1 == 1:
        return 0
    return -p1 * np.log2(p1) - (1-p1) * np.log2(1-p1)

def information_gain(y_parent, y_left, y_right):
    n_parent = len(y_parent)
    n_left, n_right = len(y_left), len(y_right)
    
    entropy_parent = calculate_entropy(y_parent)
    entropy_left = calculate_entropy(y_left)
    entropy_right = calculate_entropy(y_right)
    
    weighted_entropy = (n_left/n_parent) * entropy_left + (n_right/n_parent) * entropy_right
    return entropy_parent - weighted_entropy

# Example: Compare different splits on our dental data
print("\n   Example: Comparing splits on dental data")
y = dental_data['Success'].values

# Split 1: ISQ > 70
isq_split = dental_data['ISQ'] > 70
y_left_isq = y[~isq_split]
y_right_isq = y[isq_split]
ig_isq = information_gain(y, y_left_isq, y_right_isq)

# Split 2: Age > 60
age_split = dental_data['Age'] > 60
y_left_age = y[~age_split]
y_right_age = y[age_split]
ig_age = information_gain(y, y_left_age, y_right_age)

# Split 3: Smoker = 1
smoke_split = dental_data['Smoker'] == 1
y_left_smoke = y[~smoke_split]
y_right_smoke = y[smoke_split]
ig_smoke = information_gain(y, y_left_smoke, y_right_smoke)

print(f"   • Information Gain for ISQ > 70: {ig_isq:.4f}")
print(f"   • Information Gain for Age > 60: {ig_age:.4f}")
print(f"   • Information Gain for Smoker = Yes: {ig_smoke:.4f}")
print(f"   → Best split: {'ISQ > 70' if ig_isq == max(ig_isq, ig_age, ig_smoke) else 'Age > 60' if ig_age == max(ig_isq, ig_age, ig_smoke) else 'Smoker = Yes'}")

# Visualize information gain comparison
fig, ax = plt.subplots(figsize=(10, 6))
splits = ['ISQ > 70', 'Age > 60', 'Smoker = Yes']
gains = [ig_isq, ig_age, ig_smoke]
colors = ['skyblue', 'lightcoral', 'lightgreen']

bars = ax.bar(splits, gains, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Information Gain')
ax.set_title('Information Gain Comparison for Different Splits\n(Higher is Better)', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, gain in zip(bars, gains):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
            f'{gain:.4f}', ha='center', va='bottom', fontweight='bold')

# Highlight the best split
best_idx = np.argmax(gains)
bars[best_idx].set_color('gold')
bars[best_idx].set_edgecolor('darkgoldenrod')
bars[best_idx].set_linewidth(3)

plot_save_path = "../plots/chapter_3/decision_trees/information_gain_comparison.png"
plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
print(f"   Information gain comparison saved to: {plot_save_path}")
print("   Displaying information gain comparison...")
plt.show()

# --- 4. Training the Model: How a Tree is Built 🏋️‍♂️ ---
print("\n--- 4. Training the Model: How a Tree is Built ---")
print("   - Goal of Training:")
print("     To divide the dataset into questions that return the lowest impurity,")
print("     allowing the model to make more accurate predictions based on the features.")

print("\n   - The Algorithm: Recursive Partitioning")
print("     1. Start at the root node")
print("     2. Test every potential split on every feature")
print("     3. Calculate entropy/impurity for each potential split")
print("     4. Choose the split with highest information gain (lowest weighted impurity)")
print("     5. Divide dataset into 2 subsets")
print("     6. Repeat steps 1-5 for each child node")

print("\n   - When to Stop Splitting: The Problem of Overfitting")
print("     If we allow endless branching, the tree will:")
print("     • Perform well on training data")
print("     • Fail to generalize to new data (overfitting)")
print("     Solutions:")
print("     • Pre-pruning: Stop growing before too complex")
print("       - Minimum samples per leaf")
print("       - Maximum depth limit")
print("       - Minimum information gain threshold")
print("     • Post-pruning: Grow fully, then remove unimportant branches")

print("\n--- 4.1. Tree Building Process Visualization ---")
# Create a step-by-step tree building visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

# Step 1: Initial dataset
ax = axes[0]
success_patients = dental_data[dental_data['Success'] == 1]
failure_patients = dental_data[dental_data['Success'] == 0]

ax.scatter(success_patients['ISQ'], success_patients['Torque'], 
          c='green', alpha=0.6, label=f'Success ({len(success_patients)})', s=50)
ax.scatter(failure_patients['ISQ'], failure_patients['Torque'], 
          c='red', alpha=0.6, label=f'Failure ({len(failure_patients)})', s=50)
ax.set_xlabel('ISQ')
ax.set_ylabel('Torque (Nm)')
ax.set_title('Step 1: Initial Dataset\n(Mixed - High Impurity)')
ax.legend()
ax.grid(True, alpha=0.3)

# Step 2: First split (ISQ > 70)
ax = axes[1]
ax.scatter(success_patients['ISQ'], success_patients['Torque'], 
          c='green', alpha=0.6, label=f'Success ({len(success_patients)})', s=50)
ax.scatter(failure_patients['ISQ'], failure_patients['Torque'], 
          c='red', alpha=0.6, label=f'Failure ({len(failure_patients)})', s=50)
ax.axvline(x=70, color='blue', linestyle='--', linewidth=3, label='Split: ISQ > 70')
ax.set_xlabel('ISQ')
ax.set_ylabel('Torque (Nm)')
ax.set_title('Step 2: First Split - ISQ > 70\n(Separates data into 2 regions)')
ax.legend()
ax.grid(True, alpha=0.3)

# Step 3: Second split (Smoker in left region)
ax = axes[2]
# Color by smoking status in left region
left_region = dental_data[dental_data['ISQ'] <= 70]
left_success = left_region[left_region['Success'] == 1]
left_failure = left_region[left_region['Success'] == 0]

# Different shapes for smokers/non-smokers
for idx, row in left_success.iterrows():
    marker = '^' if row['Smoker'] == 1 else 'o'
    ax.scatter(row['ISQ'], row['Torque'], c='green', alpha=0.7, s=80, marker=marker)

for idx, row in left_failure.iterrows():
    marker = '^' if row['Smoker'] == 1 else 'o'
    ax.scatter(row['ISQ'], row['Torque'], c='red', alpha=0.7, s=80, marker=marker)

ax.axvline(x=70, color='blue', linestyle='--', linewidth=2, alpha=0.7)
ax.set_xlabel('ISQ')
ax.set_ylabel('Torque (Nm)')
ax.set_title('Step 3: Focus on Left Region\n(ISQ ≤ 70) - Split by Smoking Status')
ax.legend(['Success-NonSmoker (○)', 'Success-Smoker (△)', 'Failure-NonSmoker (○)', 'Failure-Smoker (△)'])
ax.grid(True, alpha=0.3)

# Step 4: Final tree regions
ax = axes[3]
# Create decision boundary visualization
x_min, x_max = dental_data['ISQ'].min() - 5, dental_data['ISQ'].max() + 5
y_min, y_max = dental_data['Torque'].min() - 5, dental_data['Torque'].max() + 5

# Draw rectangles for each region
ax.add_patch(plt.Rectangle((x_min, y_min), 70-x_min, y_max-y_min, 
                          alpha=0.2, color='red', label='High Risk Region'))
ax.add_patch(plt.Rectangle((70, y_min), x_max-70, y_max-y_min, 
                          alpha=0.2, color='green', label='Low Risk Region'))

ax.scatter(success_patients['ISQ'], success_patients['Torque'], 
          c='green', alpha=0.8, label='Success', s=50, edgecolors='black', linewidth=0.5)
ax.scatter(failure_patients['ISQ'], failure_patients['Torque'], 
          c='red', alpha=0.8, label='Failure', s=50, edgecolors='black', linewidth=0.5)

ax.axvline(x=70, color='blue', linestyle='-', linewidth=3, label='Primary Split: ISQ = 70')
ax.set_xlabel('ISQ')
ax.set_ylabel('Torque (Nm)')
ax.set_title('Step 4: Final Decision Regions\n(Rectangular Boundaries)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_save_path = "../plots/chapter_3/decision_trees/tree_building_process.png"
plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
print(f"   Tree building process visualization saved to: {plot_save_path}")
print("   Displaying tree building process...")
plt.show()

# --- 5. Using the Model: Prediction Phase 🔮 ---
print("\n--- 5. Using the Model: Prediction Phase ---")
print("   - How it works:")
print("     Take new patient data point features {x}, for example, {Age, Smoker, Torque, ISQ}")
print("     This will be the input to the model for making predictions using the decision tree.")

print("\n   - Process:")
print("     1. Start at the root node with the new data point")
print("     2. Traverse down the tree based on the feature values")
print("     3. Follow the branches/leaves until reaching a terminal node")
print("     4. The terminal node provides the final classification or numerical prediction")

print("\n   - Final prediction:")
print("     • Classification: Output the majority class of samples in the leaf node")
print("     • Regression: Output the average of values of samples in the leaf node")

print("\n   --- Dental Example: Patient Prediction Flow ---")
print("   Example patient data:")
print("   • Age: 45")
print("   • Smoker: Yes (1)")
print("   • Implant_torque: 35 Nm")  
print("   • Implant_ISQ: 65")

print("\n   Decision tree flow (after training):")
print("   Root node: Is ISQ > 70? → No (ISQ = 65)")
print("   ├─ Is patient a Smoker? → Yes (Smoker = 1)")
print("   │  └─ Prediction: FAILURE")
print("   Flow summary: ISQ ≤ 70 → Smoker = Yes → FAILURE")

# Create a real decision tree and show prediction process
print("\n--- 5.1. Real Decision Tree Training and Prediction ---")
# Train a simple decision tree on our dental data
X_train = dental_data[['Age', 'Smoker', 'Torque', 'ISQ']].values
y_train = dental_data['Success'].values

# Create and train decision tree
dt_classifier = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_classifier.fit(X_train, y_train)

# Visualize the actual trained tree
fig, ax = plt.subplots(figsize=(20, 12))
plot_tree(dt_classifier, 
          feature_names=['Age', 'Smoker', 'Torque', 'ISQ'],
          class_names=['Failure', 'Success'],
          filled=True, 
          rounded=True,
          fontsize=12,
          ax=ax)

ax.set_title('Trained Decision Tree for Dental Implant Success Prediction', 
             fontsize=16, fontweight='bold', pad=20)

plot_save_path = "../plots/chapter_3/decision_trees/trained_decision_tree.png"
plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
print(f"   Trained decision tree saved to: {plot_save_path}")
print("   Displaying trained decision tree...")
plt.show()

# Example predictions
print("\n   --- Example Predictions ---")
test_patients = [
    [45, 1, 35, 65],  # Age=45, Smoker=Yes, Torque=35, ISQ=65
    [35, 0, 40, 75],  # Age=35, Smoker=No, Torque=40, ISQ=75  
    [70, 1, 25, 55],  # Age=70, Smoker=Yes, Torque=25, ISQ=55
]

patient_descriptions = [
    "Patient 1: Age=45, Smoker=Yes, Torque=35, ISQ=65",
    "Patient 2: Age=35, Smoker=No, Torque=40, ISQ=75", 
    "Patient 3: Age=70, Smoker=Yes, Torque=25, ISQ=55"
]

predictions = dt_classifier.predict(test_patients)
probabilities = dt_classifier.predict_proba(test_patients)

for i, (desc, pred, prob) in enumerate(zip(patient_descriptions, predictions, probabilities)):
    outcome = "SUCCESS" if pred == 1 else "FAILURE"
    prob_success = prob[1]
    print(f"   {desc}")
    print(f"   → Prediction: {outcome} (Success probability: {prob_success:.3f})")
    print()

# --- 6. Important Considerations & Potential Confusing Points 🤔 ---
print("\n--- 6. Important Considerations & Potential Confusing Points ---")

print("\n   - Greedy Nature:")
print("     The model makes optimal decisions at each step but does not consider")
print("     the overall consequences of those choices. This might not result in")
print("     the best overall tree, but can still produce a satisfactory model.")

print("\n   - Instability:")
print("     A small change in the training data can produce a very different decision tree.")
print("     This makes individual trees less reliable than ensemble methods.")

print("\n   - Non-parametric Model:")
print("     Unlike linear and logistic regression, decision trees do not have fixed")
print("     parameters (w, b). The model structure itself is learned from data.")

print("\n   - Decision Boundaries:")
print("     Decision trees create rectangular decision boundaries, which can lead")
print("     to a lack of flexibility in modeling complex relationships.")

print("\n   - Feature Importance:")
print("     Easy to check which features are important by looking at the tree")
print("     and checking which ones are closer to the root node.")

print("\n--- 6.1. Feature Importance Visualization ---")
# Get feature importance from trained tree
feature_names = ['Age', 'Smoker', 'Torque', 'ISQ']
importances = dt_classifier.feature_importances_

# Create feature importance plot
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
bars = ax.bar(feature_names, importances, color=colors, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Feature Importance')
ax.set_title('Feature Importance in Decision Tree\n(Higher values indicate more important features)', 
             fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, importance in zip(bars, importances):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{importance:.3f}', ha='center', va='bottom', fontweight='bold')

# Highlight the most important feature
max_idx = np.argmax(importances)
bars[max_idx].set_color('darkgreen')
bars[max_idx].set_edgecolor('darkgreen')
bars[max_idx].set_linewidth(3)

plot_save_path = "../plots/chapter_3/decision_trees/feature_importance.png"
plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
print(f"   Feature importance plot saved to: {plot_save_path}")
print("   Displaying feature importance...")
plt.show()

print(f"\n   Feature importance ranking:")
importance_ranking = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
for i, (feature, importance) in enumerate(importance_ranking, 1):
    print(f"   {i}. {feature}: {importance:.3f}")

print("\n--- 6.2. Decision Boundary Limitations ---")
# Show rectangular decision boundaries
fig, ax = plt.subplots(figsize=(12, 8))

# Plot data points
success_patients = dental_data[dental_data['Success'] == 1]
failure_patients = dental_data[dental_data['Success'] == 0]

ax.scatter(success_patients['ISQ'], success_patients['Torque'], 
          c='green', alpha=0.7, label='Success', s=60, edgecolors='black', linewidth=0.5)
ax.scatter(failure_patients['ISQ'], failure_patients['Torque'], 
          c='red', alpha=0.7, label='Failure', s=60, edgecolors='black', linewidth=0.5)

# Create a mesh to show decision boundaries
x_min, x_max = dental_data['ISQ'].min() - 5, dental_data['ISQ'].max() + 5
y_min, y_max = dental_data['Torque'].min() - 5, dental_data['Torque'].max() + 5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5), np.arange(y_min, y_max, 0.5))

# Use average values for Age and Smoker for prediction mesh
avg_age = dental_data['Age'].mean()
avg_smoker = dental_data['Smoker'].mean()
mesh_features = np.c_[np.full(xx.ravel().shape, avg_age), 
                      np.full(xx.ravel().shape, avg_smoker),
                      yy.ravel(), xx.ravel()]

Z = dt_classifier.predict(mesh_features)
Z = Z.reshape(xx.shape)

# Plot decision boundary
ax.contourf(xx, yy, Z, alpha=0.3, colors=['red', 'green'])
ax.contour(xx, yy, Z, colors='black', linewidths=1, linestyles='--', alpha=0.7)

ax.set_xlabel('ISQ')
ax.set_ylabel('Torque (Nm)')
ax.set_title('Decision Tree Boundaries: Rectangular Regions\n(Averaged over Age and Smoking Status)', 
             fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plot_save_path = "../plots/chapter_3/decision_trees/decision_boundaries.png"
plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
print(f"   Decision boundaries visualization saved to: {plot_save_path}")
print("   Displaying decision boundaries...")
plt.show()

# --- 7. Dental Applications & Examples Summarized 🦷 ---
print("\n--- 7. Dental Applications & Examples Summarized ---")
print("   Decision trees in dentistry can have multiple practical applications:")

print("\n   • Periodontal Disease Risk Assessment:")
print("     Predict a patient's risk of developing severe complications based on")
print("     clinical observations such as age, smoking status, and systemic conditions.")

print("\n   • Implant Success Prediction:")
print("     Assess likelihood of implant complications based on bone density,")
print("     surgical torque, ISQ values, patient health status, and lifestyle factors.")

print("\n   • Treatment Acceptance Modeling:")
print("     Predict likelihood of patient accepting treatment options based on")
print("     demographics, treatment complexity, cost, and previous experiences.")

print("\n   • Clinical Decision Support:")
print("     Assist in treatment planning by providing evidence-based recommendations")
print("     based on patient characteristics and clinical measurements.")

print("\n--- 7.1. Comprehensive Dental Decision Tree Example ---")
# Create a more comprehensive example with multiple dental scenarios
dental_scenarios = {
    'Periodontal Disease Risk': {
        'features': ['Age', 'Smoking', 'Diabetes', 'Plaque_Score'],
        'thresholds': ['>45', 'Yes', 'Yes', '>3'],
        'high_risk_combinations': ['Age>45 + Smoking', 'Diabetes + High_Plaque', 'Age>60 + Diabetes'],
        'color': 'orange'
    },
    'Implant Success': {
        'features': ['Bone_Quality', 'Torque', 'ISQ', 'Smoking'],
        'thresholds': ['Dense', '>30Nm', '>70', 'No'],
        'success_combinations': ['Dense_Bone + High_ISQ', 'Good_Torque + Non_Smoker'],
        'color': 'blue'
    },
    'Treatment Acceptance': {
        'features': ['Age', 'Income', 'Complexity', 'Previous_Experience'],
        'thresholds': ['<65', 'High', 'Simple', 'Positive'],
        'acceptance_combinations': ['Young + High_Income', 'Simple + Positive_Experience'],
        'color': 'purple'
    }
}

# Create a comprehensive visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, (scenario, details) in enumerate(dental_scenarios.items()):
    ax = axes[idx]
    
    # Create a simple tree structure for each scenario
    y_positions = [3, 2, 1, 0]
    x_positions = [0, 1, 2, 3]
    
    # Root node
    ax.scatter([1.5], [3], s=500, c=details['color'], alpha=0.7, edgecolors='black', linewidth=2)
    ax.text(1.5, 3, details['features'][0], ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Level 1 nodes
    ax.scatter([0.5, 2.5], [2, 2], s=400, c=details['color'], alpha=0.5, edgecolors='black', linewidth=1.5)
    ax.text(0.5, 2, details['features'][1], ha='center', va='center', fontweight='bold', fontsize=9)
    ax.text(2.5, 2, details['features'][2], ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Leaf nodes
    ax.scatter([0, 1, 2, 3], [1, 1, 1, 1], s=300, c=['red', 'green', 'red', 'green'], 
              alpha=0.8, edgecolors='black', linewidth=1)
    
    outcomes = ['High Risk', 'Low Risk', 'High Risk', 'Low Risk'] if 'Risk' in scenario else \
               ['Failure', 'Success', 'Failure', 'Success'] if 'Success' in scenario else \
               ['Reject', 'Accept', 'Reject', 'Accept']
    
    for i, outcome in enumerate(outcomes):
        ax.text(i, 1, outcome, ha='center', va='center', fontweight='bold', fontsize=8)
    
    # Draw connections
    connections = [
        ([1.5, 3], [0.5, 2]),  # Root to left
        ([1.5, 3], [2.5, 2]),  # Root to right
        ([0.5, 2], [0, 1]),    # Left to leaf 1
        ([0.5, 2], [1, 1]),    # Left to leaf 2
        ([2.5, 2], [2, 1]),    # Right to leaf 3
        ([2.5, 2], [3, 1])     # Right to leaf 4
    ]
    
    for start, end in connections:
        ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', alpha=0.6, linewidth=1.5)
    
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0.5, 3.5)
    ax.set_title(scenario, fontweight='bold', fontsize=12)
    ax.axis('off')

plt.suptitle('Decision Trees in Dental Practice: Multiple Applications', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

plot_save_path = "../plots/chapter_3/decision_trees/dental_applications_overview.png"
plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
print(f"   Dental applications overview saved to: {plot_save_path}")
print("   Displaying dental applications overview...")
plt.show()

# --- 8. Strengths & Weaknesses of Decision Trees 💪 ---
print("\n--- 8. Strengths & Weaknesses of Decision Trees ---")

print("\n   --- Strengths ---")
print("   ✓ Can capture nonlinear relationships")
print("   ✓ Easy to interpret and visualize (not a black box)")
print("   ✓ Accepts both numerical and categorical inputs")
print("   ✓ Provides clear decisions based on data")
print("   ✓ No need to scale features (not based on weights and distances)")
print("   ✓ Handles missing values well")
print("   ✓ Can capture feature interactions automatically")

print("\n   --- Weaknesses ---")
print("   ✗ Can overfit easily if not pruned properly")
print("   ✗ Unstable: small data changes create different trees")
print("   ✗ Greedy approach is not guaranteed to be optimal")
print("   ✗ Struggles with diagonal relationships between features")
print("   ✗ Biased toward features with more levels")
print("   ✗ Can create overly complex trees with noise")
print("   ✗ Limited expressiveness compared to other algorithms")

print("\n--- 8.1. Strengths vs Weaknesses Comparison ---")
# Create a visual comparison of strengths and weaknesses
strengths = [
    'Nonlinear relationships',
    'Easy interpretation', 
    'Mixed data types',
    'No feature scaling',
    'Handles missing values',
    'Feature interactions',
    'Clear decisions'
]

weaknesses = [
    'Prone to overfitting',
    'High instability',
    'Greedy optimization',
    'Poor diagonal boundaries',
    'Feature bias',
    'Noise sensitivity',
    'Limited expressiveness'
]

# Create importance scores (mock scores for visualization)
strength_scores = [0.9, 0.95, 0.8, 0.85, 0.7, 0.8, 0.9]
weakness_scores = [0.8, 0.9, 0.6, 0.7, 0.5, 0.75, 0.6]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Strengths
y_pos = np.arange(len(strengths))
bars1 = ax1.barh(y_pos, strength_scores, color='green', alpha=0.7, edgecolor='darkgreen', linewidth=1.5)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(strengths)
ax1.set_xlabel('Advantage Level')
ax1.set_title('Decision Tree Strengths', fontweight='bold', color='green')
ax1.set_xlim(0, 1)
ax1.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (bar, score) in enumerate(zip(bars1, strength_scores)):
    ax1.text(score + 0.02, bar.get_y() + bar.get_height()/2, 
             f'{score:.1f}', va='center', fontweight='bold')

# Weaknesses  
bars2 = ax2.barh(y_pos, weakness_scores, color='red', alpha=0.7, edgecolor='darkred', linewidth=1.5)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(weaknesses)
ax2.set_xlabel('Concern Level')
ax2.set_title('Decision Tree Weaknesses', fontweight='bold', color='red')
ax2.set_xlim(0, 1)
ax2.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (bar, score) in enumerate(zip(bars2, weakness_scores)):
    ax2.text(score + 0.02, bar.get_y() + bar.get_height()/2, 
             f'{score:.1f}', va='center', fontweight='bold')

plt.tight_layout()

plot_save_path = "../plots/chapter_3/decision_trees/strengths_vs_weaknesses.png"
plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
print(f"   Strengths vs weaknesses comparison saved to: {plot_save_path}")
print("   Displaying strengths vs weaknesses...")
plt.show()

# Final summary
print("\n--- End of Decision Trees Introduction ---")
print("   This comprehensive introduction covers:")
print("   1. ✓ Core concepts and dental applications")
print("   2. ✓ Tree structure and impurity measures") 
print("   3. ✓ Training process and information gain")
print("   4. ✓ Prediction phase with real examples")
print("   5. ✓ Important considerations and limitations")
print("   6. ✓ Dental practice applications")
print("   7. ✓ Comprehensive strengths and weaknesses analysis")
print("   Total visualizations created: 8 plots")
print("   • Tree structure diagram")
print("   • Impurity measures comparison") 
print("   • Information gain comparison")
print("   • Tree building process")
print("   • Trained decision tree")
print("   • Feature importance")
print("   • Decision boundaries")
print("   • Dental applications overview")
print("   • Strengths vs weaknesses")
print("\n   Perfect for sharing step-by-step on social media! 🚀")
print("-" * 70) 