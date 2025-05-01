# 100_page_machine_learning: Learning ML with Andriy Burkov's Book

This repository documents my journey learning machine learning concepts by implementing examples based on **"The Hundred-Page Machine Learning Book" by Andriy Burkov**.

The goal is to solidify understanding through practical coding exercises, initially focusing on core supervised learning algorithms like Support Vector Machines (SVM).

## Project Structure

```
.
├── 100_page_machine_learning/  # Python Virtual Environment (ignored by git)
├── supervised_learning/
│   └── support_vector_machine/
│       ├── supervised_vector_learning_example.py # DEPRECATED: Basic SVM visualization (Wide vs Narrow Margin concept)
│       ├── implant_svm_hyperparameters.py      # Baseline SVM evaluation (Scaling, Split, Pipeline) & C/Gamma Visualization
│       └── svm_kernel_trick_example.py         # Demo: Linear vs RBF Kernel for non-linear data (Kernel Trick)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Key Learnings & Examples (from SVM Section - Chapter 4)

This project currently explores **Support Vector Machines (SVM)**, a powerful supervised learning algorithm for classification.

### Core SVM Concept (Illustrated initially in `supervised_vector_learning_example.py`)

*   **Goal:** Find the optimal hyperplane (line/plane/surface) that best separates data points belonging to different classes.
*   **Margin:** SVM aims to maximize the margin – the distance between the hyperplane and the nearest data points (support vectors) from each class. A wider margin generally leads to better generalization.
*   **Support Vectors:** These are the critical data points lying closest to the hyperplane, which *define* the margin. Other points further away don't influence the boundary.

### Hyperparameters & Non-Linearity (Illustrated in `implant_svm_hyperparameters.py` & `svm_kernel_trick_example.py`)

Burkov emphasizes that real-world data isn't always linearly separable. SVM handles this using:

1.  **Soft Margin (Penalty Parameter `C`):**
    *   Allows some misclassifications in exchange for a wider, potentially more generalizable margin.
    *   `C` controls this trade-off: Low `C` = wider margin, more tolerant of errors; High `C` = narrower margin, less tolerant (can overfit).
    *   *(Visualized in `implant_svm_hyperparameters.py`)*
2.  **Kernel Trick:**
    *   The core idea allowing SVMs to handle non-linear data efficiently.
    *   Instead of explicitly mapping data to a complex high-dimensional space (`phi(x)`), kernels (`K(x,z)`) compute the necessary dot products (`phi(x)·phi(z)`) directly from the original data.
    *   This makes using powerful non-linear mappings computationally feasible.
    *   *(Concept demonstrated visually in `svm_kernel_trick_example.py`)*
3.  **Common Kernels:**
    *   `linear`: For linearly separable data. Effectively `phi(x) = x`.
    *   `rbf` (Radial Basis Function): Excellent default for non-linear data. Maps to infinite dimensions. Creates smooth, potentially complex boundaries.
    *   `poly`: Polynomial kernel.
4.  **`gamma` (for RBF kernel):**
    *   Defines the influence ("reach" or "width") of a single training point (support vector). Low `gamma` = far reach (smoother boundary, risk of underfitting); High `gamma` = close reach (complex/wiggly boundary, risk of overfitting).
    *   *(Visualized in `implant_svm_hyperparameters.py`)*

### Machine Learning Best Practices (Implemented in `implant_svm_hyperparameters.py`)

Building reliable models requires a solid process:

1.  **Feature Scaling (`StandardScaler`):** Essential for SVMs as they are sensitive to feature ranges.
2.  **Train-Test Split (`train_test_split`):** Crucial for evaluating model generalization on unseen data.
3.  **Pipelines (`make_pipeline`, `Pipeline`):** Streamline workflows (e.g., scaling + model) and prevent data leakage.
4.  **Baseline Model:** Establish performance with a simple model before complex tuning.
5.  **Hyperparameter Tuning (`GridSearchCV`):** Systematically search for optimal parameters (like C, gamma) using cross-validation within the training set.
6.  **Evaluation Metrics (`classification_report`, `confusion_matrix`):** Use metrics beyond accuracy (Precision, Recall, F1) to understand model performance nuances.

### Code Examples Summary

*   `supervised_vector_learning_example.py`: *Initial basic visualization, now largely superseded.* Shows plotting support vectors and margins for simple linear cases.
*   `implant_svm_hyperparameters.py`: Demonstrates establishing a **baseline linear SVM** following best practices (scaling, split, pipeline, evaluation). Also includes detailed **visualizations of the C and Gamma parameters' effects** on the RBF kernel's decision boundary using simulated BIC/ISQ data.
*   `svm_kernel_trick_example.py`: Focuses specifically on the **Kernel Trick**. Uses the `make_moons` dataset to visually contrast the failure of a **linear kernel** vs the success of an **RBF kernel** (tuned with `GridSearchCV`) on clearly non-linear data, demonstrating *why* kernels are necessary.

## Next Steps

*   Implement `GridSearchCV` within the `implant_svm_hyperparameters.py` script to find optimal C/gamma for that simulated data.
*   Explore other kernels (e.g., `poly`).
*   Apply concepts to real (anonymized) dental implant data when available.
*   Continue through other chapters of the book. 