# 100_page_machine_learning: Learning ML with Andriy Burkov's Book

This repository documents my journey learning machine learning concepts by implementing examples based on **"The Hundred-Page Machine Learning Book" by Andriy Burkov**.

The goal is to solidify understanding through practical coding exercises, initially focusing on core supervised learning algorithms like Support Vector Machines (SVM).

## Project Structure

```
.
├── 100_page_machine_learning/  # Python Virtual Environment (ignored by git)
├── supervised_learning/
│   └── support_vector_machine/
│       ├── baseline_linear_svm.py            # Baseline Linear SVM: Data Gen, Split, Scale, Train, Evaluate
│       ├── implant_svm_hyperparameters.py    # Visualization: Effect of C & Gamma on RBF SVM Boundary
│       └── svm_kernel_trick_example.py       # Demo: Linear vs RBF Kernel for non-linear data (Kernel Trick)
├── .gitignore                   # Files ignored by git
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Key Learnings & Examples (from SVM Section - Chapter 4)

This project currently explores **Support Vector Machines (SVM)**, a powerful supervised learning algorithm for classification.

### Core SVM Concept

*   **Goal:** Find the optimal hyperplane (line/plane/surface) that best separates data points belonging to different classes.
*   **Margin:** SVM aims to maximize the margin – the distance between the hyperplane and the nearest data points (support vectors) from each class. A wider margin generally leads to better generalization.
*   **Support Vectors:** These are the critical data points lying closest to the hyperplane, which *define* the margin.

### Hyperparameters & Non-Linearity

SVM handles non-linearly separable data using:

1.  **Soft Margin (Penalty Parameter `C`):** Controls the trade-off between maximizing margin width and minimizing classification errors on training data. Low `C` = wider margin, more error tolerant (risk: underfitting); High `C` = narrower margin, less tolerant (risk: overfitting).
2.  **Kernel Trick:** Allows SVMs to operate in a high-dimensional feature space without explicit computation, enabling non-linear boundaries. The `linear` kernel uses the original space. The `rbf` kernel is a powerful default for non-linear data.
3.  **`gamma` (for RBF kernel):** Controls the influence of individual data points. Low `gamma` = broad influence (smoother boundary, risk: underfitting); High `gamma` = localized influence (complex boundary, risk: overfitting).

### Machine Learning Best Practices

Building reliable models requires a solid process:

1.  **Feature Scaling (`StandardScaler`):** Essential for SVMs.
2.  **Train-Test Split (`train_test_split`):** Evaluate generalization.
3.  **Pipelines (`make_pipeline`, `Pipeline`):** Streamline workflows, prevent data leakage.
4.  **Baseline Model:** Establish performance before tuning.
5.  **Hyperparameter Tuning (`GridSearchCV`):** Systematically find optimal `C`, `gamma` etc. using cross-validation.
6.  **Evaluation Metrics (`classification_report`, etc.):** Use metrics beyond accuracy (Precision, Recall, F1).

### Code Examples Summary

*   `baseline_linear_svm.py`: Implements a basic **linear SVM pipeline** on simulated data (make_blobs). Follows best practices: scaling, train/test split, evaluation with classification report & confusion matrix. Establishes baseline performance.
*   `implant_svm_hyperparameters.py`: **Visualizes the effect of `C` and `Gamma`** on the RBF SVM decision boundary using a 2x2 plot. Helps build intuition about underfitting/overfitting trade-offs. Runs as a standalone demo.
*   `svm_kernel_trick_example.py`: Focuses on demonstrating the **Kernel Trick**. Uses the `make_moons` dataset to visually contrast the failure of a `linear` kernel vs the success of an `rbf` kernel on non-linear data. Includes `GridSearchCV` for the RBF example.
*   `supervised_vector_learning_example.py`: *Deprecated.* Initial basic visualization.

## Next Steps (D-1 and beyond based on feedback)

*   Implement `GridSearchCV` in a dedicated script (e.g., `grid_search_rbf.py`) based on the `baseline_linear_svm.py` structure.
*   Save best parameters found by `GridSearchCV`.
*   Log experiment results systematically (e.g., to a CSV).
*   Replace synthetic data with real (anonymized) implant data.
*   Implement `argparse` for script parameters.
*   Analyze support vectors from real data for clinical insights.
*   Continue through other chapters of the book.

## Special Thanks 🙏

A huge thank you to **Andriy Burkov** ([@aburkov](https://github.com/aburkov)) for writing incredibly clear and concise books that make complex topics accessible. This repository heavily relies on the excellent explanations in **"The Hundred-Page Machine Learning Book"**, and I also appreciate his work on **"The Hundred-Page Language Model Book"**. These resources are invaluable for practical learning! 