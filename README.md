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
│       ├── grid_search_rbf.py                # RBF SVM: GridSearchCV Tuning (C, gamma), Evaluation, Param Saving
│       ├── implant_svm_hyperparameters.py    # Visualization: Effect of C & Gamma on RBF SVM Boundary
│       └── svm_kernel_trick_example.py       # Demo: Linear vs RBF Kernel for non-linear data (Kernel Trick)
├── .gitignore                   # Files ignored by git
├── best_rbf_params.json         # Best hyperparameters found by GridSearchCV for RBF SVM
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
*   `grid_search_rbf.py`: Implements **hyperparameter tuning for an RBF SVM** using `GridSearchCV` on the same simulated data. Finds optimal `C` and `gamma`, evaluates the tuned model, saves the best parameters, and visualizes the result. Completes D-1 task.
*   `implant_svm_hyperparameters.py`: **Visualizes the effect of `C` and `Gamma`** on the RBF SVM decision boundary using a 2x2 plot. Helps build intuition about underfitting/overfitting trade-offs. Runs as a standalone demo.
*   `svm_kernel_trick_example.py`: Focuses on demonstrating the **Kernel Trick**. Uses the `make_moons` dataset to visually contrast the failure of a `linear` kernel vs the success of an `rbf` kernel on non-linear data. Includes `GridSearchCV` for the RBF example.
*   `supervised_vector_learning_example.py`: *Deprecated.* Initial basic visualization.

## Next Steps (D-2 and beyond based on feedback)

*   Load real implant CSV, repeat baseline + grid search; export results row to `experiments.csv`.
*   Create margin-distance histogram.
*   Review support vectors from real data → map to patient IDs; write clinical insight.
*   Log experiment results systematically (e.g., to a CSV or using `mlflow`).
*   Implement `argparse` for script parameters.
*   Continue through other chapters of the book.

## Special Thanks 🙏

A huge thank you to **Andriy Burkov** ([@aburkov](https://github.com/aburkov)) for writing incredibly clear and concise books that make complex topics accessible. This repository heavily relies on the excellent explanations in **"The Hundred-Page Machine Learning Book"**, and I also appreciate his work on **"The Hundred-Page Language Model Book"**. These resources are invaluable for practical learning! 