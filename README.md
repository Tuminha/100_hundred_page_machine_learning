# 100_page_machine_learning: Learning ML with Andriy Burkov's Book

This repository documents my journey learning machine learning concepts by implementing examples based on **"The Hundred-Page Machine Learning Book" by Andriy Burkov**.

The goal is to solidify understanding through practical coding exercises, initially focusing on core supervised learning algorithms like Support Vector Machines (SVM).

## Project Structure

```
.
├── 100_page_machine_learning/  # Python Virtual Environment (ignored by git)
├── supervised_learning/
│   └── support_vector_machine/
│       ├── supervised_vector_learning_example.py # Basic SVM visualization (Wide vs Narrow Margin concept)
│       └── implant_svm_hyperparameters.py      # SVM hyperparameters (C, kernel, gamma) demo + Baseline evaluation
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Key Learnings & Examples (from SVM Section - Chapter 4)

This project currently explores **Support Vector Machines (SVM)**, a powerful supervised learning algorithm for classification.

### Core SVM Concept (Illustrated in `supervised_vector_learning_example.py`)

*   **Goal:** Find the optimal hyperplane (line/plane/surface) that best separates data points belonging to different classes.
*   **Margin:** SVM aims to maximize the margin – the distance between the hyperplane and the nearest data points (support vectors) from each class. A wider margin generally leads to better generalization.
*   **Support Vectors:** These are the critical data points lying closest to the hyperplane, which *define* the margin. Other points further away don't influence the boundary.

### Hyperparameters & Non-Linearity (Illustrated in `implant_svm_hyperparameters.py` initially)

Burkov emphasizes that real-world data isn't always linearly separable. SVM handles this using:

1.  **Soft Margin (Penalty Parameter `C`):**
    *   Allows some misclassifications in exchange for a wider, potentially more generalizable margin.
    *   `C` controls this trade-off: Low `C` = wider margin, more tolerant of errors; High `C` = narrower margin, less tolerant (can overfit).
2.  **Kernel Trick:**
    *   Implicitly maps data to a higher-dimensional space where linear separation might be possible.
    *   Common kernels:
        *   `linear`: For linearly separable data.
        *   `rbf` (Radial Basis Function): Good default for non-linear data. Controlled by `gamma`.
        *   `poly`: Polynomial kernel.
    *   **`gamma` (for RBF):** Defines the influence of a single training point. Low `gamma` = far reach (smoother boundary); High `gamma` = close reach (complex boundary, can overfit).

### Machine Learning Best Practices (Implemented in `implant_svm_hyperparameters.py`)

Beyond the algorithm itself, building reliable models requires a solid process:

1.  **Feature Scaling (`StandardScaler`):** Essential when features have different units or ranges (like BIC % and ISQ). SVM is sensitive to feature scales, and scaling prevents features with larger values from dominating.
2.  **Train-Test Split (`train_test_split`):** Crucial for evaluating model generalization. Train the model on one subset of data and test its performance on a separate, unseen subset.
3.  **Pipelines (`make_pipeline`):** Streamline workflows by chaining steps like scaling and model training. Prevents data leakage from the test set into the scaler fitted on the training set.
4.  **Baseline Model:** Establish a simple model's performance (e.g., Linear SVM with default C) before attempting complex tuning.
5.  **Evaluation Metrics (`classification_report`, `confusion_matrix`):** Go beyond accuracy. Use metrics like precision, recall, F1-score, and the confusion matrix to understand *how* the model performs, especially regarding different types of errors (e.g., false positives vs. false negatives in implant success/failure prediction).

## Next Steps

*   Implement hyperparameter tuning using `GridSearchCV`.
*   Explore different kernels (RBF).
*   Apply concepts to real (anonymized) dental implant data when available.
*   Continue through other chapters of the book. 