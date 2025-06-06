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
├── chapter_2_foundations/      # Scripts for Chapter 2 concepts
│   ├── data_structures_demo.py
│   ├── notation_examples.py
│   ├── sigma_notation_sse.py
│   ├── pi_notation_joint_prob.py
│   ├── set_operations_demo.py
│   ├── vector_operations_demo.py
│   ├── max_argmax_demo.py
│   ├── assignment_operator_demo.py
│   ├── functions_demo.py
│   ├── gradient_derivative_demo.py
│   ├── random_variables_demo.py
│   ├── estimators_bias_demo.py
│   ├── bayes_rule_demo.py
│   ├── parameter_estimation_demo.py
│   ├── model_vs_instance_based_demo.py
│   ├── shallow_vs_deep_demo.py
│   ├── params_vs_hyperparams.py
│   └── classification_vs_regression.py
├── chapter_3_fundamental_algorithms/ # Scripts for Chapter 3 concepts
│   └── linear_regression_intro_demo.py # Introduction to Linear Regression
├── .gitignore                   # Files ignored by git
├── best_rbf_params.json         # Best hyperparameters found by GridSearchCV for RBF SVM
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Foundational Concepts (Chapter 2: Notation and Definitions)

Before diving deeper into algorithms, we covered essential mathematical and statistical foundations based on Chapter 2 of the book. These concepts provide the language and tools needed to understand machine learning principles.

Key topics explored:

*   **Data Structures (Scalars, Vectors, Matrices, Tensors):** Understanding how data is organized, from single numbers (0D) to multi-dimensional arrays (ND). See `chapter_2_foundations/data_structures_demo.py` for visualizations.
*   **Notation & Operations:** Understanding symbols like Σ (summation), Π (product), set operations (∈, ∉, ∪, ∩, ⊆, ...), vector operations (addition, dot product `·`, Hadamard product `∘`, etc.), concepts like `max` and `arg max`, and the assignment operator (`:=` or `←` in pseudocode, `=` in Python) for iterative updates. See `chapter_2_foundations/notation_examples.py`, `chapter_2_foundations/sigma_notation_sse.py`, `chapter_2_foundations/pi_notation_joint_prob.py`, `chapter_2_foundations/set_operations_demo.py`, `chapter_2_foundations/vector_operations_demo.py`, `chapter_2_foundations/max_argmax_demo.py`, and `chapter_2_foundations/assignment_operator_demo.py` for practical examples and visualizations.
*   **Functions:** Understanding a function as a mapping from inputs (Domain) to a single output (in the Codomain). Key in ML for models, activation functions (e.g., Sigmoid), and loss functions (e.g., Squared Error). See `chapter_2_foundations/functions_demo.py` for visualizations.
*   **Derivatives & Gradients:** The derivative measures the instantaneous rate of change of a function (1D), while the gradient generalizes this to multivariate functions (vector of partial derivatives). Both are foundational for optimization in ML (e.g., gradient descent). The new 'elongated bowl' example visually demonstrates how the gradient can be much larger in one direction than another, illustrating feature sensitivity and anisotropy. See `chapter_2_foundations/gradient_derivative_demo.py` for visualizations, geometric intuition, and dental examples.
*   **Random Variables:** Explores the difference between discrete (countable, e.g., number of successful implants) and continuous (measurable, e.g., ISQ values) random variables. Includes visualizations: lollipop plots for discrete, range/interval plots for continuous.  
    - **PMF (Probability Mass Function):** Assigns probabilities to each possible value of a discrete random variable. Illustrated with both dice and dental implant success examples.  
    - **PDF (Probability Density Function):** Describes the likelihood of a continuous random variable falling within an interval. Demonstrated with ISQ values modeled as a normal distribution, including shaded area plots for probability intervals and a special plot showing how the PDF can exceed 1 for narrow distributions (clarifying a common confusion).
    - **Expected Value (Mean, \(E[X]\)):** Defines the long-run average or "center of mass" of a random variable. Calculated as \(\sum x \cdot P(X=x)\) for discrete RVs and \(\int x \cdot f(x) dx\) for continuous RVs. Illustrated with dedicated plots showing \(E[X]\) on PMFs (die roll, implant success) and PDFs (ISQ values), reinforcing its concept as the balancing point. Key confusion points about its calculation for continuous RVs are also discussed.
    - **Variance (\(\sigma^2\)) and Standard Deviation (\(\sigma\)):** Measure the spread or dispersion of a random variable. Variance is the average of the squared deviations from the mean (\(E[(X - E[X])^2]\)), providing a sense of variability in squared units. Standard Deviation, its square root (\(\sigma = \sqrt{Var(X)}\)), brings this measure back to the original units of the data, making it more interpretable. These statistics are crucial for understanding how tightly data clusters around the mean. Visualizations include the die roll and implant success PMFs showing \(E[X] \pm SD\), and the ISQ (normal distribution) PDF highlighting \(E[X]\) and shaded regions for \(\mu \pm 1\sigma\) and \(\mu \pm 2\sigma\), illustrating the Empirical Rule.
    - See `chapter_2_foundations/random_variables_demo.py` for code, explanations, and all visualizations pertaining to Discrete/Continuous RVs, PMF, PDF, Expected Value, Variance, and Standard Deviation.
*   **Key Statistics:** Defining Expected Value (Mean μ - center of distribution), Variance (σ² - spread in squared units), and Standard Deviation (σ - spread in original units). These summarize distributions.
*   **Estimators & Bias:** Understanding how sample statistics (e.g., sample mean \(\bar{x}\)) are used to estimate unknown true population parameters (e.g., population mean \(\mu\)). Explores the concept of bias (systematic over/underestimation) and distinguishes between biased and unbiased estimators. Key illustration includes why sample mean \(\bar{x}\) is an unbiased estimator for \(\mu\), and a detailed look at sample variance: demonstrating why using a denominator of \(n\) leads to a biased estimator for population variance \(\sigma^2\), while Bessel's correction (using \(n-1\) in the denominator) yields an unbiased estimator \(s^2\). The script `chapter_2_foundations/estimators_bias_demo.py` provides a simulation and visualizations to clearly show these concepts in action, including histograms of estimates and their convergence (or lack thereof) to true parameters.
*   **Bayes' Rule:** A fundamental theorem for updating probabilities based on new evidence: \( P(H|E) = \\frac{P(E|H)P(H)}{P(E)} \). Crucial for probabilistic reasoning and algorithms like Naive Bayes. See `chapter_2_foundations/bayes_rule_demo.py`.
*   **Parameter Estimation:** Methods for determining the unknown parameters of a model or distribution from data, including:
    - **Maximum Likelihood Estimation (MLE):** Finds parameter values that maximize the likelihood of observed data.
    - **Method of Moments (MoM):** Matches sample moments (like mean, variance) to population moments.
    - **Bayesian Estimation:** Updates prior beliefs about parameters with new data to form a posterior.
    - **Properties of estimators:** Unbiasedness, efficiency, and consistency.
    - **Dental application:** Estimating implant success rates using MLE and Bayesian methods, with visualizations of prior/posterior and confidence intervals.
    - See `chapter_2_foundations/parameter_estimation_demo.py` for code, explanations, and visualizations.
*   **Parameters vs. Hyperparameters:** A critical distinction! **Hyperparameters** (e.g., SVM's `C`, `gamma`, `kernel`) are chosen *before* training to configure the algorithm. **Parameters** (e.g., SVM's learned boundary definition) are learned *during* training from the data. See `chapter_2_foundations/params_vs_hyperparams.py` for a visual explanation.
*   **Classification vs. Regression:** Two main supervised learning tasks. Classification predicts categories (e.g., Success/Failure), while Regression predicts continuous values (e.g., final ISQ score). See `chapter_2_foundations/classification_vs_regression.py` for visualizations.
*   **Model-Based vs. Instance-Based Learning:** Distinguishes two main learning paradigms. 
    - **Model-Based Learning** (e.g., Logistic Regression, SVM, Decision Trees) aims to build an explicit model from the training data by learning its parameters. Predictions are made by feeding new data into this learned model.
    - **Instance-Based Learning** (e.g., k-Nearest Neighbors) memorizes training instances and makes predictions for new data by comparing it to these stored instances, often using a similarity or distance measure. 
    - The script `chapter_2_foundations/model_vs_instance_based_demo.py` visually contrasts these approaches using Logistic Regression and k-NN on a synthetic dataset, showing their different decision boundaries and discussing conceptual dental applications.
*   **Shallow vs. Deep Learning:** Differentiates learning algorithms based on model complexity and feature learning. 
    - **Shallow Learning** (e.g., Logistic Regression, SVM, Decision Trees) typically involves simpler architectures with fewer layers, often relying on hand-crafted features. They are generally more interpretable and less computationally demanding.
    - **Deep Learning** (e.g., Multi-Layer Perceptrons with many layers, CNNs, RNNs) uses deep architectures with multiple hidden layers to automatically learn hierarchical feature representations from raw data. They can model highly complex patterns but require large datasets and more computation.
    - The script `chapter_2_foundations/shallow_vs_deep_demo.py` conceptually demonstrates this by comparing Logistic Regression (shallow) with a simple MLP (representing deep) on a non-linearly separable dataset, visualizing their differing abilities to capture complex decision boundaries and includes conceptual dental examples for caries risk (shallow) vs. radiographic analysis (deep).
*   **Supervised, Unsupervised, and Reinforcement Learning:** The three main categories of machine learning problems (placeholder for future content).

## Fundamental Algorithms (Chapter 3)

This section will cover fundamental machine learning algorithms, starting with Linear Regression.

*   **Linear Regression:** A foundational algorithm for predicting a continuous numerical value from input features. It aims to find the best linear relationship between inputs and the output by minimizing the sum of squared errors.
    - Key concepts include the model equation (Y = wX + b), the Mean Squared Error (MSE) loss function, and the process of learning weights (w) and bias (b) from data. The `linear_regression_intro_demo.py` script also conceptually demonstrates prediction using these learned parameters.
    - The `chapter_3_fundamental_algorithms/linear_regression_intro_demo.py` script introduces these core concepts with extensive print explanations and multiple visualizations. These include an illustrative 1D example (Study Hours vs. Exam Score), and a detailed dental implant example with mock data. For the dental example, plots cover individual features (HU, Torque, BIC) vs. mock actual scores, predicted vs. actual scores, and a feature contribution bar chart for a single patient. These visuals are designed to clearly explain the model's components and the importance of aspects like feature scaling for weight interpretation.
    - For more detailed information on the progress and content of Chapter 3, including discussions on model training, prediction with learned parameters, key assumptions (like linearity, homoscedasticity), and the strengths/weaknesses of linear regression, please see the dedicated `chapter_3_fundamental_algorithms/README.md`.

*   **Logistic Regression:** A key algorithm for binary classification problems, used to predict the probability of an input belonging to one of two classes.
    - It uses a linear combination of inputs (like linear regression) to calculate a score (z-score), then applies the Sigmoid function to transform this score into a probability (0 to 1).
    - The Binary Cross-Entropy (BCE) loss function is typically used for training.
    - The initial concepts, including the z-score, Sigmoid function, BCE loss, and a dental example for implant success/failure classification, are introduced in `chapter_3_fundamental_algorithms/logistic_regression_intro_demo.py`.
    - Further details are also available in `chapter_3_fundamental_algorithms/README.md`.

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