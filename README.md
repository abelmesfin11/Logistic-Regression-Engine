# Logistic Regression Engine

Tech stack: Python, NumPy, Pandas

A from-scratch implementation of a logistic regression classifier using numerical optimization. The model is trained via custom gradient descent using numerical approximations (finite differences) rather than autograd or analytic gradients, which provides deeper insight into how gradient-based optimization works under the hood.

✨ Features:

Numerical Gradient Estimation: Approximated gradients for each weight using the finite difference method (∂L/∂θ ≈ [L(θ+h) - L(θ)] / h), which helps demystify how optimization algorithms function at a fundamental level.

L2 Regularization: Optional regularization added to control model complexity and reduce overfitting, especially useful for high-dimensional clinical datasets.

Custom Loss Function: Implemented the negative log-likelihood for binary classification (log loss), compatible with probabilistic outputs from the sigmoid function.

Sigmoid Activation: Used for probability output; prediction based on thresholding at 0.5.

Evaluation Metrics: Included both accuracy and mean negative log-likelihood for robust model assessment.

📊 Datasets:

Simulated Data: Validated correctness and parameter convergence of the model on synthetic binary classification data.

Breast Cancer Dataset: Used real-world medical data (from breast_cancer.csv) to classify malignant vs. benign tumors. Achieved over 90% accuracy on the test set.

🧪 Experiments:

Compared three models with different levels of L2 regularization (λ = 0, 0.01, 0.2) on a validation set to find the optimal bias-variance trade-off.

Visualized model performance by printing weight values and assessing which features contributed most to prediction.
