# Logistic Regression from Scratch
Implementation of Logistic Regression with manually derived gradients and cross-entropy loss, compared against Scikit-Learn's implementation for validation purposes.

---

## Mathematical Foundation
### Model Definition

Logistic Regression models the probability of a binary outcome:

$$
z = w^T x + b
$$

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

The sigmoid function maps any real-valued input to the range (0, 1), making it suitable for binary classification.

---

### Cross-Entropy Loss
$$
L = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i) \right]
$$

Where:
- $m$ is the number of samples  
- $y$ is the true label  
- $\hat{y}$ is the predicted probability  

---

### Gradient Derivation
Using the chain rule, the gradients are:

$$
\frac{\partial L}{\partial w} = \frac{1}{m} X^T(\hat{y} - y)
$$

$$
\frac{\partial L}{\partial b} = \frac{1}{m} \sum (\hat{y} - y)
$$

These gradients are used in gradient descent:

$$
w := w - \alpha \frac{\partial L}{\partial w}
$$

$$
b := b - \alpha \frac{\partial L}{\partial b}
$$

where $\alpha$ is the learning rate.

---

## Implementation Details
- Optimization: Gradient Descent  
- Learning Rate: 0.1 (tunable)  
- Iterations: 1000  
- Activation Function: Sigmoid  
- Loss Function: Cross-Entropy  

---

## Comparison with Scikit-Learn
The custom implementation is evaluated against Scikit-Learn's `LogisticRegression` to validate correctness. Both models achieve comparable performance on the test dataset.

---

## Results
- Custom Model Accuracy: ~90%  
- Scikit-Learn Accuracy: ~90%  

---

## Visualizations
1. Loss curve showing convergence during training  
2. Decision boundary illustrating class separation  

---

## How to Run
```bash
python logistic_regression.py
