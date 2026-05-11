# Perceptron

## What is a Perceptron?

A **perceptron** is the simplest type of artificial neuron used for binary classification. It takes a vector of inputs, computes a weighted sum plus a bias, and applies an activation function (typically a step function) to produce a binary output (0 or 1).

## Mathematical Formulation

Given inputs $x_1, x_2, \dots, x_n$, weights $w_1, w_2, \dots, w_n$, and bias $b$, the perceptron computes:

$$z = \sum_{i=1}^n w_i x_i + b$$

The output $y$ is produced using a step activation function:

$$y = f(z) = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{otherwise} \end{cases}$$

## Perceptron Learning Rule

The perceptron learns by updating weights whenever it makes a prediction error. For a training example $(\mathbf{x}, y)$ with target label $y \in \{0, 1\}$ and prediction $\hat{y}$:

$$\Delta w_i = \eta (y - \hat{y}) x_i$$
$$\Delta b = \eta (y - \hat{y})$$

where $\eta$ is the learning rate (typically 0.01 to 0.1).

**Training Algorithm:**
1. Initialize weights and bias (usually to zero or small random values)
2. For each epoch:
   - For each training example:
     - Compute prediction $\hat{y}$
     - Update weights and bias using the learning rule above
3. Repeat until convergence or max epochs reached

The perceptron algorithm **guarantees convergence** if the dataset is linearly separable.

## Example: AND Gate

The AND function can be learned by the perceptron. One solution:
- $w_1 = 1$, $w_2 = 1$, $b = -1.5$

With these weights:
- AND(0, 0) = 0: $z = 1(0) + 1(0) - 1.5 = -1.5 < 0$ → output = 0 ✓
- AND(0, 1) = 0: $z = 1(0) + 1(1) - 1.5 = -0.5 < 0$ → output = 0 ✓
- AND(1, 0) = 0: $z = 1(1) + 1(0) - 1.5 = -0.5 < 0$ → output = 0 ✓
- AND(1, 1) = 1: $z = 1(1) + 1(1) - 1.5 = 0.5 \geq 0$ → output = 1 ✓

---

## Limitations

- **Linearly Separable Only**: Cannot solve non-linearly separable problems like XOR.
- **Hard Threshold**: The step function is non-differentiable, preventing use of gradient-based optimization (unlike MLPs).
- **Binary Classification**: Limited to two classes; multi-class requires one-vs-rest or other strategies.

---