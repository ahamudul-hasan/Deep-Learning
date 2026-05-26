# How a Neuron Works in a Neural Network

A neuron in a neural network is like a tiny decision-making unit.

When data enters a neuron, it goes through several steps before producing an output.

---

# Step 1: Inputs Enter the Neuron

Suppose the neuron receives three inputs:

$$
x_1, x_2, x_3
$$

Example:

- $x_1 = 2$
- $x_2 = 5$
- $x_3 = 1$

These inputs can be:
- pixel values from an image,
- features like age or salary,
- or outputs from previous neurons.

---

# Step 2: Each Input Has a Weight

Every input connection has a weight:

$$
w_1, w_2, w_3
$$

Example:

- $w_1 = 0.5$
- $w_2 = -0.2$
- $w_3 = 0.8$

Weights determine how important each input is.

- Large positive weight → strong positive effect
- Negative weight → opposite effect
- Small weight → less important

---

# Step 3: Weighted Sum Calculation

The neuron multiplies each input by its weight and adds them together.

$$
z = x_1w_1 + x_2w_2 + x_3w_3 + b
$$

where:
- $b$ = bias

Example:

$$
z = (2)(0.5) + (5)(-0.2) + (1)(0.8) + 1
$$

$$
z = 1 - 1 + 0.8 + 1
$$

$$
z = 1.8
$$

This value represents the total signal entering the neuron.

---

# Step 4: Activation Function

The neuron now decides how strongly it should activate.

It applies an activation function to the weighted sum.

## ReLU Activation Function

$$
f(x) = \max(0, x)
$$

Example:

$$
f(1.8) = 1.8
$$

If the value is negative:

$$
f(-3) = 0
$$

ReLU removes negative values and keeps positive values.

---

# Common Activation Functions

1. ReLU
2. Sigmoid
3. Tanh
4. Softmax

---

# Step 5: Output Sent Forward

The final output of the neuron becomes input for neurons in the next layer.

Data flows like this:

$$
Input Layer \rightarrow Hidden Layer \rightarrow Output Layer
$$

---

# Real-Life Analogy

Imagine predicting whether a student will pass an exam.

Inputs:
- Study hours
- Attendance
- Sleep

Weights determine importance:
- Study hours may matter a lot
- Sleep may matter moderately
- Attendance may matter less

The neuron combines everything and predicts:

> "High chance of passing"

---

# What Happens During Training?

Initially:
- weights are random,
- predictions are inaccurate.

The neural network improves by:

1. Making predictions
2. Calculating error (loss)
3. Adjusting weights using backpropagation

Over time, neurons learn useful patterns from data.

---

# Mathematical Formula

The complete neuron equation:

$$
\text{Output} = f\left(\sum x_iw_i + b\right)
$$

where:
- \(x_i\) = inputs
- \(w_i\) = weights
- \(b\) = bias
- \(f\) = activation function

---

# Summary

A neuron performs 5 main tasks:

1. Receives inputs
2. Multiplies inputs by weights
3. Adds bias
4. Applies activation function
5. Sends output to the next layer

---