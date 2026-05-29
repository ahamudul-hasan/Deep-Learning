# Backpropagation Explained in Simple Way

Backpropagation is the process a neural network uses to **learn from mistakes**.

Think of it like this:

1. The network makes a prediction.
2. It checks how wrong the prediction is.
3. It sends the error backward through the network.
4. It adjusts weights to reduce future mistakes.

---

# Step-by-Step Backpropagation

Suppose we want a neural network to recognize handwritten digits from the MNIST dataset.

---

# Step 1: Input Goes Into Network (Forward Pass)

Example image → network input.

The data moves layer by layer:

Input → Hidden Layer → Output Layer

Each neuron does:

$$z = \sum_{i=1}^{n} (input_i \times weight_i) + bias$$

Or more simply:

$$z = w \cdot x + b$$

Then activation function is applied.

Example:

$$a = ReLU(z)$$

The network finally gives a prediction.

Example:

Prediction = 3  
Actual = 5

---

# Step 2: Calculate Error (Loss)

Now the network checks:

“How wrong was I?”

This is done using a **loss function**.

Example:

$$Loss = Actual - Predicted$$

Real neural networks use functions like:

- Mean Squared Error (MSE)
- Cross Entropy

If prediction is very wrong → loss becomes large.

---

# Step 3: Send Error Backward

This is the “back” in backpropagation.

The network starts from the output layer and moves backward.

It calculates:

“Which weights caused the error?”

Using calculus (derivatives/gradients), it finds how much each weight contributed to the mistake.

- Large contribution → larger correction  
- Small contribution → smaller correction  

---

# Step 4: Calculate Gradients

Gradient means:

“Which direction should the weight move to reduce error?”

- Positive gradient → decrease weight  
- Negative gradient → increase weight  

The network computes gradients for every weight.

---

# Step 5: Update Weights

Weights are updated using:

$$w_{new} = w_{old} - \eta \cdot \frac{\partial L}{\partial w}$$

Where:

- w = weight  
- η = learning rate  
- L = loss  

Simple idea:

new_weight = old_weight - correction

This helps the network make better predictions next time.

---

# Step 6: Repeat Many Times

The process repeats for many epochs:

Forward Pass  
→ Calculate Loss  
→ Backpropagation  
→ Update Weights  
→ Repeat  

Over time:

- loss decreases  
- accuracy increases  
- predictions improve  

---

# Real-Life Analogy

Imagine learning basketball:

1. You throw the ball  
2. You miss the basket  
3. Your brain checks the error  
4. You adjust hand force and angle  
5. You try again better  

That adjustment process is like backpropagation.

---

# Tiny Example

Prediction = 0.2  
Actual = 1  

Loss = 0.8  

Backpropagation tells the network:

- Increase some weights  
- Decrease others  

After updating:

New Prediction = 0.7  

Closer to correct answer.

---

# In One Line

Backpropagation is:

The method neural networks use to send errors backward and adjust weights to learn better predictions.

---