# Bahdanau Attention (Additive Attention)

## What is Bahdanau Attention?

**Bahdanau Attention**, also known as **Additive Attention**, is an attention mechanism introduced by **Dzmitry Bahdanau** in 2014.

It improves the original Encoder-Decoder architecture by allowing the decoder to **focus on different parts of the input sequence** while generating each output word.

Instead of using only one fixed context vector, the decoder computes a **new context vector at every decoding step**.

---

## Why Do We Need Bahdanau Attention?

In the original Seq2Seq model:

- The encoder compresses the entire input into a single context vector.
- The decoder relies only on this vector to generate all output words.

This becomes a problem for long sentences because a single vector cannot capture every detail.

Bahdanau Attention solves this problem by allowing the decoder to look at **all encoder hidden states** whenever it predicts a new word.

---

# How Bahdanau Attention Works

Suppose the input sentence is

```
I love machine learning
```

The encoder produces a hidden state for every word.

```
Word:        I      love    machine   learning
Hidden:      h1      h2        h3         h4
```

These hidden states are stored instead of keeping only the last one.

---

## Step 1: Compute Alignment Scores

When predicting the next word, the decoder compares its **previous hidden state** with every encoder hidden state.

Unlike basic attention, Bahdanau uses a **small feed-forward neural network** to calculate these scores.

The alignment score is

$$
e_{ij}=v^T\tanh(W_s s_{i-1}+W_h h_j)
$$

where


- \(s_{i-1}\) = previous decoder hidden state
- \(h_j\) = encoder hidden state
- \(W_s, W_h, v\) = learnable parameters

This neural network learns how important each encoder hidden state is.

---

## Step 2: Apply Softmax

The alignment scores are converted into attention weights.

Example

```
Scores

h1 = 1.5
h2 = 0.8
h3 = 3.7
h4 = 0.4
```

After Softmax

```
α1 = 0.11
α2 = 0.06
α3 = 0.77
α4 = 0.06
```

The weights sum to 1.

---

## Step 3: Compute Context Vector

The context vector is calculated as

$$
c_i=\sum_j \alpha_{ij}h_j
$$

Example

```
Context

= 0.11h1
+0.06h2
+0.77h3
+0.06h4
```

Since **h3** has the highest attention weight, it contributes the most.

---

## Step 4: Generate the Output

The decoder combines

- Previous decoder hidden state
- Context vector
- Previous output word

to predict the next output word.

The same process repeats for every output word.

---

# Example

Input

```
I eat apples
```

Encoder

```
I -------- h1
eat ------ h2
apples --- h3
```

### Predict "Je"

```
h1 = 0.75
h2 = 0.15
h3 = 0.10
```

Focus is on **I**.

---

### Predict "mange"

```
h1 = 0.10
h2 = 0.80
h3 = 0.10
```

Focus shifts to **eat**.

---

### Predict "des pommes"

```
h1 = 0.05
h2 = 0.10
h3 = 0.85
```

Focus shifts to **apples**.

---

# Advantages

- Solves the fixed context vector problem.
- Performs well on long sentences.
- Learns where to focus automatically.
- Produces better translations than the original Seq2Seq model.

---

# Limitations

- Requires extra computations because of the neural network.
- Slower than dot-product attention.
- More parameters to train.

---

# Bahdanau Attention Workflow

```
Encoder Hidden States
h1   h2   h3   h4
 │    │    │    │
 └────┴────┴────┘
        │
        ▼
Feed-Forward Neural Network
        │
        ▼
Alignment Scores
        │
        ▼
Softmax
        │
        ▼
Attention Weights
        │
        ▼
Context Vector
        │
        ▼
Decoder
        │
        ▼
Output Word
```

---

# Luong Attention (Multiplicative Attention)

## What is Luong Attention?

**Luong Attention**, also called **Multiplicative Attention** or **Dot-Product Attention**, was introduced by **Minh-Thang Luong** in 2015.

Like Bahdanau Attention, it allows the decoder to focus on different encoder hidden states.

The main difference is **how the attention scores are calculated**.

Instead of using a neural network, Luong Attention computes scores using **dot products**, making it faster and computationally cheaper.

---

## Why Do We Need Luong Attention?

Luong Attention addresses the same problem as Bahdanau Attention:

- A single context vector cannot represent long sentences well.
- The decoder should dynamically focus on different input words.

However, Luong proposed a simpler and more efficient way to compute attention.

---

# How Luong Attention Works

Suppose the encoder produces

```
Word:        I      love    machine   learning
Hidden:      h1      h2        h3         h4
```

---

## Step 1: Compute Attention Scores

The decoder compares its current hidden state with every encoder hidden state.

The simplest scoring function is

$$
score(s,h)=s^Th
$$

where

- \(s\) = decoder hidden state
- \(h\) = encoder hidden state

Luong also proposed two other scoring methods.

### Dot

$$
score=s^Th
$$

### General

$$
score=s^TWh
$$

### Concat

$$
score=v^T\tanh(W[s;h])
$$

The **Dot** method is the simplest and fastest.

---

## Step 2: Apply Softmax

Example

```
Scores

h1 = 0.9
h2 = 2.8
h3 = 1.1
h4 = 0.2
```

After Softmax

```
α1 = 0.10
α2 = 0.70
α3 = 0.15
α4 = 0.05
```

The decoder mainly focuses on **h2**.

---

## Step 3: Compute Context Vector

The context vector is

$$
c=\sum_i \alpha_i h_i
$$

Example

```
Context

=0.10h1
+0.70h2
+0.15h3
+0.05h4
```

---

## Step 4: Generate the Output

The decoder combines

- Current decoder hidden state
- Context vector

to predict the next output word.

The process repeats for every decoder step.

---

# Example

Input

```
I eat apples
```

Encoder

```
I -------- h1
eat ------ h2
apples --- h3
```

### Predict "Je"

```
h1 = 0.80
h2 = 0.10
h3 = 0.10
```

Focus is on **I**.

---

### Predict "mange"

```
h1 = 0.10
h2 = 0.80
h3 = 0.10
```

Focus shifts to **eat**.

---

### Predict "des pommes"

```
h1 = 0.05
h2 = 0.10
h3 = 0.85
```

Focus shifts to **apples**.

---

# Advantages

- Faster than Bahdanau Attention.
- Simpler implementation.
- Requires fewer computations.
- Performs well for many sequence-to-sequence tasks.

---

# Limitations

- Dot-product scores may be less expressive than a learned neural network.
- Performance may degrade when hidden vector dimensions become very large.

---

# Luong Attention Workflow

```
Encoder Hidden States
h1   h2   h3   h4
 │    │    │    │
 └────┴────┴────┘
        │
        ▼
Dot Product
(Current Decoder State × Encoder States)
        │
        ▼
Attention Scores
        │
        ▼
Softmax
        │
        ▼
Attention Weights
        │
        ▼
Context Vector
        │
        ▼
Decoder
        │
        ▼
Output Word
```

---
