# Attention Mechanism

## What is Attention?

The **Attention Mechanism** is a technique used in Sequence-to-Sequence (Seq2Seq) models that allows the decoder to **focus on the most relevant parts of the input sentence** while generating each output word.

Instead of relying on a single fixed context vector, the decoder dynamically decides which input words are most important at every decoding step.

---

## Why Do We Need Attention?

In the original Encoder-Decoder architecture:

- The encoder reads the entire input sequence.
- It compresses all information into a single **context vector**.
- The decoder uses only this vector to generate the entire output.

This works well for short sentences but struggles with long ones because one vector cannot store every detail.

### Example

**Input Sentence**

```
The student who was sitting near the window finished the assignment before the deadline.
```

Compressing all this information into one vector causes information loss.

The Attention Mechanism solves this problem by allowing the decoder to access **all encoder hidden states** instead of only the final one.

---

# How Attention Works

Suppose the input sentence is:

```
I love machine learning
```

The encoder generates a hidden state for each input word.

```
Word:        I      love    machine   learning
Hidden:      h1      h2        h3         h4
```

Instead of keeping only the last hidden state, the model keeps all of them:

```
h1, h2, h3, h4
```

---

## Step 1: Compute Attention Scores

When the decoder wants to generate the next word, it compares its current hidden state with every encoder hidden state.

```
Decoder State
      │
      ▼
Compare with:
h1
h2
h3
h4
```

This produces an attention score for each encoder hidden state.

Example:

```
Score(h1) = 1.2
Score(h2) = 0.8
Score(h3) = 3.4
Score(h4) = 0.5
```

A higher score means that hidden state is more important for predicting the next output word.

---

## Step 2: Apply Softmax

The scores are converted into probabilities using the Softmax function.

Example:

```
Scores

h1 = 1.2
h2 = 0.8
h3 = 3.4
h4 = 0.5
```

After Softmax:

```
α1 = 0.10
α2 = 0.07
α3 = 0.76
α4 = 0.07
```

These values are called **attention weights**.

The weights always sum to 1.

```
0.10 + 0.07 + 0.76 + 0.07 = 1
```

The model will focus mostly on **h3**.

---

## Step 3: Create the Context Vector

The context vector is computed as a weighted sum of all encoder hidden states.

\[
c = \sum_i \alpha_i h_i
\]

Example:

```
Context Vector

= 0.10 × h1
+ 0.07 × h2
+ 0.76 × h3
+ 0.07 × h4
```

Since **h3** has the highest weight, it contributes the most.

---

## Step 4: Predict the Next Word

The decoder combines:

- Previous decoder hidden state
- Context vector
- Previous output word

to predict the next word.

```
Previous Output
        │
        ▼
      Decoder
        ▲
        │
 Context Vector
```

After generating one word, the decoder repeats the same process for the next word.

---

# Example

Input:

```
I eat apples
```

Encoder:

```
I -------- h1
eat ------ h2
apples --- h3
```

### Predicting "Je"

Attention weights:

```
h1 = 0.75
h2 = 0.15
h3 = 0.10
```

The decoder mainly focuses on **"I"**.

---

### Predicting "mange"

Attention weights:

```
h1 = 0.10
h2 = 0.80
h3 = 0.10
```

Now it focuses on **"eat"**.

---

### Predicting "des pommes"

Attention weights:

```
h1 = 0.05
h2 = 0.10
h3 = 0.85
```

Now it focuses on **"apples"**.

Notice that the attention shifts to different input words as each output word is generated.

---

# Advantages of Attention

- Solves the fixed-length context vector problem.
- Works much better for long sentences.
- Allows the decoder to focus on the most relevant input words.
- Improves translation quality and many other NLP tasks.
- Makes the model more interpretable because attention weights show where the model is focusing.

---

# Limitations

- Requires additional computations at every decoding step.
- Uses more memory than the original Encoder-Decoder model.
- RNN-based attention models are still sequential and cannot be fully parallelized.

---

# Attention Workflow

```
Input Sentence
      │
      ▼
Encoder
      │
      ▼
Hidden States
h1   h2   h3   h4
 │    │    │    │
 └────┴────┴────┘
        │
        ▼
Compute Attention Scores
        │
        ▼
Softmax
        │
        ▼
Attention Weights (α)
        │
        ▼
Weighted Sum
(Context Vector)
        │
        ▼
Decoder
        │
        ▼
Next Output Word
```

---

# Key Idea

> **The Attention Mechanism allows the decoder to dynamically focus on the most relevant encoder hidden states while generating each output word, instead of relying on a single fixed context vector.**
