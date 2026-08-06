# Self-Attention Mechanism

Self-Attention allows every word in a sentence to look at **all other words**, including itself, to determine which words are most important for understanding its meaning.

For the sentence:

> **"money bank grows"**

each word generates a new representation by attending to every word in the sentence.

---

## Step 1: Input Embeddings

Each word is first converted into a vector (embedding).

```
money  → e_money
bank   → e_bank
grows  → e_grows
```

These embeddings contain the semantic meaning of each word.

---

## Step 2: Compute Similarity Scores

Suppose we want to compute the new representation of **"money"**.

The embedding of **money** is compared with every word in the sentence.

```
money ↔ money
money ↔ bank
money ↔ grows
```

These comparisons produce similarity scores.

```
s11 = similarity(money, money)
s12 = similarity(money, bank)
s13 = similarity(money, grows)
```

The larger the score, the more important that word is to **money**.

---

## Step 3: Apply Softmax

The similarity scores are converted into probabilities using the Softmax function.

```
Softmax(s11, s12, s13)

↓

w11
w12
w13
```

The weights satisfy:

```
w11 + w12 + w13 = 1
```

These are called **attention weights**.

---

## Step 4: Weighted Sum

Each embedding is multiplied by its attention weight.

```
y_money =
w11 × e_money
+ w12 × e_bank
+ w13 × e_grows
```

The result is the **new embedding** of **money**.

---

## Step 5: Repeat for Every Word

The same process is repeated for every word.

### For **bank**

Compute similarity scores:

```
s21 = similarity(bank, money)
s22 = similarity(bank, bank)
s23 = similarity(bank, grows)
```

Apply Softmax:

```
(s21, s22, s23)
        ↓
(w21, w22, w23)
```

Compute weighted sum:

```
y_bank =
w21 × e_money
+ w22 × e_bank
+ w23 × e_grows
```

---

### For **grows**

Compute similarity scores:

```
s31 = similarity(grows, money)
s32 = similarity(grows, bank)
s33 = similarity(grows, grows)
```

Apply Softmax:

```
(s31, s32, s33)
        ↓
(w31, w32, w33)
```

Compute weighted sum:

```
y_grows =
w31 × e_money
+ w32 × e_bank
+ w33 × e_grows
```

---

# Complete Flow

```text
Sentence
    │
    ▼
Word Embeddings
    │
    ▼
Compute Similarity Scores
    │
    ▼
Softmax
    │
    ▼
Attention Weights
    │
    ▼
Weighted Sum of Embeddings
    │
    ▼
New Context-Aware Embeddings
```

---

# Intuition

Imagine you're reading the sentence:

> **"Money grows in the bank."**

When understanding the word **bank**, you don't look at only **bank** itself.

Instead, you also consider:

- money
- grows
- bank

Each contributes differently.

For example:

| Word | Attention Weight |
|------|-----------------:|
| money | 0.60 |
| bank | 0.30 |
| grows | 0.10 |

The new representation becomes:

```
0.60 × e_money
+ 0.30 × e_bank
+ 0.10 × e_grows
```

This makes the embedding of **bank** aware of its surrounding context.

---

# Why Self-Attention Works

Instead of treating every word independently, Self-Attention allows each word to gather information from all other words in the sentence.

As a result:

- Every word becomes **context-aware**.
- Important words receive higher attention weights.
- Less relevant words receive lower attention weights.
- Long-range relationships between words are captured effectively.

---

# Final Output

Original embeddings:

```
e_money
e_bank
e_grows
```

become context-aware embeddings:

```
y_money
y_bank
y_grows
```

These updated embeddings are then passed to the **Feed Forward Neural Network** inside the Transformer encoder.