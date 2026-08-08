# Scaled Dot-Product Attention

Scaled Dot-Product Attention is the attention mechanism used in the Transformer architecture.

The main idea is:

> **For each word, find which other words are important, give them weights, and combine their information.**

---

## 1. Query, Key, and Value

Attention uses three things:

- **Query (Q)** → "What am I looking for?"
- **Key (K)** → "What information do I contain?"
- **Value (V)** → "What information should I give you?"

### Simple Analogy

Think about a library:

| Attention | Library Analogy |
|---|---|
| Query | Your search question |
| Key | Labels/indexes on books |
| Value | Actual book information |

For example:

> **"The cat drank the milk because it was thirsty."**

When processing **"it"**, attention helps the model determine that **"cat"** is important.

---

# 2. Calculate Similarity Using Q × Kᵀ

The first step is:

$$
QK^T
$$

This calculates how strongly each Query matches each Key.

We use the **dot product** to calculate this similarity.

### Example

Suppose:

$$
Q = [1,2]
$$

and:

$$
K = [2,3]
$$

The dot product is:

$$
Q \cdot K = (1)(2)+(2)(3)
$$

$$
=2+6=8
$$

So the similarity score is:

$$
8
$$

### Important

> **Larger score = more similarity/relevance**

---

# 3. Why Do We Divide by √dₖ?

This is the most important part of Scaled Dot-Product Attention.

The formula contains:

$$
\frac{QK^T}{\sqrt{d_k}}
$$

where:

$$
d_k = \text{dimension of the Key vectors}
$$

## The Problem

When the vectors become large, their dot products can also become very large.

For example:

$$
[2,15,80,120]
$$

If we directly pass these values into Softmax:

$$
softmax([2,15,80,120])
$$

the output can become extremely extreme:

$$
[0,0,0,1]
$$

Almost all the attention goes to one value.

This is bad because the Softmax function enters a region where its gradients become **very small**.

Very small gradients make learning difficult.

---

# 4. Scaling Solves the Problem

Suppose:

$$
d_k=64
$$

Then:

$$
\sqrt{d_k}=\sqrt{64}=8
$$

Instead of:

$$
[2,15,80,120]
$$

we divide by 8:

$$
[0.25,1.875,10,15]
$$

This keeps the values at a more reasonable scale.

### Simple idea

> **We divide by √dₖ to prevent attention scores from becoming too large.**

---

# 5. Apply Softmax

After calculating and scaling the scores, we apply:

$$
softmax
$$

Softmax converts the scores into **attention weights**.

For example:

$$
[2,1,0]
$$

might become approximately:

$$
[0.67,0.24,0.09]
$$

This means:

- Word 1 → **67% attention**
- Word 2 → **24% attention**
- Word 3 → **9% attention**

The weights tell the model **how much attention to give each word**.

---

# 6. Multiply by V

Now we use the attention weights to combine the Values.

Suppose:

$$
Attention\ Weights=[0.7,0.2,0.1]
$$

and:

$$
V_1=[10,5]
$$

$$
V_2=[2,4]
$$

$$
V_3=[8,1]
$$

The output becomes:

$$
Output = 0.7V_1+0.2V_2+0.1V_3
$$

So:

$$
Output = 0.7[10,5]+0.2[2,4]+0.1[8,1]
$$

The result is a **weighted combination of the Values**.

The more important a word is, the larger its contribution to the final output.

---

# 7. Complete Formula

The complete Scaled Dot-Product Attention formula is:

$$
\boxed{
Attention(Q,K,V)
= softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V}
$$
It can be understood as four steps:

```text
        Q × Kᵀ
           ↓
   Calculate similarity
           ↓
      Divide by √dₖ
           ↓
        Softmax
           ↓
    Attention weights
           ↓
          × V
           ↓
      Final output