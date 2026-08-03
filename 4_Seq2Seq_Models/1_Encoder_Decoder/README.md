# Sequence to Sequence Learning with Neural Networks

**Paper:** [arXiv:1409.3215](https://arxiv.org/abs/1409.3215)

---

## Overview

This is the seminal 2014 paper that introduced the **encoder-decoder (seq2seq)** architecture using LSTMs, one of the most influential ideas in modern NLP/deep learning. It laid the groundwork for later developments including attention mechanisms and eventually the Transformer architecture.

---

## The Core Problem

Standard Deep Neural Networks (DNNs) excel at tasks with **fixed-size inputs and outputs** (e.g., image classification). But many important problems — translation, speech recognition, question answering — are inherently **sequential with variable, unknown lengths**. A sentence in English might be 5 or 50 words, and its translation won't match that length. Standard DNNs cannot naturally handle this.

---

## The Core Idea: Encoder-Decoder with LSTMs

1. **Encoder LSTM** — reads the input sequence one token at a time. After processing the whole sentence, its final hidden state becomes a single **fixed-dimensional vector** summarizing the sentence's meaning.
2. **Decoder LSTM** — a *separate* LSTM initialized with that vector as its starting hidden state, generating the output sequence one word at a time (feeding each prediction back in as the next input) until it emits a special `<EOS>` token.

Mathematically:

$$
P(y_1, y_2, \ldots, y_{T'} \mid x_1, x_2, \ldots, x_T)
=
\prod_{t=1}^{T'}
P(y_t \mid v, y_1, y_2, \ldots, y_{t-1})
$$

where `v` is the fixed-size encoder output vector, and each conditional is a softmax over the vocabulary.

### Three key design choices

- **Two separate LSTMs** (encoder + decoder), not one shared network — cheap in compute, and generalizes to multiple language pairs.
- **Depth** — 4-layer deep LSTMs significantly outperformed shallow ones (~10% perplexity reduction per extra layer).
- **Reversing the source sentence** — described below; one of the paper's most important findings.

---

## The "Reverse the Source Sentence" Trick

Instead of feeding the encoder the sentence in normal order (a, b, c → predict α, β, γ), they feed it **reversed** (c, b, a → still predict α, β, γ in normal order).

**Why it helps:** Reversing doesn't change the *average* distance between corresponding source/target words, but it drastically shortens the distance between the **first** words of the source and the **first** words of the target. This reduces the "minimal time lag" problem, making it much easier for backpropagation/SGD to establish correspondence between source and target early in training.

**Empirical impact:**
- Test perplexity: 5.8 → 4.7
- Test BLEU: 25.9 → 30.6

Surprisingly, it also improved performance on **long sentences** — the opposite of what the authors initially expected, since they thought reversal would only help the early part of the translation.

---

## Experimental Setup

- **Task:** WMT'14 English → French translation
- **Data:** 12M sentence pairs (348M French words, 304M English words)
- **Vocabulary:** 160,000 most frequent English words (source), 80,000 most frequent French words (target); out-of-vocabulary words mapped to `UNK`

### Model size
- 4-layer LSTMs, 1000 cells/layer, 1000-dim word embeddings
- Sentence representation: 8000-dimensional vector
- **384M total parameters** (64M purely recurrent connections)

### Training details
- Parameters initialized uniformly in [-0.08, 0.08]
- Plain SGD (no momentum), learning rate 0.7, halved every half-epoch after epoch 5
- Trained for 7.5 epochs total
- Batch size 128
- **Gradient clipping:** rescale gradient if its L2 norm exceeds 5
- **Length-bucketed minibatches:** sentences of similar length grouped together → 2x speedup

### Parallelization
- Single GPU: ~1,700 words/sec (too slow)
- **8-GPU setup:** one GPU per LSTM layer (4 GPUs) + 4 GPUs splitting the softmax over 80k vocab
- Achieved: 6,300 words/sec
- Total training time: **~10 days**

### Decoding
- Simple **left-to-right beam search**
- Maintains B partial hypotheses, extends each with every vocabulary word, keeps top-B by log-probability
- Worked reasonably even with beam size 1; beam size 2 captured most of the benefit of larger beams

---

## Results

### Direct Translation (Table 1)

| Method | BLEU |
|---|---|
| Baseline SMT system | 33.30 |
| Single forward (non-reversed) LSTM | 26.17 |
| Single reversed LSTM | 30.59 |
| Ensemble of 5 reversed LSTMs, beam 12 | **34.81** |

This ensemble beat the phrase-based SMT baseline — the first time a pure neural translation system beat a strong phrase-based system at this scale.

### Rescoring (Table 2)

Instead of translating from scratch, the LSTM re-ranked the SMT baseline's 1000-best candidate translations (averaging LSTM log-probability with SMT score):

| Method | BLEU |
|---|---|
| Baseline SMT | 33.30 |
| Rescoring w/ ensemble of 5 reversed LSTMs | **36.5** |
| Best WMT'14 system overall | 37.0 |

This got within 0.5 BLEU of the best published system on the task, just by re-ranking existing candidates.

---

## Performance on Long Sentences

The authors expected the fixed-size vector bottleneck to hurt long sentences (a common concern with encoder-decoder models, later addressed by attention mechanisms). Surprisingly, performance held up well up to ~35 words, with only minor degradation on the longest sentences — attributed to the reversing trick improving "memory utilization."

---

## Qualitative Analysis: Learned Sentence Representations

PCA visualizations of encoder hidden vectors (Figure 2) showed the representations:

- **Cluster by meaning**, not surface form
- Are **sensitive to word order** (e.g., "John admires Mary" sits far from "Mary admires John")
- Are **relatively invariant to active vs. passive voice** (e.g., "I gave her a card" and "She was given a card by me" end up close together)

This suggests the model learned something like semantic representations, not just surface pattern matching — something a bag-of-words model couldn't achieve.

---

## Related Work Positioning

- **Kalchbrenner & Blunsom (2013)** — first to map sentences to vectors and back, but used CNNs, losing word order.
- **Cho et al. (2014)** — similar LSTM-like encoder-decoder idea, but used mainly to *rescore* SMT hypotheses; struggled on long sentences.
- **Bahdanau et al. (2014)** — introduced attention to fix the long-sentence weakness of Cho et al.'s approach (seed of the dominant attention paradigm).
- **Graves (2013)** — differentiable attention mechanisms in general.

---

## Conclusion & Significance

1. A large, deep LSTM with minimal task-specific engineering can beat a mature, heavily-engineered phrase-based SMT system — evidence that scale + the right architecture can outperform hand-crafted linguistic features.
2. The **reversing trick** shows that engineering the *problem representation* to have more short-range dependencies can make optimization dramatically easier — a lesson broader than just translation.
3. The framework correctly generalized to many other sequence-to-sequence tasks beyond translation.

**Historical note:** This paper, alongside Cho et al. and Bahdanau et al., founded the encoder-decoder paradigm that dominated NLP for years, directly motivating attention mechanisms and eventually the Transformer architecture ("Attention Is All You Need," 2017).

---
