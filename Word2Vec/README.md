# Spoken Dialogue Systems - Theory & Explanations

This document contains structured explanations for theory questions from the SDS exercises.

---

## Table of Contents

- [Exercise 1: Word2Vec & Word Embeddings](#exercise-1-word2vec--word-embeddings)

---

## Exercise 1: Word2Vec & Word Embeddings

### Q1: Two Advantages of Dense Parameterized Word Representations vs One-Hot Vectors

**1. Captures Semantic Relationships**
- One-hot vectors treat all words as equally different (orthogonal)
- Dense embeddings place similar words close in vector space
- Enables vector arithmetic: `king - man + woman ≈ queen`

**2. Dimensionality Efficiency**
- One-hot: dimension = vocabulary size (10K-1M+)
- Dense: fixed low dimension (50-300)
- Dense vectors are more memory-efficient and computationally tractable

| Aspect | One-Hot | Dense Embeddings |
|--------|---------|------------------|
| Dimensionality | Vocabulary size (huge) | Fixed small size (50-300) |
| Semantic similarity | None (all orthogonal) | Captured via vector proximity |
| Generalization | None | Can infer similar words |
| Memory efficiency | Poor (sparse) | Good (dense) |

---

### Q2: Intuition Behind Word2Vec Training

**Core Principle: "You shall know a word by the company it keeps"**

Word2Vec is based on the **distributional hypothesis**: words appearing in similar contexts have similar meanings.

**Training Architectures:**

1. **Skip-gram**: Predict context words from target word
   - Input: Center word (e.g., "apple")
   - Output: Surrounding words (e.g., "tasty", "fruit")

2. **CBOW**: Predict target word from context words
   - Input: Surrounding words
   - Output: Center word

**Why It Works:**
Words like "apple" and "orange" both appear near "fruit", "eat", "tasty". The model learns to give them similar vectors to generate similar context predictions. Through gradient descent, words with interchangeable contexts cluster together in vector space.

---

### Q6: Issue Revealed by Similarity Examples

**The Problem: Polysemy (Words with Multiple Meanings)**

The word "apple" has two distinct meanings:
1. **The fruit** - related to "fruit", "tasty", "eat", "orange"
2. **The company** - related to "computer", "technology", "iPhone", "Mac"

**What Happens:**
- Word2Vec creates **a single vector** for "apple" that averages BOTH meanings
- When 'computer' is in the candidate list, it may score higher than 'fruit' because the training corpus contains many co-occurrences of "Apple" (company) with "computer"
- The embedding cannot distinguish which sense you intended

**The Core Issue: Word Embeddings Conflate Multiple Meanings**

| Aspect | Problem |
|--------|---------|
| **Single vector per word** | Cannot represent different senses separately |
| **Context-free** | Same vector regardless of usage context |
| **Frequency bias** | More common meaning dominates |

**Why This Matters:**
- Static embeddings (Word2Vec, GloVe) cannot handle polysemous words properly
- "Apple" ≈ "computer" and "Apple" ≈ "orange" simultaneously in the same vector space
- This leads to ambiguous similarity results depending on which other words are present

**Modern Solution:** Contextualized embeddings (BERT, GPT) generate different vectors based on surrounding context, solving this polysemy problem.

---

### Q8: Why Sentence Similarity Remains Close

**The Problem: Averaging Dilutes Semantic Signals**

When averaging word embeddings, generic function words ("is", "more", "than") pull the sentence vector toward a neutral center. The sentence becomes a "blurry" mix where:
- Specific words like "tasty" and "orange" suggest the fruit meaning of "apple"
- But the averaging with 3 generic words dilutes this signal
- Both "fruit" and "computer" end up with similar similarity scores because the sentence vector is too generic to strongly distinguish either

**Key Issue:** Averaging is a bag-of-words approach that loses word order and compositionality. "Apple is tasty" and "apple computer" have very different meanings, but word averaging cannot tell them apart.

**Better Approaches:** TF-IDF weighting, SIF (Smooth Inverse Frequency), or contextualized embeddings (BERT) that capture word order and context.

---

### Q9: Advantage of Embeddings for Neural Networks

**Yes — embeddings and clusters provide significant advantages:**

1. **Semantic Feature Representation**: Instead of raw text or one-hot vectors, the NN receives meaningful dense features that capture word relationships, enabling better generalization.

2. **Clustering Captures Domain Structure**: In DSTC2, the three clusters (price levels, locations, food types) help the network recognize patterns and predict user intent (e.g., detecting budget constraints or cuisine preferences).

3. **Data Efficiency**: Words not seen during training can be handled if similar words were seen — knowledge transfers from "italian" to "thai" because their vectors are close.

4. **Dimensionality Reduction**: Embeddings compress thousands of vocabulary dimensions to 50-300, speeding up training and reducing overfitting.

---
### Dataset: DSTC2
**DSTC2** (Dialog State Tracking Challenge 2) is a benchmark dataset of human-computer dialogues where users search for restaurants. It contains utterances with domain-specific vocabulary organized into clear semantic clusters:
- **Food types**: italian, indian, chinese
- **Locations**: north, south, east, west, centre
- **Price levels**: cheap, moderate, expensive

---

### Word2Vec vs Classification Neural Network Training

| Aspect | Word2Vec | Classification NN |
|--------|----------|-------------------|
| **Architecture** | Shallow (embedding + linear) | Shallow or deep (RNN, Transformer) |
| **Training Type** | Self-supervised | Supervised |
| **Labels** | None required (creates from context) | Human-labeled data required |
| **Task** | Predict context words | Predict class (intent, category) |
| **Loss Function** | Negative sampling / Softmax | Cross-entropy for classification |
| **Output** | Word vectors | Class probabilities |

**Key Difference:** Word2Vec is self-supervised — it creates training pairs from raw text (e.g., Input="apple", Target="tasty"). Classification NNs require labeled examples (e.g., Input="cheap italian food", Target="price=cheap, food=italian").

**Typical Workflow:** First train Word2Vec (unsupervised) to get word vectors, then use those vectors as input features to a classification NN (supervised).

---

### Word2Vec Hyperparameters Reference

| Parameter | Role | Effect |
|-----------|------|--------|
| **`vector_size`** | Embedding dimension | Higher = more expressive, slower (common: 50-300) |
| **`window`** | Context window size | Words within N distance considered neighbors |
| **`min_count`** | Min word frequency | Words appearing < N times are ignored (filters noise) |
| **`epochs`** | Training iterations | More passes = better convergence, longer training |
| **`sg`** | Algorithm choice | 1=Skip-gram (rare words), 0=CBOW (frequent words, faster) |
| **`negative`** | Negative samples | How many "wrong" words to contrast against |
| **`sample`** | Downsampling rate | Reduces frequent words ("the", "is") from training |

**Key Trade-offs:**
- **Higher `vector_size`**: Better quality, more memory/time
- **Larger `window`**: Captures broader context but dilutes local patterns
- **Skip-gram vs CBOW**: Skip-gram for small data/rare words, CBOW for large data/speed

---

### t-SNE Visualization

**t-SNE** (t-Distributed Stochastic Neighbor Embedding) is a dimensionality reduction technique for visualizing high-dimensional data (like word embeddings) in 2D or 3D.

**Purpose:** Word embeddings are 50-300 dimensions — t-SNE projects them to 2D while preserving local structure, so words with similar meanings cluster together.

**Key Parameters:**
| Parameter | Effect |
|-----------|--------|
| `n_components` | Output dimensions (2 or 3) |
| `perplexity` | Number of neighbors considered (5-50) |
| `n_iter` | Iterations for convergence |

**Limitation:** Preserves local structure, not global distances. Don't use t-SNE coordinates for measuring actual similarity.

---

### Word2Vec vs FastText Comparison

| Aspect | Word2Vec | FastText |
|--------|----------|----------|
| **Representation** | Single vector per word | Word = sum of character n-gram vectors |
| **OOV words** | Cannot handle (error) | Can generate vectors for unknown words |
| **Training speed** | Faster | Slower (more parameters) |
| **Morphology** | Ignores word structure | Captures subword patterns |
| **Example** | "inexpensive" → error | "inexpensive" → vector from "in", "expen", "sive" |

**Key Advantage of FastText:** Can handle out-of-vocabulary (OOV) words by composing vectors from subword units. Better for morphologically rich languages, typos, and rare words.

---

### Q10: Problem with FastText for User Preference Detection

**The Issue: False Similarities from Subword Overlap**

While FastText handles OOV words well, it creates **spurious similarities** that confuse classifiers:

1. **Character overlap ≠ Semantic similarity**: Words sharing n-grams get similar vectors even with different meanings (e.g., "indian" (food) and "indiana" (location) share "india")

2. **Cross-category confusion**: FastText conflates words across preference categories (price, location, cuisine) based on superficial character matches rather than actual semantics

3. **Noise from common subwords**: Frequent character sequences create "false neighbors" that don't represent true semantic relationships

**For preference detection**, clean category boundaries are essential. FastText's subword approach blurs these boundaries, making it harder for classifiers to distinguish user intents accurately.

---

### Gensim and FastText Libraries

**Gensim**: Open-source Python library for topic modeling and document similarity. Provides efficient implementations of Word2Vec, FastText, and other embedding methods.

**FastText** (Facebook): Extension of Word2Vec that uses character n-grams. Key advantage: handles out-of-vocabulary words by composing vectors from subwords.

---

### SkipGram Implementation from Scratch (PyTorch)

**Architecture:**
```python
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)  # U matrix
        self.output_layer = nn.Linear(embedding_dim, vocab_size)     # W matrix
```

**Components:**
| Layer | Purpose | Dimensions |
|-------|---------|------------|
| `nn.Embedding` | Lookup table for word vectors | (vocab_size, embedding_dim) |
| `nn.Linear` | Projects to vocabulary space | (embedding_dim, vocab_size) |

**Training Loop:**
1. **Shuffle** training pairs (center_word, context_word)
2. **Convert** numpy arrays to PyTorch tensors (`torch.LongTensor`)
3. **Forward pass**: Input word index → embedding → linear projection → output logits
4. **Loss**: Cross-entropy between predicted distribution and true context word
5. **Backward pass**: `loss.backward()` computes gradients
6. **Optimizer step**: `optimizer.step()` updates U and W matrices

**Common Issues Fixed:**
- `idx_pairs` must be converted to numpy array for indexing
- Model must inherit from `nn.Module` (not plain Python class)
- Tensors must be explicitly converted from numpy
- Accumulate loss with `total_loss += loss.item()` for monitoring

---

---

### SkipGram Activation Functions Explained

**Key Insight:** The custom SkipGram implementation uses **no explicit activation functions** in the forward pass. Here's why:

**1. Embedding Layer (`nn.Embedding`)**
- **No activation** — simply a lookup table
- Maps word index → dense vector by indexing into learnable matrix $U$
- Shape: `(vocab_size, embedding_dim)`

**2. Output Layer (`nn.Linear`)**
- **No activation** — returns raw **logits** (unnormalized scores)
- Linear transformation: $z = v \cdot W$ (where $v$ is the embedding vector)
- Shape: `(embedding_dim, vocab_size)`

**3. Where Softmax Actually Happens**
The "activation" is **implicit in `nn.CrossEntropyLoss`**:
- `CrossEntropyLoss` combines `LogSoftmax` + `Negative Log-Likelihood`
- Takes raw logits as input, applies log-softmax internally
- This is why you must NOT apply softmax manually in `forward()`

**Common Pitfall:** Applying `F.softmax()` in `forward()` then using `CrossEntropyLoss` = **double softmax**, which breaks training.

---

### CrossEntropyLoss: How It Determines Classes

**No need to specify number of classes!**

```python
output = model(X_batch)  # shape: (batch_size, vocab_size)
loss = criterion(output, Y_batch)  # CrossEntropyLoss infers automatically
```

PyTorch uses the **second dimension** of the output tensor as the number of classes:
- Your `output_layer` has `vocab_size` units
- Output shape: `(batch_size, vocab_size)`
- `CrossEntropyLoss` treats each of the `vocab_size` logits as a separate class

| Output Shape | Implied Classes |
|--------------|-----------------|
| `(32, 100)` | 100 classes |
| `(32, 10000)` | 10,000 classes |

The number of classes is **dynamic** — it adapts to whatever your model outputs.

---

### `find_most_similar` Function Explained

**Purpose:** Query the trained embeddings to find words with the highest cosine similarity to a given word.

```python
def find_most_similar(word, num_similar=3):
    if word not in word2idx:          # Check vocabulary membership
        return []

    word_idx = word2idx[word]          # Get integer index
    word_embedding = model.embeddings.weight[word_idx]  # Extract vector from U matrix

    similarities = torch.cosine_similarity(
        word_embedding.unsqueeze(0),   # Shape: (1, embedding_dim)
        model.embeddings.weight,       # Shape: (vocab_size, embedding_dim)
        dim=1                          # Compare along embedding dimension
    )                                  # Returns: (vocab_size,) similarity scores

    similar_indices = torch.topk(similarities, k=num_similar + 1).indices[1:]
    # topk(4) gets [self, word1, word2, word3]
    # [1:] skips self-similarity (always 1.0), keeps top 3 others

    similar_words = [idx2word[idx.item()] for idx in similar_indices]
    return similar_words               # Convert indices back to strings
```

**Step-by-Step Breakdown:**

| Step | Operation | Output |
|------|-------------|--------|
| 1 | Vocabulary lookup | Word index (int) |
| 2 | Index into embedding matrix | Vector of shape `(embedding_dim,)` |
| 3 | Cosine similarity vs all words | Score tensor `(vocab_size,)` |
| 4 | Top-k selection | Indices of most similar words |
| 5 | Index-to-word mapping | List of similar word strings |

**Key Details:**
- **Cosine similarity** measures vector alignment: $\cos(\theta) = \frac{A \cdot B}{||A|| \ ||B||}$
- **Skip self:** The target word is always most similar to itself (similarity = 1.0), so it's excluded
- **No output layer needed:** Similarity is computed directly on the embedding matrix $U$, not the output layer $W$

---

*Last updated: 2026-04-21*
