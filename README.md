# SASRec: Self-Attentive Sequential Recommendation

A production-ready implementation of **Self-Attentive Sequential Recommendation (SASRec)** demonstrating how Transformer architectures outperform traditional statistical methods for next-item prediction tasks.

---

## 🎯 Problem Statement

**Task:** Given a user's interaction history `[Item₁, Item₂, ..., Itemₙ]`, predict what they'll interact with next.

**Dataset:** MovieLens-1M (1M ratings, filtered to users/movies with ≥5 interactions)

**Evaluation Metric:** Hit Rate @ 10 (Leave-One-Out validation), Full Rank W/ ~59,000 candidates

---

## 📊 Performance Results

<div align="center">
<img src="assets/baseline_comparison.png" alt="Performance Comparison" width="600"/>
</div>

| Model | Approach | Hit@10 | vs. Baseline |
|-------|----------|--------|--------------|
| **Global Popularity** | Most-watched items | **4.45%** | — |
| **Genre-Based** | User's favorite genre | **3.84%** | -14% |
| **Markov Chain (AR-1)** | Last-item transitions | **15.85%** | +256% |
| **Hybrid (α=0.5)** | Markov + Popularity | **15.97%** | +259% |
| **AR(2)** | Two-item context | **17.72%** | +298% |
| **SASRec** | Transformer Encoder | **28.00%** | 551+% |

### Why the Gap?

**Markov Chains** excel at local patterns (sequels, franchises) but fail to capture:
- Long-range dependencies (signals from 50+ steps ago)
- Multi-interest users (alternating between genres)
- Temporal dynamics (evolving tastes)

**SASRec** uses self-attention to dynamically weigh the entire interaction history, learning complex patterns that simple transition probabilities miss.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│  Input: [movie₁, movie₂, ..., movieₙ]  │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│         Embedding Fusion                │
│  • Item Embedding (num_movies)          │
│  • Position Embedding (max_len=200)     │
│  • Genre Embedding (num_genres)         │
│  • Year Embedding (13 buckets)          │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│      Transformer Encoder (L=3)          │
│  ┌───────────────────────────────────┐  │
│  │  Multi-Head Self-Attention (H=4)  │  │
│  │  + Residual + LayerNorm           │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │  Feed-Forward (dim × 4)           │  │
│  │  + Residual + LayerNorm           │  │
│  └───────────────────────────────────┘  │
│  (Causal Masking: prevents leakage)     │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  Output Layer: Linear(256 → vocab)      │
└─────────────────────────────────────────┘
                  ↓
      Predictions over all items
```

**Configuration:**
- Embedding Dimension: 256
- Transformer Layers: 3
- Attention Heads: 4
- Max Sequence Length: 200
- Dropout: 0.1
- Activation: GELU

---

## 🚀 Key Engineering Features

### 1. Mixed Precision Training (FP16)
```python
with torch.amp.autocast('cuda'):
    logits = model(input_seq, genres, years)
    loss = criterion(logits, targets)
scaler.scale(loss).backward()
```
- Reduces VRAM usage by ~40%
- Speeds up training on modern GPUs (RTX series)
- Automatic loss scaling prevents underflow

### 2. Gradient Accumulation
```python
loss = loss / ACCUMULATION_STEPS
scaler.scale(loss).backward()
if (step + 1) % ACCUMULATION_STEPS == 0:
    scaler.step(optimizer)
```
- Simulates larger batch sizes (32 × 4 = 128 effective)
- Fits training on consumer GPUs

### 3. Vectorized Metadata Lookup
```python
# Pre-compute mappings: O(N) once vs O(N²) per epoch
genre_lookup = torch.tensor([idx_to_genre[i] for i in range(num_movies)])
genres_tensor = genre_lookup[input_ids]  # Instant batch lookup
```
- Replaces slow Python loops with tensor indexing
- Dramatically speeds up data loading

### 4. Causal Attention Masking
```python
mask = torch.triu(torch.ones(L, L) * float('-inf'), diagonal=1)
transformer_encoder(x, mask=mask)
```
- Prevents model from "cheating" by seeing future items
- Essential for valid sequential modeling

---

## 📂 Project Structure

```
sasrec-recommender/
├── data/
│   ├── ratings.csv          # MovieLens ratings
│   └── movies.csv           # Movie metadata
├── src/
│   ├── __init__.py
│   ├── model.py             # SASRecModel (Transformer)
│   ├── dataset.py           # SASRecDatasetSeq
│   └── utils.py             # Data preprocessing
├── train.py                 # Training script with AMP
├── evaluate.py              # Benchmark baselines + SASRec
├── requirements.txt
├── images/
└── README.md
```

---

## 🛠️ Installation & Usage


### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/sasrec-recommender.git
cd sasrec-recommender

# Install dependencies
pip install -r requirements.txt

# Download MovieLens-1M
# Place ratings.csv and movies.csv in data/
```

### Training
```bash
python train.py
```

**Expected Output:**
```
Loading raw data...
Building sequence tensors...
Starting Training on 6040 sequences...
Epoch 1 | Loss: 1.6977 | Time: 169.2s
Epoch 2 | Loss: 1.5365 | Time: 169.7s
...
Model Saved.
```

### Evaluation
```bash
python evaluate.py
```

**Output:**
```
Evaluating ...
Hit Rate @ 10: 0.2800
```

---

## 📈 Baseline Implementations

### Global Popularity
```python
def predict_global(history, k=10):
    recs = []
    seen = set(history)
    for movie in popularity_rankings:
        if movie not in seen:
            recs.append(movie)
            if len(recs) == k: break
    return recs
```

### Markov Chain (First-Order)
```python
# Learn P(next | current)
transitions = defaultdict(lambda: defaultdict(int))
for seq in train_data:
    for i in range(len(seq) - 1):
        transitions[seq[i]][seq[i+1]] += 1

def predict_markov(history, k=10):
    last_movie = history[-1]
    candidates = sorted(transitions[last_movie].items(), 
                       key=lambda x: x[1], reverse=True)
    return [movie for movie, _ in candidates[:k]]
```

### AR(2) Model
```python
# Learn P(next | prev_two)
transitions_ar2 = defaultdict(lambda: defaultdict(int))
for seq in train_data:
    for i in range(len(seq) - 2):
        state = (seq[i], seq[i+1])
        transitions_ar2[state][seq[i+2]] += 1
```

---

## 🔬 Implementation Details

### Data Preprocessing
1. **Temporal Sorting:** Order by timestamp (chronological sequences)
2. **ID Mapping:** Assign contiguous indices [1, N] (0 reserved for padding)

### Feature Engineering
- **Year Buckets:** `[<1980, 1980-1990, ..., 2010-2020, >2020]` (7 bins)
- **Primary Genre:** First listed genre (e.g., "Action|Thriller" → "Action")

### Training Configuration
- **Loss Function:** CrossEntropyLoss with `ignore_index=0` (padding mask)
- **Optimizer:** Adam (lr=0.001, β₁=0.9, β₂=0.999)
- **Batch Size:** 32 with 4-step gradient accumulation (effective: 128)
- **Epochs:** 20
- **Hardware:** Single NVIDIA GPU (CUDA-enabled)

### Model Architecture
```python
SASRecModel(
    num_movies=59048,      # Vocabulary size (after filtering)
    num_genres=21,        # Unique genres
    num_years=7,          # Year buckets
    embed_dim=256,
    max_len=200,
    num_layers=3,
    num_heads=4,
    dropout=0.1
)
```

---

## 📊 Baseline Results Breakdown

| Baseline | Strengths | Weaknesses | Use Case |
|----------|-----------|------------|----------|
| **Global Pop** | Simple, fast | Ignores personalization | Cold-start users |
| **Genre-Based** | Captures basic preferences | Oversimplifies taste | Genre-specific apps |
| **Markov Chain** | Good for sequential patterns | Limited context window | Session-based rec |
| **AR(2)** | Improved context | Still local | Short sequences |
| **Hybrid** | Balances popularity & transitions | Manual tuning | Production fallback |

---

## Future Improvement

Contributions welcome! Areas for improvement:
- [ ] Add evaluation metrics (NDCG, MRR)
- [ ] Hyperparameter tuning scripts
- [ ] Inference optimization (ONNX export)

---

## 📚 References

1. **SASRec Paper:** Kang, W.-C., & McAuley, J. (2018). [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781). In KDD'18.
2. **Transformer Architecture:** Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). In NeurIPS'17.
3. **MovieLens Dataset:** Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets. ACM TIIS.

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **GroupLens Research** for the MovieLens dataset
- **Original SASRec authors** for the architecture design
- **PyTorch team** for the deep learning framework

---
