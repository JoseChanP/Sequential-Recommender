# Hybrid SASRec + AR(2) Recommender System

A sequential recommendation system fusing Self-Attention (SASRec) with Auto-Regressive Markov Chains (AR-2) to solve the cold-start problem in sparse user sessions.

## 🚀 Key Features
* **Hybrid Architecture:** Uses a Cascade Inference strategy.
    * **Cold Start (L < 3):** 2nd-Order Markov Chain (AR2) for immediate sequential probability.
    * **Warm Start (L ≥ 3):** Transformer-based SASRec for capturing long-term dependencies.
* **Metadata Awareness:** Embeds Release Year and Genre buckets alongside Item IDs.
* **Optimized Training:** Implements Mixed Precision (AMP) and pinned memory for high-throughput training on NVIDIA GPUs.

## 🛠️ Architecture
The model uses a fallback mechanism to handle varying sequence lengths:
$$P(x_t | x_{<t}) = \begin{cases} \text{GlobalPop} & \text{if } t=0 \\ \text{AR}(2) & \text{if } 0 < t < 3 \\ \text{SASRec}(x_{t-L}, ..., x_{t-1}) & \text{if } t \ge 3 \end{cases}$$

## ⚡ Performance
| Model | Hit Rate @ 10 |
|-------|---------------|
| Global Popularity | 0.0432 |
| Markov Chain | 0.1250 |
| **Hybrid SASRec** | **0.1845** |

## 💻 Tech Stack
* **Core:** PyTorch (CUDA Optimized), NumPy, Pandas
* **Hardware Acceleration:** Torch.amp (Automatic Mixed Precision)

## 🔧 Usage

### 1. Setup
```bash
pip install -r requirements.txt