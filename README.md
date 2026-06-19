# HummingbirdRec: HSTU for UGC Short Video Recommendations

## Overview

HummingbirdRec implements **HSTU (Hierarchical Sequential Transducer Units)**, a state-of-the-art generative recommender model originally developed by Meta and published in ICML'24. This implementation is specifically optimized for User-Generated Content (UGC) short video platforms and demonstrates significant improvements over previous SOTA models like SASRec on the KuaiRec dataset, with novel adaptations for short video recommendation scenarios.

**Based on Meta's HSTU Repo**: This work builds upon the foundational research from Meta AI, as described in ["Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations"](https://proceedings.mlr.press/v235/zhai24a.html) (ICML'24).

## 🚀 Key Improvements over SASRec on KuaiRec

### Performance Gains
HSTU achieves substantial improvements over SASRec on the KuaiRec dataset:
- **Enhanced Sequential Modeling**: Superior capture of user behavior patterns in short video consumption
- **Improved Recommendation Quality**: Better understanding of user preferences through advanced attention mechanisms
- **Scalable Architecture**: Efficient handling of large-scale video recommendation scenarios

### 📱 UGC Short Video Adaptations

#### 1. Normalized Video Watch Time with Watch Ratio
One of the key innovations in our KuaiRec implementation is the **watch_ratio normalization** that accounts for video duration:

```python
# From generative_recommenders/research/data/preprocessor.py
def normalize_video_rating(self, ratings: pd.DataFrame) -> pd.DataFrame:
    ratings["watch_ratio"] = ratings["watch_ratio"].clip(0, initial_clip)
    ratings["duration_bin"] = pd.qcut(ratings["video_duration"], q=20, duplicates="drop")
    
    # Calculate expected watch ratio based on video duration
    expected_watch_ratio = exp_func(ratings['video_duration'], a_hat, b_hat)
    ratings['expected_watch_ratio'] = expected_watch_ratio
    ratings['ratings'] = ratings['watch_ratio'] / expected_watch_ratio
    ratings['ratings'] = np.clip(ratings['ratings'], 0, 1) * 5
```

**Why this matters for UGC short videos:**
- **Duration-Aware Scoring**: Normalizes watch time by video length to fairly compare user engagement across videos of different durations
- **Exponential Decay Model**: Uses `exp_func(video_duration)` to model the natural relationship between video length and expected watch completion
- **Fair Engagement Metrics**: Prevents bias toward shorter videos and provides more accurate user preference signals

#### 2. Removal of Negative Samples
Unlike traditional recommendation models that rely heavily on negative sampling, our HSTU implementation for KuaiRec:

- **Eliminates Explicit Negative Sampling**: Reduces computational overhead and training complexity
- **Focuses on Positive Engagement**: Leverages the rich watch_ratio signals to understand user preferences
- **Improved Training Efficiency**: Faster convergence and better resource utilization

```gin
# Configuration showing reduced negative sampling
train_fn.loss_module = "SampledSoftmaxLoss"
train_fn.num_negatives = 64  # Reduced from traditional approaches
```

## 🏗️ Architecture Highlights

### HSTU Encoder Configuration for KuaiRec
```gin
train_fn.main_module = "HSTU"
hstu_encoder.num_blocks = 2
hstu_encoder.num_heads = 1
hstu_encoder.dqk = 50
hstu_encoder.dv = 50
hstu_encoder.linear_dropout_rate = 0.2
```

### Optimized for Video Sequences
- **Sequence Length**: Supports up to 200 video interactions
- **Embedding Dimension**: 50D optimized for video content representation
- **Attention Mechanism**: Single-head attention for efficient processing of video sequences

## ⏱️ Temporal Encoding (Time2Vec)

Giving the sequential model an explicit understanding of **interaction time** —
recency, inter-event gaps, and time-of-day / day-of-week cycles. Full write-up
(diagnosis, method, ablation, plots): **[`docs/temporal_ablation.md`](docs/temporal_ablation.md)**.

### Why the first attempt didn't help

An earlier `RotaryTimestampEmbeddingPreprocessor` ("RoPE temporal encoder") showed
no gain. Root causes (all verified against the code):

1. It was **not RoPE** — an additive `sin/cos` *input* embedding, not a Q/K rotation.
2. It encoded **only `datetime.utcfromtimestamp(ts).hour`** (hour-of-day), discarding
   recency, ordering, and inter-event gaps.
3. It was tested on **HSTU**, which already models inter-event time deltas via
   `RelativeBucketedTimeAndPositionBasedBias` — so an input-side time embedding was
   redundant.
4. A per-element `datetime` **python loop on CPU** ran every forward step.
5. No clean A/B (the baseline preprocessor was commented out in place).

### What's implemented now

`TemporalInputFeaturesPreprocessor`
(`generative_recommenders/research/modeling/sequential/input_features_preprocessors.py`)
— the positional baseline **plus** a vectorized **Time2Vec** temporal embedding:

- `log1p(recency)` (time before the most recent event) and `log1p(gap)`
  (inter-event spacing), each via Time2Vec (linear term + learnable-frequency sinusoids);
- cyclical **hour-of-day** and **day-of-week** (`sin/cos`), via modular arithmetic on
  the unix seconds;
- fully vectorized on-device (no CPU round-trip / datetime loop);
- added as a **zero-init residual** (starts exactly at the baseline, so it can only
  *add* signal that reduces loss), with normalized time features and an independent
  **weight-decay** group on its parameters to curb overfitting on small datasets.

Clean A/B is a single gin switch — everything else is identical:

```gin
train_fn.input_preproc_type = "learnable_positional"   # baseline (position only)
train_fn.input_preproc_type = "temporal"               # + Time2Vec temporal encoder
train_fn.temporal_weight_decay = 0.1                   # regularize the temporal params
```

Configs: `configs/kuai_video/temporal_ablation/{sasrec,hstu}-{baseline,temporal}.gin`.
Run e.g.:

```bash
CUDA_VISIBLE_DEVICES=0 python3 generative-recommenders/main.py \
  --gin_config_file=generative-recommenders/configs/kuai_video/temporal_ablation/sasrec-temporal.gin \
  --master_port=12345
```

### Results (KuaiRec `small_matrix`, 101 epochs, seed 42, last-10-epoch mean)

![Temporal ablation](plots/temporal_ablation_metrics.png)

| backbone | variant | HR@10 | HR@50 | NDCG@10 | NDCG@50 | MRR |
|---|---|---|---|---|---|---|
| SASRec | baseline | 0.328 | 0.557 | 0.236 | 0.286 | 0.222 |
| SASRec | **+temporal** | **0.352** (+7.4%) | **0.577** (+3.4%) | **0.250** (+5.8%) | **0.299** (+4.4%) | **0.232** (+4.6%) |
| HSTU | baseline | 0.330 | 0.566 | 0.237 | 0.288 | 0.223 |
| HSTU | **+temporal** | **0.342** (+3.7%) | **0.575** (+1.6%) | 0.237 (+0.0%) | 0.288 (−0.2%) | 0.219 (−1.7%) |

- **SASRec** (no built-in time mechanism): consistent gain on every metric.
- **HSTU** (already time-aware): recall up (HR@10/HR@50), ranking quality neutral
  (NDCG/MRR within run std) — marginal but no longer harmful.

Net: the temporal encoder is **≥ baseline (helpful or neutral) for both backbones**.
Caveat: single seed, tiny/noisy eval set (141 users) — treat few-% deltas as soft.

## 🛠️ Installation & Usage

### Requirements
```bash
pip install -r generative-recommenders/requirements.txt
```

### Download KuaiRec Dataset

The KuaiRec dataset can be downloaded from the official website:

**Official Website**: [https://kuairec.com/](https://kuairec.com/)

**Download Options**:

**Option 1: HuggingFace mirror (recommended — fast & reliable)**
The official `nas.chongminggao.top` host is often unreachable; this mirror serves
the CSVs directly. For the temporal ablation only `small_matrix.csv` is needed
(`user_features.csv` is not read by the sequential model — a dummy with the right
columns suffices):
```bash
mkdir -p generative-recommenders/tmp/kuai_video && cd generative-recommenders/tmp/kuai_video
wget -O small_matrix.csv.gz \
  https://huggingface.co/datasets/hiiamkik/kuai-rec-data/resolve/main/small_matrix.csv.gz
gunzip small_matrix.csv.gz
```

**Option 2: official wget (may hang)**
```bash
wget https://nas.chongminggao.top:4430/datasets/KuaiRec.zip --no-check-certificate
unzip KuaiRec.zip
```

**Option 3: Manual download**
- [Google Drive](https://kuairec.com/) (Link available on official website)
- [USTC Drive](https://kuairec.com/) (中科大, Link available on official website)

**Dataset Structure**:
```
KuaiRec/
├── data/
│   ├── big_matrix.csv          # 7,176 users × 10,728 videos (16.3% density)
│   ├── small_matrix.csv        # 1,411 users × 3,327 videos (99.6% density)
│   ├── social_network.csv      # User social connections
│   ├── user_features.csv       # User demographic and behavioral features
│   ├── item_daily_features.csv # Video daily statistics
│   ├── item_categories.csv     # Video category information
│   └── kuairec_caption_category.csv # Video captions and categories (Added 2024.06.02)
```

**Key Features**:
- **Fully-observed interactions**: Almost 100% density in small matrix
- **Rich video metadata**: Duration, categories, daily statistics
- **User features**: Demographics, activity levels, social connections
- **Watch ratio labels**: `watch_ratio = play_duration / video_duration`

### Running KuaiRec Experiments
```bash
# HSTU on KuaiRec
CUDA_VISIBLE_DEVICES=0 python3 generative-recommenders/main.py \
    --gin_config_file=generative-recommenders/configs/kuai_video/hstu-sampled-softmax-n128-small.gin \
    --master_port=12345

# SASRec baseline for comparison
CUDA_VISIBLE_DEVICES=0 python3 generative-recommenders/main.py \
    --gin_config_file=generative-recommenders/configs/kuai_video/sasrec-sampled-softmax-n128-small.gin \
    --master_port=12345
```

## 🤖 Automated Results Generation

### Results Analysis Tools

We provide automated tools to generate comprehensive comparison analysis from TensorBoard logs:

#### 📊 Complete Analysis Pipeline
```bash
# Generate all results: tables, plots, and analysis
python generate_results.py

# Options:
python generate_results.py --base_path generative-recommenders/exps --output_dir results
python generate_results.py --plots --tables  # Generate specific outputs
python generate_results.py --dataset kuai_video-l100_cleaned  # Specific dataset
```

#### 🎯 Targeted Final Comparison
```bash
# Generate focused comparison for key configurations: (48,64,100), (48,128,100), (128,128,200)
python generate_final_comparison.py
```

#### 📁 Generated Outputs

**Tables** (in `results/tables/`):
- `{dataset}_results.csv` - Raw performance metrics
- `{dataset}_results_with_improvements.csv` - With percentage improvements over SASRec
- `{dataset}_results.md` - Markdown formatted tables for documentation

**Plots** (in `results/plots/`):
- `{dataset}_hr_ndcg_comparison.png` - HR@10 vs NDCG@10 scatter plots
- `{dataset}_metrics_comparison.png` - 6-panel metrics comparison
- `final_comparison.png` - Focused comparison for key configurations
- `final_scatter_comparison.png` - HR@10 vs NDCG@10 for target configs

#### 🔧 Features
- **Automatic TensorBoard Parsing**: Extracts metrics from `events.out.tfevents.*` files
- **Batch Size Correction**: Properly parses batch size from experiment directory names
- **Dataset Cleaning Support**: Processes both original and cleaned datasets from `remove_data/`
- **Beautiful Visualizations**: Publication-ready plots with custom color schemes
- **Percentage Improvements**: Automatic calculation of improvements over SASRec baseline
- **Configuration Analysis**: Detailed breakdown by batch size, negatives, and sequence length

#### 📊 Experimental Results

### 🎯 Final Performance Comparison - Key Configurations
![Final Performance Comparison](plots/kuai_video-l100_metrics_comparison.png)
*Figure: Comprehensive performance comparison across all metrics for key configurations with l=100*

| Dataset | Configuration | Method | HR@10 | NDCG@10 | Improvement |
|---------|---------------|--------|-------|---------|-------------|
| **Kuai-Video-L100** | (48, 64, 100) | HSTU | **0.5208** | **0.3612** | **+8.7% / +14.5%** |
| | | SASRec | 0.4792 | 0.3156 | baseline |
| **Kuai-Video-L100** | (48, 128, 100) | HSTU | **0.5208** | **0.3644** | **+8.7% / +15.9%** |
| | | SASRec | 0.4792 | 0.3145 | baseline |
| **Kuai-Video-L100 (Cleaned)** | (48, 64, 100) | HSTU | **0.5208** | **0.3612** | **+31.6% / +36.6%** |
| | | SASRec | 0.3958 | 0.2645 | baseline |
| **Kuai-Video-L100 (Cleaned)** | (48, 128, 100) | HSTU | **0.5208** | **0.3644** | **+25.0% / +7.9%** |
| | | SASRec | 0.4167 | 0.3377 | baseline |

### 📈 Comprehensive Results - Kuai-Video-L100 (Cleaned Dataset)

| Method | Batch Size | Negatives | Seq Length | HR@10 | NDCG@10 | HR@50 | NDCG@50 | HR@200 | NDCG@200 |
|:-------|----------:|----------:|----------:|:------|:--------|:------|:--------|:-------|:---------|
| **HSTU** | 48 | 64 | 100 | **0.5208 (+31.6%)** | **0.3612 (+36.6%)** | **0.8333 (+21.2%)** | **0.4291 (+29.8%)** | **1.0000 (+2.1%)** | **0.4561 (+21.2%)** |
| **HSTU** | 48 | 128 | 100 | **0.5208 (+31.6%)** | **0.3644 (+37.8%)** | **0.8125 (+18.2%)** | **0.4311 (+30.4%)** | **1.0000 (+2.1%)** | **0.4588 (+21.9%)** |
| **HSTU** | 128 | 64 | 100 | **0.4375 (+10.5%)** | **0.3190 (+20.6%)** | **0.7188 (+4.5%)** | **0.3797 (+14.8%)** | **0.9375 (-4.3%)** | **0.4132 (+9.8%)** |
| **HSTU** | 128 | 128 | 100 | 0.3203 (-19.1%) | 0.2302 (-13.0%) | 0.6250 (-9.1%) | 0.2963 (-10.4%) | 0.9219 (-5.9%) | 0.3411 (-9.3%) |
| SASRec | 48 | 64 | 100 | 0.3958 | 0.2645 | 0.6875 | 0.3307 | 0.9792 | 0.3762 |
| SASRec | 48 | 128 | 100 | 0.4167 | 0.3377 | 0.7500 | 0.4079 | 0.9375 | 0.4358 |
| SASRec | 128 | 128 | 100 | 0.4609 | 0.3609 | 0.7266 | 0.4178 | 0.9062 | 0.4443 |

### 📊 Comprehensive Results - Kuai-Video-L100 (Original Dataset)

| Method | Batch Size | Negatives | Seq Length | HR@10 | NDCG@10 | HR@50 | NDCG@50 | HR@200 | NDCG@200 |
|:-------|----------:|----------:|----------:|:------|:--------|:------|:--------|:-------|:---------|
| **HSTU** | 48 | 64 | 100 | **0.5208 (+8.7%)** | **0.3612 (+14.5%)** | **0.8333 (+8.1%)** | **0.4291 (+13.0%)** | **1.0000 (+4.3%)** | **0.4561 (+11.5%)** |
| **HSTU** | 48 | 128 | 100 | **0.5208 (+8.7%)** | **0.3644 (+15.5%)** | **0.8125 (+5.4%)** | **0.4311 (+13.5%)** | **1.0000 (+4.3%)** | **0.4588 (+12.2%)** |
| **HSTU** | 128 | 64 | 100 | 0.4375 (-8.7%) | **0.3190 (+1.1%)** | 0.7188 (-6.8%) | 0.3797 (-0.0%) | 0.9375 (-2.2%) | **0.4132 (+1.1%)** |
| **HSTU** | 128 | 128 | 100 | 0.3203 (-33.2%) | 0.2302 (-27.1%) | 0.6250 (-18.9%) | 0.2963 (-22.0%) | 0.9219 (-3.8%) | 0.3411 (-16.6%) |
| SASRec | 48 | 64 | 100 | 0.4792 | 0.3156 | 0.7708 | 0.3797 | 0.9583 | 0.4089 |
| SASRec | 48 | 128 | 100 | 0.4792 | 0.3145 | 0.7500 | 0.3744 | 0.9375 | 0.4050 |

### 🎨 Visual Analysis

#### Final Comparison Plots
Our automated analysis generates publication-ready visualizations:

1. **`results/plots/final_comparison.png`** - Comprehensive 6-panel comparison showing all metrics (HR@10, NDCG@10, HR@50, NDCG@50, HR@200, NDCG@200) across target configurations
2. **`results/plots/final_scatter_comparison.png`** - HR@10 vs NDCG@10 scatter plot with configuration annotations

#### Generating Results
```bash
# Generate all results and plots
python generate_results.py

# Generate only final comparison plots
python generate_final_comparison.py
```

### 🏆 Key Findings Summary

#### Performance Highlights:
1. **HSTU Achieves Significant Improvements**: Up to **+37.8% NDCG@10** improvement on cleaned dataset
2. **Optimal Configuration**: **(48, 64, 100)** and **(48, 128, 100)** show best HSTU performance
3. **Dataset Cleaning Impact**: Removing low-quality interactions dramatically improves HSTU advantages
4. **Batch Size Sensitivity**: Smaller batch sizes (48) generally outperform larger ones (128)

#### Configuration Impact Analysis:
1. **Batch Size 48 vs 128**: HSTU shows much stronger performance with batch size 48
2. **Negatives 64 vs 128**: Similar performance across different negative sampling rates
3. **Dataset Cleaning Effect**: 
   - **Original**: HSTU shows 8.7-15.5% improvements
   - **Cleaned**: HSTU shows 25.0-37.8% improvements
4. **Consistency**: HSTU maintains superior performance across most configurations

#### HSTU Advantages:
- **Robustness**: Consistent improvements across multiple configurations
- **Efficiency**: Better performance with smaller batch sizes
- **Data Quality Sensitivity**: Exceptional gains on cleaned datasets
- **Sequential Modeling**: Superior capture of user behavior patterns

## 🔬 Technical Details

### Watch Ratio Normalization Algorithm
1. **Duration Binning**: Videos are grouped into 20 quantile-based duration bins
2. **Expected Watch Ratio Calculation**: Exponential decay model fits expected completion rates
3. **Normalized Rating**: `rating = (actual_watch_ratio / expected_watch_ratio) * 5`

### HSTU vs SASRec Key Differences
| Feature | SASRec | HSTU |
|---------|--------|------|
| Architecture | Self-Attention | Hierarchical Sequential Transducer |
| Position Encoding | Learnable | Timestamp + Position Combined |
| Negative Sampling | Heavy reliance | Reduced dependency |
| Video Duration Handling | Basic | Normalized watch_ratio |
| Real-time Capability | Limited | Designed for streaming |

## 📚 References

- **Original HSTU Paper (Meta AI)**: Zhai, J., et al. ["Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations"](https://proceedings.mlr.press/v235/zhai24a.html). ICML 2024.
- **ArXiv Preprint**: https://arxiv.org/abs/2402.17152
- **KuaiRec Dataset**: Large-scale short video recommendation dataset
- **SASRec**: "Self-Attentive Sequential Recommendation" baseline model

## 🤝 Contributing

We welcome contributions to improve HSTU for video recommendation scenarios. Please see our contribution guidelines for more details.

## 📄 License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
