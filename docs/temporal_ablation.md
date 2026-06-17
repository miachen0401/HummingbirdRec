# Temporal Encoding Ablation on KuaiRec

Goal: test whether giving the sequential recommender an explicit understanding
of **interaction time** improves next-item recommendation on the KuaiRec
offline dataset (`small_matrix.csv`, fully-observed user×video matrix).

## Background: what was wrong with the previous "RoPE" attempt

The earlier `RotaryTimestampEmbeddingPreprocessor` (commit `664c2e4`) did not
deliver a gain. Root causes (all verified against the code):

1. **It was not RoPE.** It added a `sin/cos` embedding to the input (the
   original Transformer's *additive* absolute encoding), rather than *rotating*
   Q/K by a position-dependent angle inside attention (what RoPE actually is).
2. **It discarded almost all temporal information.** It encoded only
   `datetime.utcfromtimestamp(ts).hour` — the hour-of-day (0–23). Recency,
   ordering, and inter-event gaps were all thrown away. Two events three days
   apart but at the same hour got identical encodings.
3. **It competed with a stronger, already-present mechanism.** HSTU already
   models time via `RelativeBucketedTimeAndPositionBasedBias`
   (`hstu.py`), which buckets the *time deltas* `ts(j) - ts(i)` and adds a
   learned bias inside attention (enabled by default). Adding hour-of-day on the
   input is redundant/noisy on top of that.
4. **Severe perf bug.** Every forward step did `.cpu().numpy()` then a Python
   double loop of `datetime.utcfromtimestamp` over all `B×N` elements, then
   moved back to GPU — breaking the GPU pipeline.
5. **No clean A/B.** The baseline preprocessor was commented out and replaced
   in-code, and the replacement also interleaved rating embeddings
   (`N → N*2`), conflating two changes at once.

## What this ablation does instead

`TemporalInputFeaturesPreprocessor`
(`research/modeling/sequential/input_features_preprocessors.py`) is the
learnable-positional baseline **plus** a learnable temporal embedding, added on
top of the otherwise-untouched baseline path. Features per valid position `i`
(unix-seconds `t_i`):

- `log1p(recency) = log1p(t_last - t_i)` — time before the most recent event;
- `log1p(gap) = log1p(t_i - t_{i-1})` — inter-event spacing;
  both via **Time2Vec** (one linear term + sinusoids with learnable freq/phase);
- cyclical **hour-of-day** and **day-of-week** (`sin/cos`), from modular
  arithmetic on the unix seconds.

All vectorized on-device (no CPU round-trip, no `datetime` loop). Concatenated
features are projected to `embedding_dim` and summed in.

## Clean A/B via gin

The input preprocessor is selected by `train_fn.input_preproc_type`:

- `"learnable_positional"` → baseline (absolute position only)
- `"temporal"` → baseline + Time2Vec temporal encoder

Everything else is identical, so the switch isolates the temporal contribution.

## Runs (4 configs, `configs/kuai_video/temporal_ablation/`)

| config | backbone | input preproc | note |
|---|---|---|---|
| `sasrec-baseline.gin` | SASRec | positional | temporally-blind baseline |
| `sasrec-temporal.gin` | SASRec | temporal | + Time2Vec |
| `hstu-baseline.gin` | HSTU | positional | already has relative time-delta bias |
| `hstu-temporal.gin` | HSTU | temporal | + Time2Vec (tests complementary signal) |

SASRec is the clean test (no built-in time mechanism). HSTU is the hard test
(it already models time deltas), so a gain there means the explicit
recency/cyclical features add signal beyond relative deltas.

Run one (logs to wandb project `HummingbirdRec-temporal`):

```bash
cd generative-recommenders
mkdir -p tmp/ && python3 preprocess_public_data.py   # one-time data prep
CUDA_VISIBLE_DEVICES=0 python3 main.py \
  --gin_config_file=configs/kuai_video/temporal_ablation/hstu-temporal.gin \
  --master_port=12345
```

## Tests

`research/modeling/sequential/tests/input_features_preprocessors_test.py`
(CPU-only — passing on CPU itself proves there is no host-side per-element
loop): shapes/lengths, padding masked to zero, output depends on timestamps
(the key regression guard vs the hour-only bug), reduces to the baseline when
the temporal projection is zeroed, gradients flow to the temporal params.

```bash
cd generative-recommenders && python3 -m pytest \
  generative_recommenders/research/modeling/sequential/tests/input_features_preprocessors_test.py -q
```

## Results

_TBD — filled in from the wandb runs once GPU is free on hula._
