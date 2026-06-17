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

Run on hula (RTX 4080 SUPER, torch 2.6+cu124), 101 epochs each, seed 42, KuaiRec
`small_matrix` (1270 train / 141 eval users). Metrics = **mean over the last 10
epochs** (the 141-user eval set is small and the per-epoch metric is noisy, so
"max over epochs" cherry-picks spikes; last-10-mean is the stable headline).
Logged to wandb project `HummingbirdRec-temporal`. Plots in `plots/`.

| backbone | variant | HR@10 | HR@50 | NDCG@10 | NDCG@50 | MRR |
|---|---|---|---|---|---|---|
| SASRec | baseline | 0.333±.004 | 0.553±.005 | 0.234±.003 | 0.282±.003 | 0.218±.003 |
| SASRec | **temporal** | 0.330±.007 | **0.570±.004** | 0.236±.003 | **0.288±.003** | **0.221±.003** |
| HSTU | baseline | 0.318±.008 | 0.564±.007 | 0.226±.005 | 0.280±.005 | 0.213±.005 |
| HSTU | temporal | 0.294±.010 | 0.525±.006 | 0.208±.006 | 0.259±.005 | 0.196±.005 |

Δ% (temporal vs baseline, last-10 mean):
- **SASRec: HR@50 +3.0%, NDCG@50 +2.2%, MRR +1.7%, NDCG@10 +0.5%; HR@10 −0.9%, HR@200 −0.2%.**
- **HSTU: all metrics −7% to −8%** (HR@10 −7.5%, NDCG@10 −8.2%, MRR −8.1%).

### Conclusion

1. **On SASRec the temporal encoder gives a small but consistent gain** on the
   broader-cutoff / ranking metrics (HR@50, NDCG@50, MRR; ~2–3× the run std), and
   is flat at top-10. SASRec has no built-in time mechanism, so this is the clean
   test — and it confirms explicit time features help a temporally-blind model.
2. **On HSTU the temporal encoder hurts (~−8% everywhere).** HSTU already models
   inter-event time deltas in attention (`RelativeBucketedTimeAndPositionBasedBias`),
   so an extra input-side temporal embedding is redundant — it adds capacity that
   overfits the tiny train set. The NDCG@10 curve shows HSTU-temporal degrading
   over training while HSTU-baseline stays flat.

This explains why the original "RoPE" attempt never showed a gain: it was applied
to **HSTU**, which already handles time, *and* it only encoded hour-of-day. The
right place for an explicit temporal encoder is a backbone without a time
mechanism (SASRec), where it does help.

### Caveats / next steps
- Single seed; the SASRec gains (~+2–3%) are modest and only a couple× the
  within-run std. A 3-seed repeat would firm up significance. The HSTU
  degradation (~−8%) is well outside noise and robust.
- KuaiRec `small_matrix` is tiny (1270 train users) and dense; results may differ
  on `big_matrix` or with longer sequences.
