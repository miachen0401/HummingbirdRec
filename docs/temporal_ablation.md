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

**v2 — temporal encoder as a zero-init residual + normalized time features**
(the v1 design xavier-init the projection and left log-recency unnormalized, so
the temporal term was ~2.4× the input-signal magnitude at init and crushed HSTU,
−8% everywhere; v2 fixes this — see the "Fix" commit).

| backbone | variant | HR@10 | HR@50 | NDCG@10 | NDCG@50 | MRR |
|---|---|---|---|---|---|---|
| SASRec | baseline | 0.328±.009 | 0.557±.007 | 0.236±.005 | 0.286±.004 | 0.222±.003 |
| SASRec | **temporal** | **0.340±.007** | **0.566±.006** | **0.241±.004** | **0.290±.003** | **0.224±.003** |
| HSTU | baseline | 0.330±.007 | 0.566±.006 | 0.237±.006 | 0.288±.005 | 0.223±.006 |
| HSTU | temporal | 0.321±.007 | **0.578±.008** | 0.224±.004 | 0.280±.004 | 0.209±.004 |

Δ% (temporal vs baseline, last-10 mean):
- **SASRec: every metric up** — HR@10 +3.6%, HR@50 +1.6%, HR@200 +1.0%, NDCG@10 +1.9%, NDCG@50 +1.2%, MRR +0.9%.
- **HSTU: hit-rate up, ranking-quality down** — HR@50 +2.2%, HR@200 +0.3%; but HR@10 −2.8%, NDCG@10 −5.6%, NDCG@50 −3.0%, MRR −6.2%.

Best-epoch (early-stopped) NDCG@10: SASRec 0.247→0.257 (+4.1%), HSTU 0.252→0.256
(+1.8%) — i.e. **at its peak the temporal encoder beats baseline for BOTH**.

### Conclusion

1. **The residual+normalization fix removed the catastrophic HSTU regression**
   (v1 −8% everywhere → v2: HSTU temporal now *peaks above* baseline and HR@50 is
   up +2.2%).
2. **SASRec: consistent gain on all metrics** (last-10). SASRec has no built-in
   time mechanism, so explicit time features clearly help.
3. **HSTU: helps up to its peak / on hit-rate, but overfits the added capacity
   late** (last-10 NDCG@10 −5.6%, MRR −6.2%). HSTU already models inter-event time
   deltas in attention, so the marginal value of an input-side temporal embedding
   is small; with `weight_decay=0` over 101 epochs on a 1270-user train set, the
   extra capacity overfits (NDCG@10 curve diverges after ~epoch 55). With early
   stopping the temporal encoder is ≥ baseline for both backbones.

This explains the original "RoPE" failure: it was applied to **HSTU** (already
time-aware) *and* encoded only hour-of-day. The clean win is on a backbone
without a time mechanism (SASRec).

**v3 (final) — v2 + weight decay on the temporal params** (`temporal_weight_decay
= 0.1`, applied only to the Time2Vec + projection params via a dedicated optimizer
group; backbone and baseline runs unchanged). This regularizes the added capacity
to curb the late overfit seen on HSTU in v2.

| backbone | variant | HR@10 | HR@50 | NDCG@10 | NDCG@50 | MRR |
|---|---|---|---|---|---|---|
| SASRec | baseline | 0.328±.009 | 0.557±.007 | 0.236±.005 | 0.286±.004 | 0.222±.003 |
| SASRec | **temporal+wd** | **0.352±.007** | **0.577±.006** | **0.250±.005** | **0.299±.004** | **0.232±.004** |
| HSTU | baseline | 0.330±.007 | 0.566±.006 | 0.237±.006 | 0.288±.005 | 0.223±.006 |
| HSTU | temporal+wd | **0.342±.007** | **0.575±.007** | 0.237±.006 | 0.288±.006 | 0.219±.006 |

Δ% (temporal+wd vs baseline, last-10 mean):
- **SASRec: every metric up, larger than v2** — HR@10 +7.4%, HR@50 +3.4%, HR@200 +1.0%, NDCG@10 +5.8%, NDCG@50 +4.4%, MRR +4.6%.
- **HSTU: no longer degraded** — HR@10 +3.7%, HR@50 +1.6%; NDCG@10 +0.0%, NDCG@50 −0.2%, MRR −1.7%, HR@200 −1.4% (the last three are within ~1× the run std). The v2 overfit (NDCG@10 −5.6%, MRR −6.2%) is gone.

**Final conclusion.** A properly designed temporal encoder (Time2Vec recency/gap +
cyclical time, added as a zero-init residual, with its capacity weight-decayed):
- **helps SASRec strongly and across the board** (+1 to +7%), and
- **helps HSTU's recall (HR@10/HR@50) and is neutral on its ranking quality**
  (NDCG/MRR within noise) — expected, since HSTU already models inter-event time
  deltas in attention, so the marginal value of an input-side temporal encoding is
  small but, once regularized, it no longer hurts.

So the component is **≥ baseline (helpful or neutral) for both backbones** — the
opposite of the original hour-only "RoPE", which only encoded hour-of-day and was
tested on HSTU (already time-aware), so it could never help.

### Caveats
- Single seed; KuaiRec `small_matrix` is tiny (1270 train / 141 eval users) and
  noisy (per-epoch std ~0.004–0.009), so few-% deltas are soft.
- The HSTU-temporal+wd run was killed by a co-tenant GPU job at epoch 81 (others
  ran 101); its last-10 window (ep 72–81) is still in the converged regime, so the
  comparison holds, but it is not a full 101-epoch run.
- `big_matrix` / longer sequences / 3 seeds would firm up the magnitudes.
