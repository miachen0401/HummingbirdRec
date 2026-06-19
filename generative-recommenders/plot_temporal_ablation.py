#!/usr/bin/env python3
"""Plot the KuaiRec temporal-encoding ablation: baseline vs Time2Vec temporal,
for SASRec and HSTU backbones.

Reads the per-epoch eval scalars (``eval_epoch/<metric>``) from the tfevents
written under ``exps/kuai_video-l100/`` by the 4 ablation runs, selects each
run's metrics at its best-NDCG@10 epoch (the same early-stopping rule for all
runs, to avoid cherry-picking), and emits:
  1) plots/temporal_ablation_metrics.png  — grouped bars per metric
  2) plots/temporal_ablation_ndcg10_curve.png — NDCG@10 vs epoch (4 runs)

Usage:  ../.venv/bin/python plot_temporal_ablation.py
"""

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

EXPS_GLOB = "exps/kuai_video-l100/*"
OUT_DIR = "plots"
METRICS = ["hr@10", "hr@50", "hr@200", "ndcg@10", "ndcg@50", "mrr"]
SELECT_BY = "ndcg@10"  # epoch selection rule (same for all runs)


def classify(path: str):
    name = os.path.basename(path)
    backbone = "SASRec" if "SASRec" in name else ("HSTU" if "HSTU" in name else None)
    if "pp_temporal" in name:
        variant = "temporal"
    elif "pp_learnable_positional" in name:
        variant = "baseline"
    else:
        variant = None
    return backbone, variant


def load_curves(path: str):
    acc = EventAccumulator(path, size_guidance={"scalars": 0})
    acc.Reload()
    tags = set(acc.Tags().get("scalars", []))
    out = {}
    for m in METRICS:
        tag = f"eval_epoch/{m}"
        if tag in tags:
            ev = acc.Scalars(tag)
            out[m] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))
    return out


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    runs = {}  # (backbone, variant) -> curves
    for path in glob.glob(EXPS_GLOB):
        if not os.path.isdir(path):
            continue
        backbone, variant = classify(path)
        if backbone is None or variant is None:
            continue
        curves = load_curves(path)
        if SELECT_BY in curves:
            runs[(backbone, variant)] = curves
            print(f"loaded {backbone}/{variant}: {len(curves[SELECT_BY][1])} epochs <- {os.path.basename(path)}")

    if not runs:
        raise SystemExit(f"No runs found under {EXPS_GLOB}")

    # Summary per run. The eval set here is tiny (141 users) so the per-epoch
    # metric is very noisy and "max over epochs" cherry-picks noise spikes.
    # PRIMARY = mean over the last LAST_K epochs (converged, stable) with std;
    # best-epoch and final are kept for reference only.
    LAST_K = 10
    summary = {}  # (backbone, variant) -> {metric: mean, metric+"_std", ...}
    for key, curves in runs.items():
        steps, sel = curves[SELECT_BY]
        best_epoch = int(steps[int(np.argmax(sel))])
        vals = {"_best_epoch": best_epoch}
        for m in METRICS:
            if m not in curves:
                continue
            cs, cv = curves[m]
            tail = cv[-LAST_K:]
            vals[m] = float(np.mean(tail))            # PRIMARY: last-K mean
            vals[m + "_std"] = float(np.std(tail))
            vals[m + "_best"] = float(np.max(cv))
            j = np.where(cs == best_epoch)[0]
            vals[m + "_atbest"] = float(cv[j[0]]) if len(j) else float("nan")
            vals[m + "_final"] = float(cv[-1])
        summary[key] = vals

    # ---- Figure 1: grouped bars per metric ----
    backbones = [b for b in ["SASRec", "HSTU"] if any(k[0] == b for k in runs)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    colors = {"baseline": "#4C72B0", "temporal": "#C44E52"}
    for ax, m in zip(axes, METRICS):
        x = np.arange(len(backbones))
        w = 0.36
        for off, variant in zip([-w / 2, w / 2], ["baseline", "temporal"]):
            ys = [summary.get((b, variant), {}).get(m, np.nan) for b in backbones]
            es = [summary.get((b, variant), {}).get(m + "_std", 0.0) for b in backbones]
            bars = ax.bar(x + off, ys, w, yerr=es, capsize=3, label=variant,
                          color=colors[variant])
            for rect, y in zip(bars, ys):
                if not np.isnan(y):
                    ax.text(rect.get_x() + rect.get_width() / 2, y, f"{y:.3f}",
                            ha="center", va="bottom", fontsize=8)
        # delta annotation
        for xi, b in enumerate(backbones):
            base = summary.get((b, "baseline"), {}).get(m, np.nan)
            temp = summary.get((b, "temporal"), {}).get(m, np.nan)
            if not (np.isnan(base) or np.isnan(temp)) and base != 0:
                d = (temp - base) / base * 100
                ax.text(xi, max(base, temp) * 1.06, f"{d:+.1f}%",
                        ha="center", fontsize=8, color="green" if d > 0 else "red")
        ax.set_xticks(x)
        ax.set_xticklabels(backbones)
        ax.set_title(m.upper())
        ax.set_ylabel("score")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        "KuaiRec temporal-encoding ablation: baseline (positional) vs Time2Vec temporal\n"
        "(bars = mean over last 10 epochs, error bars = std; Δ% on the means; higher = better)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    f1 = os.path.join(OUT_DIR, "temporal_ablation_metrics.png")
    fig.savefig(f1, dpi=130)
    print("wrote", f1)

    # ---- Figure 2: NDCG@10 convergence ----
    fig2, ax = plt.subplots(figsize=(9, 5.5))
    styles = {"baseline": "-", "temporal": "--"}
    bbcolor = {"SASRec": "#55A868", "HSTU": "#8172B3"}
    for (backbone, variant), curves in sorted(runs.items()):
        steps, vals = curves[SELECT_BY]
        ax.plot(steps, vals, styles[variant], color=bbcolor[backbone],
                label=f"{backbone} / {variant}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("NDCG@10 (eval)")
    ax.set_title("KuaiRec NDCG@10 over training: temporal vs baseline (SASRec, HSTU)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig2.tight_layout()
    f2 = os.path.join(OUT_DIR, "temporal_ablation_ndcg10_curve.png")
    fig2.savefig(f2, dpi=130)
    print("wrote", f2)

    # ---- text summary ----
    print("\n=== last-10-epoch mean +/- std (PRIMARY, stable) ===")
    hdr = ["backbone", "variant"] + METRICS
    print("  ".join(f"{h:>16}" for h in hdr))
    for b in backbones:
        for v in ["baseline", "temporal"]:
            s = summary.get((b, v))
            if s is None:
                continue
            row = [b, v] + [f"{s.get(m, float('nan')):.4f}±{s.get(m+'_std', 0):.3f}" for m in METRICS]
            print("  ".join(f"{c:>16}" for c in row))
    print("\n=== reference: best-epoch / final-epoch NDCG@10 ===")
    for b in backbones:
        for v in ["baseline", "temporal"]:
            s = summary.get((b, v))
            if s is None:
                continue
            print(f"{b:>8} {v:>9}: best_ep={s['_best_epoch']:>3} "
                  f"ndcg@10 best={s.get('ndcg@10_best', float('nan')):.4f} "
                  f"final={s.get('ndcg@10_final', float('nan')):.4f} "
                  f"last10={s.get('ndcg@10', float('nan')):.4f}")
    # Δ% (temporal vs baseline) on last-10 means
    print("\n=== Δ% temporal vs baseline (last-10 mean) ===")
    for b in backbones:
        sb, st = summary.get((b, "baseline")), summary.get((b, "temporal"))
        if not (sb and st):
            continue
        deltas = []
        for m in METRICS:
            base = sb.get(m, np.nan)
            if base and not np.isnan(base):
                deltas.append(f"{m} {((st.get(m, np.nan)-base)/base*100):+.1f}%")
        print(f"{b:>8}: " + "  ".join(deltas))


if __name__ == "__main__":
    main()
