#!/usr/bin/env python3
"""Corrected HSTU vs SASRec on KuaiRec small_matrix, from the committed tfevents.
Uses the FULL per-epoch eval (eval_epoch/*), last-10-epoch mean averaged across
the committed runs per (batch, negatives) config. Replaces the earlier figure,
whose numbers came from single-batch in-loop eval peaks (inflated ~1.4-1.5x).
"""
import glob, re, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as E
METS=['hr@10','hr@50','hr@200','ndcg@10','ndcg@50','mrr']
agg={}
for d in sorted(glob.glob('exps/kuai_video-l100/*2025*/')):
    name=d.split('/')[-2]
    model='HSTU' if name.startswith('HSTU') else 'SASRec'
    neg=re.search(r'-n(\d+)-',name).group(1); bat=re.search(r'-b(\d+)-lr',name).group(1)
    f=glob.glob(d+'*tfevents*'); a=E(f[0]); a.Reload(); tags=a.Tags()['scalars']
    if 'eval_epoch/ndcg@10' not in tags: continue
    row={}; ok=True
    for m in METS:
        t='eval_epoch/'+m
        if t not in tags: ok=False; break
        row[m]=np.array([x.value for x in a.Scalars(t)])[-10:].mean()
    if ok: agg.setdefault((f'b{bat}',f'n{neg}'),{}).setdefault(model,[]).append(row)
cfgs=sorted(agg, key=lambda c:(c[0],c[1]))
labels=[f'{b}/{n}' for b,n in cfgs]
fig,axes=plt.subplots(2,3,figsize=(15,8)); axes=axes.flatten()
col={'HSTU':'#4C72B0','SASRec':'#C44E52'}
for ax,m in zip(axes,METS):
    x=np.arange(len(cfgs)); w=0.38
    for off,model in zip([-w/2,w/2],['HSTU','SASRec']):
        ys=[np.mean([r[m] for r in agg[c].get(model,[{m:np.nan}])]) for c in cfgs]
        es=[np.std([r[m] for r in agg[c].get(model,[{m:np.nan}])]) if model in agg[c] else 0 for c in cfgs]
        bars=ax.bar(x+off,ys,w,yerr=es,capsize=3,label=model,color=col[model])
        for rect,y in zip(bars,ys):
            if not np.isnan(y): ax.text(rect.get_x()+rect.get_width()/2,y,f'{y:.3f}',ha='center',va='bottom',fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=8); ax.set_title(m.upper())
    ax.set_ylabel('score'); ax.grid(axis='y',alpha=0.3); ax.legend(fontsize=8)
fig.suptitle('KuaiRec small_matrix: HSTU vs SASRec (FULL per-epoch eval, last-10 mean +/- std over committed runs)\n'
             'config = batch/negatives ; higher = better', fontsize=12)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig('plots/kuai_video-l100_metrics_comparison.png',dpi=130)
print('wrote plots/kuai_video-l100_metrics_comparison.png')
