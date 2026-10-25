from __future__ import annotations
import json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parents[1]/'outputs'
PAIRS=[
 ('T0_seed44_window_vs_Oct20_I0','trained_t0_seed44_i0_window','baseline_oct20_i0'),
 ('T0_seed44_stream_vs_Oct20_I1','trained_t0_seed44_i1_stream','baseline_oct20_i1'),
 ('T1_seed42_stream_vs_Oct20_I1','trained_t1_seed42_i1_stream','baseline_oct20_i1'),
 ('T2_seed43_stream_vs_Oct20_I1','trained_t2_seed43_i1_stream','baseline_oct20_i1'),
 ('T0_seed44_stream_vs_window','trained_t0_seed44_i1_stream','trained_t0_seed44_i0_window'),
]
METRICS=['time_iou_macro','event_f1_tiou_05','transition_f1']
def load(name):
 rows=json.loads((OUT/name/'metrics_summary.json').read_text(encoding='utf-8'))['videos']
 return {r['video']:r for r in rows}
def main():
 rng=np.random.default_rng(20261024);out={'method':'paired bootstrap over 30 videos, 20000 resamples, percentile 95% CI; descriptive because videos may be dependent','comparisons':[]}
 for label,left_name,right_name in PAIRS:
  left,right=load(left_name),load(right_name); common=sorted(set(left)&set(right));item={'comparison':label,'candidate':left_name,'reference':right_name,'paired_videos':len(common),'metrics':{}}
  for metric in METRICS:
   diffs=np.array([right[v][metric]-left[v][metric] for v in common if left[v].get(metric) is not None and right[v].get(metric) is not None],dtype=float)
   samples=rng.integers(0,len(diffs),size=(20000,len(diffs)));means=diffs[samples].mean(axis=1)
   item['metrics'][metric]={'n':len(diffs),'reference_minus_candidate_mean':float(diffs.mean()),'ci95_percentile':[float(v) for v in np.quantile(means,[.025,.975])],'reference_better':int((diffs>0).sum()),'ties':int((diffs==0).sum()),'candidate_better':int((diffs<0).sum())}
  out['comparisons'].append(item)
 p=OUT/'trained_checkpoint_paired_bootstrap_2026-10-24.json'
 p.write_text(json.dumps(out,indent=2),encoding='utf-8')
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
