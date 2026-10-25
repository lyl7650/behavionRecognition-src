from __future__ import annotations
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parents[1]/'outputs'
COMPARISONS=[
 ('oct20_i0_vs_i1','baseline_oct20_i0','baseline_oct20_i1'),
 ('jun24_i0_vs_i1','baseline_jun24_i0','baseline_jun24_i1'),
 ('oct20_i1_vs_forced_reset8','baseline_oct20_i1','baseline_oct20_i3_forced_reset8'),
]
METRICS=['time_iou_macro','event_f1_tiou_05','transition_f1']

def load(name):
 data=json.loads((OUT/name/'metrics_summary.json').read_text(encoding='utf-8'))['videos']
 return {row['video']:row for row in data}

def main():
 rng=np.random.default_rng(20261024)
 result={'method':'paired bootstrap over the 30 videos, 20000 resamples, percentile 95% CI; descriptive only, videos may not be independent','comparisons':[]}
 for label,left_name,right_name in COMPARISONS:
  left,right=load(left_name),load(right_name)
  common=sorted(set(left)&set(right))
  item={'comparison':label,'left':left_name,'right':right_name,'paired_videos':len(common),'metrics':{}}
  for metric in METRICS:
   pairs=[(left[v].get(metric),right[v].get(metric)) for v in common]
   pairs=[(a,b) for a,b in pairs if a is not None and b is not None]
   diffs=np.asarray([b-a for a,b in pairs],dtype=np.float64)
   samples=rng.integers(0,len(diffs),size=(20000,len(diffs)))
   means=diffs[samples].mean(axis=1)
   item['metrics'][metric]={'n':len(diffs),'mean_delta_right_minus_left':float(diffs.mean()),'ci95_percentile':[float(x) for x in np.quantile(means,[.025,.975])],'videos_improved':int((diffs>0).sum()),'videos_tied':int((diffs==0).sum()),'videos_worsened':int((diffs<0).sum())}
  result['comparisons'].append(item)
 path=OUT/'paired_bootstrap_2026-10-24.json';path.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
