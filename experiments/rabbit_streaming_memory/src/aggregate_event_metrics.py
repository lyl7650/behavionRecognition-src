from __future__ import annotations
import csv, json
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/"outputs"
RUNS=["baseline_oct20_i0","baseline_oct20_i1","baseline_oct20_i2_fsm","baseline_oct20_i3_forced_reset8","baseline_jun24_i0","baseline_jun24_i1"]
CLASSES=["ACTIVE","EATING","RESTING","DRINKING"]

def main():
 rows=[]
 for run in RUNS:
  totals={name:{"tp":0,"fp":0,"fn":0} for name in CLASSES}
  for path in (OUT/run).glob("*/metrics.json"):
   payload=json.loads(path.read_text(encoding="utf-8"))
   by_class=payload["p0_quality"]["event_detection_f1"]["by_class"]
   for name in CLASSES:
    if name not in by_class: continue
    for key in ("tp","fp","fn"): totals[name][key]+=int(by_class[name][key])
  for name,count in totals.items():
   tp,fp,fn=count["tp"],count["fp"],count["fn"]
   precision=tp/(tp+fp) if tp+fp else None
   recall=tp/(tp+fn) if tp+fn else None
   f1=2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else 0.0
   rows.append({"run":run,"class":name,**count,"precision":precision,"recall":recall,"f1":f1})
 path=OUT/"event_metrics_by_class.csv"
 with path.open("w",newline="",encoding="utf-8-sig") as handle:
  writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 print(f"wrote {len(rows)} rows: {path}")
if __name__=="__main__":main()
