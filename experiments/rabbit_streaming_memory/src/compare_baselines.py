from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument("--i0",type=Path,required=True);p.add_argument("--i1",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 r0=json.loads((a.i0/"metrics_summary.json").read_text(encoding="utf-8"));r1=json.loads((a.i1/"metrics_summary.json").read_text(encoding="utf-8"))
 keys=["time_iou_macro","event_f1_tiou_05","transition_f1","p95_action_ms_mean_per_video"]
 rows=[]
 for k in keys:
  v0=r0["aggregate"].get(k);v1=r1["aggregate"].get(k);rows.append({"metric":k,"i0":v0,"i1":v1,"i1_minus_i0":None if v0 is None or v1 is None else v1-v0})
 out={"i0":str(a.i0.resolve()),"i1":str(a.i1.resolve()),"metrics":rows}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding="utf-8");print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
