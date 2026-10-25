"""Apply isolated dashboard candidate fusion/FSM to existing probability traces.

The input probabilities and boxes are reused at 0.5s sampling. This is a
fusion-only comparison, not an exact dashboard model/preprocessing replay.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
LAB=HERE.parent
sys.path.insert(0,str(HERE))
from dashboard_fusion_snapshot import DashboardFusionSnapshot, CLASS_NAMES
SOURCE=Path(r"E:\ActionRecognition\handover\ppc_dashboard\backend\ml\decision\fusion_agent.py")

def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
 return h.hexdigest()

def replay_one(source_dir,dest_dir):
 trace=source_dir/'track_trace.jsonl'
 summary=source_dir/'summary.json'
 rows=[json.loads(x) for x in trace.read_text(encoding='utf-8').splitlines() if x.strip()]
 agent=DashboardFusionSnapshot(track_id=0);out=[];missing=0
 for row in rows:
  bbox=row.get('bbox');time_s=float(row['video_time_s']);pm=row.get('probabilities')
  probs=None if not pm else [float(pm[n.upper()] if n.upper() in pm else pm[n]) for n in CLASS_NAMES]
  if bbox is None:
   missing+=1
   state={"rule_state":None,"speed_px_s":None,"estimated_update_fps":agent.estimated_update_fps,"model_ran":False,"model_prob_used":False,"reset_after_gap":False,"candidate":None,"final_state":agent.state,"fsm":{"previous_state":agent.state,"reason":"no_detection_hold_state","transitioned":False}}
  else:
   box=[bbox['x1'],bbox['y1'],bbox['x2'],bbox['y2']]
   state=agent.update(box,probs,time_s)
  item=dict(row);item['fusion']={'final_state':state['final_state']};item['dashboard_fusion_snapshot']=state;out.append(item)
 dest_dir.mkdir(parents=True,exist_ok=True)
 with (dest_dir/'track_trace.jsonl').open('w',encoding='utf-8') as f:
  for row in out:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 meta=json.loads(summary.read_text(encoding='utf-8'));meta.update({'fusion':'dashboard_rabbit_fusion_snapshot','fusion_source_sha256':sha256(SOURCE),'fusion_scope':'sampled probability/box fusion-only; original YOLO/model crop, 4Hz geometry history and GRU reset outputs are not recomputed','missing_box_count':missing})
 (dest_dir/'summary.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
 return {'video':source_dir.name,'records':len(out),'missing_boxes':missing,'state_counts':{s:sum(r['fusion']['final_state']==s for r in out) for s in ('RESTING','ACTIVE','EATING','DRINKING')}}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 if not SOURCE.is_file():raise FileNotFoundError(SOURCE)
 videos=sorted(x for x in a.run_dir.iterdir() if x.is_dir());results=[replay_one(x,a.output_dir/x.name) for x in videos]
 manifest={'dashboard_source':str(SOURCE),'dashboard_source_sha256':sha256(SOURCE),'replay_scope':'FusionAgent candidate/FSM copied from source and driven by 2Hz experiment traces. Probabilities were generated using experiment temporal-window crop preprocessing; do not interpret as exact production performance.','videos':results}
 a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/'dashboard_snapshot_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
 print(json.dumps({'video_count':len(results),'output_dir':str(a.output_dir),'source_sha256':manifest['dashboard_source_sha256']},indent=2))
if __name__=='__main__':main()
