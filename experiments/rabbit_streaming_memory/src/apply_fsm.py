"""Apply isolated FSM decoding to an existing raw baseline run."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from rabbit_fsm import RabbitFSM

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--fps',type=float,default=2.0);p.add_argument('--scale-thresholds',action='store_true');a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 for video_dir in sorted(x for x in a.run_dir.iterdir() if x.is_dir()):
  trace=video_dir/'track_trace.jsonl';summary=video_dir/'summary.json'
  if not trace.exists() or not summary.exists():continue
  rows=[json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines() if line.strip()];fsm=RabbitFSM(a.fps,5.0,a.scale_thresholds)
  out= a.output_dir/video_dir.name;out.mkdir(parents=True,exist_ok=True); decoded=[]
  for row in rows:
   candidate=row['raw_argmax'].upper();state,info=fsm.update(candidate,float(row['video_time_s']));row['fusion']={'final_state':state};row['fsm']=info;decoded.append(row)
  with (out/'track_trace.jsonl').open('w',encoding='utf-8') as h:
   for row in decoded:h.write(json.dumps(row,ensure_ascii=False)+'\n')
  payload=json.loads(summary.read_text(encoding='utf-8'));payload['fsm']='isolated_rabbit_fsm';payload['source_trace']=str(trace);(out/'summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
 print(json.dumps({'videos':len([x for x in a.output_dir.iterdir() if x.is_dir()]),'output':str(a.output_dir)},ensure_ascii=False))
if __name__=='__main__':main()
