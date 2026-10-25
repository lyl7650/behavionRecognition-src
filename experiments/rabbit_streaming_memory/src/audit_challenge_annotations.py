from __future__ import annotations
import json,sys
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
LAB=HERE.parent
ROOT=LAB.parents[1]
sys.path.insert(0,str(HERE))
from run_baselines import import_run_test

def main():
 pairs=import_run_test().find_rabbit_pairs()
 transitions=Counter(); segment_counts=Counter(); videos_with=Counter()
 transition_rows=[]
 for video,annotation in pairs:
  payload=json.loads(annotation.read_text(encoding='utf-8'))
  segments=payload['segments']; states=[str(s['state']).upper() for s in segments]
  for state in states: segment_counts[state]+=1
  seen=set()
  for left,right in zip(segments,segments[1:]):
   pair=(str(left['state']).upper(),str(right['state']).upper())
   transitions[pair]+=1;seen.add(pair)
   transition_rows.append({'video':video.name,'from':pair[0],'to':pair[1],'boundary_ms':left['end_ms'],'from_duration_ms':left['end_ms']-left['start_ms'],'to_duration_ms':right['end_ms']-right['start_ms']})
  for pair in seen: videos_with[pair]+=1
 challenge={'eating_to_active':('EATING','ACTIVE'),'active_to_resting':('ACTIVE','RESTING'),'resting_to_active':('RESTING','ACTIVE')}
 report={'video_count':len(pairs),'segment_count':sum(segment_counts.values()),'videos_with_multiple_segments':sum(1 for video,annotation in pairs if len(json.loads(annotation.read_text(encoding='utf-8'))['segments'])>1),'segment_state_counts':dict(segment_counts),'transition_counts':{f'{a}->{b}':n for (a,b),n in transitions.items()},'challenge_support':{name:{'transition_count':transitions[pair],'videos':videos_with[pair],'pair':list(pair)} for name,pair in challenge.items()},'transition_rows':transition_rows,'interpretation':'The current annotations mark coarse state segments only; they do not mark head-down bite vs head-up chewing or short within-active pauses. Those exact challenge subevents cannot be validated as dedicated subsets without finer annotations.'}
 out=LAB/'outputs/annotation_challenge_audit.json';out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='transition_rows'},indent=2,ensure_ascii=False))
if __name__=='__main__':main()
