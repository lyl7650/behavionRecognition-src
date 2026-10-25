from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
LAB=Path(__file__).resolve().parents[1]

def main():
 audit=json.loads((LAB/'manifests/audit_full_2026-10-24.json').read_text(encoding='utf-8'))
 report={'audit':str((LAB/'manifests/audit_full_2026-10-24.json').resolve()),'thresholds_seconds':[2,12],'splits':{}}
 for split in ('train','val'):
  rows=[row for row in audit['videos'] if row['split']==split]
  durations=sorted(float(row['duration_seconds']) for row in rows)
  stats={'videos':len(rows),'min_seconds':durations[0],'median_seconds':durations[len(durations)//2],'max_seconds':durations[-1]}
  for threshold in report['thresholds_seconds']:
   short=[row for row in rows if float(row['duration_seconds'])<threshold]
   stats[f'under_{threshold}s']={'count':len(short),'by_class':dict(Counter(row['label'] for row in short))}
  report['splits'][split]=stats
 path=LAB/'outputs/duration_distribution.json';path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
