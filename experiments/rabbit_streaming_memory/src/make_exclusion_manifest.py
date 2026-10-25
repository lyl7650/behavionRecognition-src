"""Create a read-only derived manifest excluding explicitly listed invalid files.

This does not modify or move source videos. It is only for a user-approved
experiment run after the audit identifies damaged media.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

HERE=Path(__file__).resolve().parent
LAB=HERE.parent

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--audit', type=Path, default=LAB/'manifests/audit.json')
    p.add_argument('--output', type=Path, default=LAB/'manifests/training_exclusion.json')
    args=p.parse_args()
    report=json.loads(args.audit.read_text(encoding='utf-8'))
    invalid=[row['path'] for row in report.get('invalid_videos', [])]
    payload={
      'source_audit': str(args.audit.resolve()),
      'source_dataset_root': report.get('dataset_root'),
      'policy': 'exclude_only_the_audit_list; do_not_modify_source_data',
      'excluded_paths': invalid,
      'excluded_count': len(invalid),
      'requires_explicit_user_approval': True,
      'approved': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(payload, indent=2, ensure_ascii=False))

if __name__=='__main__': main()
