from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
LAB=HERE.parent
ROOT=LAB.parents[1]
sys.path.insert(0,str(HERE))
from data_selection import load_split_exclusions
from stream_dataset import StreamingVideoDataset,scan_videos
from timm_gru_student import TimmGRUStudent

CLASSES=['resting','active','eating','drinking']
CHECKPOINTS={f'{mode}_seed{seed}': LAB / 'outputs/checkpoints' / f'{mode}_seed{seed}' / 'model_ema.pth' for mode in ['t0','t1','t2'] for seed in [42,43,44]}

def main():
 p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--output',type=Path,default=LAB/'outputs/selected_validation_confusion_2026-10-24');args=p.parse_args()
 if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 device=torch.device(args.device); data_root=ROOT/'dataset/rabbit/2026-10-22'
 excluded_train,excluded_val=load_split_exclusions(LAB/'manifests/audit_final_2026-10-24.json',LAB/'manifests/training_selection_approved.json',data_root)
 specs=scan_videos(data_root/'val',CLASSES,excluded_val); per_class=12; selected=[]
 for ci in range(4):selected.extend([s for s in specs if s.label==ci][:per_class])
 report={'video_count':len(selected),'class_names':CLASSES,'models':{}}
 args.output.mkdir(parents=True,exist_ok=True)
 for name,path in CHECKPOINTS.items():
  model=TimmGRUStudent('repvit_m1',4,hidden_dim=256,pretrained=False).to(device).eval();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
  mode='window' if name.startswith('t0') else ('uniform' if name.startswith('t1') else 'step')
  ds=StreamingVideoDataset(selected,block_seconds=2.0 if mode=='window' else 12.0,burn_in_seconds=0 if mode=='window' else 5.0,imgsz=224,fps=8,is_train=False,pad_short=(mode=='window'))
  confusion=[[0]*4 for _ in range(4)];rows=[];total_loss=0.0
  with torch.inference_mode():
   for i in range(len(ds)):
    batch=ds[i];x=batch['pixel_values'].unsqueeze(0).to(device);y=batch['labels'].unsqueeze(0).to(device)
    if mode=='window': logits=model.forward_seq(x); loss=torch.nn.functional.cross_entropy(logits,y[:,-1]); endpoint=logits.argmax(-1)
    else:
     logits,_=model.forward_stream(x);burn=int(batch['burn_in_frames'])
     if mode=='uniform':loss=torch.nn.functional.cross_entropy(logits[:,burn:].reshape(-1,4),y[:,burn:].reshape(-1))
     else:loss=torch.nn.functional.cross_entropy(logits[:,-1],y[:,-1])
     endpoint=logits[:,-1].argmax(-1)
    truth=int(y[0,-1]);pred=int(endpoint[0]);confusion[truth][pred]+=1;total_loss+=float(loss)
    rows.append({'video':Path(batch['video_path']).name,'truth':CLASSES[truth],'prediction':CLASSES[pred]})
  recalls={CLASSES[i]:confusion[i][i]/sum(confusion[i]) if sum(confusion[i]) else None for i in range(4)}
  report['models'][name]={'checkpoint':str(path.resolve()),'mode':mode,'loss':total_loss/len(ds),'accuracy':sum(confusion[i][i] for i in range(4))/len(ds),'confusion_rows_truth_columns_prediction':confusion,'per_class_recall':recalls,'predicted_counts':{CLASSES[j]:sum(confusion[i][j] for i in range(4)) for j in range(4)}}
  with (args.output/f'{name}_validation_predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  del model
 out=args.output/'validation_confusion.json';out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=='__main__':main()


