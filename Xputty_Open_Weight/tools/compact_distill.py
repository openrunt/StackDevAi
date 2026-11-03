"""Build a genuinely smaller pretrained student and compare supervised/distilled recovery."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vox.compact_distill import train

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','dataset','output','cache-file'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--steps',type=int,default=128);p.add_argument('--keep',type=int,default=5)
    p.add_argument('--alpha',type=float,default=.5);p.add_argument('--temperature',type=float,default=2.)
    p.add_argument('--top-k',type=int,default=64);p.add_argument('--context',type=int,default=384)
    p.add_argument('--train-samples',type=int,default=128);p.add_argument('--eval-samples',type=int,default=16)
    p.add_argument('--eval-every',type=int,default=16);p.add_argument('--resume',action='store_true')
    print(json.dumps(train(**vars(p.parse_args())),indent=2))
