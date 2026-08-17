"""BINARY NONE-DETECTOR — first-principles response to the residual decomposition.

61% of the champion's 56 errors are gold-None rows leaked into a stance pool (17 in predF,
17 in predA). Recovering ~10+10 of them clears #3. Every prior encoder rule-out in this project
was for the 3-CLASS stance task, where only 2 seen targets make target-generalisation impossible.
This asks a different and plausibly target-AGNOSTIC question: "would two annotators fail to agree
that a stance is present?" -- a property of the TEXT (vagueness, questions, brevity, topic drift)
rather than of the target's semantics.

Trains on train_track_2 (2 seen targets), validates on dev_track_2 (1 UNSEEN target) = an honest
leave-one-target-out read, and scores the blind test.
"""
import os, sys, json, argparse
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

D='<datasets>/MawqifV2'
ap=argparse.ArgumentParser()
ap.add_argument('--model',default='UBC-NLP/MARBERTv2')
ap.add_argument('--epochs',type=int,default=4); ap.add_argument('--lr',type=float,default=2e-5)
ap.add_argument('--bs',type=int,default=16); ap.add_argument('--maxlen',type=int,default=128)
ap.add_argument('--seed',type=int,default=42); ap.add_argument('--out',required=True)
a=ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
def load(f):
    return pd.read_csv(f'{D}/{f}',keep_default_na=False,encoding='utf-8-sig')
tr=load('train_track_2.csv'); dv=load('dev_track_2.csv')
te=pd.read_csv('test_norm.csv',keep_default_na=False,encoding='utf-8-sig')
tcol=lambda d:'text' if 'text' in d.columns else 'tweet_text'
def mk(d,lab=True):
    X=[(str(r['target']),str(r[tcol(d)])) for _,r in d.iterrows()]
    y=[1 if str(r['stance']).strip()=='None' else 0 for _,r in d.iterrows()] if lab else [0]*len(d)
    return X,np.array(y)
Xtr,ytr=mk(tr); Xdv,ydv=mk(dv); Xte,_=mk(te,False)
print(f'train {len(Xtr)} (None {ytr.sum()}) | dev {len(Xdv)} (None {ydv.sum()}) | test {len(Xte)}',flush=True)
tok=AutoTokenizer.from_pretrained(a.model)
class DS(Dataset):
    def __init__(s,X,y): s.X,s.y=X,y
    def __len__(s): return len(s.X)
    def __getitem__(s,i):
        e=tok(s.X[i][0],s.X[i][1],truncation=True,max_length=a.maxlen,padding='max_length',return_tensors='pt')
        return {k:v[0] for k,v in e.items()}|{'labels':torch.tensor(s.y[i])}
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m=AutoModelForSequenceClassification.from_pretrained(a.model,num_labels=2).to(dev)
# class weights: None is the minority class
w=torch.tensor([1.0, float((ytr==0).sum())/max(1,(ytr==1).sum())],dtype=torch.float).to(dev)
print('class weight for None:',w[1].item(),flush=True)
dl=DataLoader(DS(Xtr,ytr),batch_size=a.bs,shuffle=True,drop_last=False)
opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=0.01)
sch=get_linear_schedule_with_warmup(opt,int(0.1*len(dl)*a.epochs),len(dl)*a.epochs)
lossf=torch.nn.CrossEntropyLoss(weight=w)
for ep in range(a.epochs):
    m.train(); tot=0
    for b in dl:
        b={k:v.to(dev) for k,v in b.items()}; lb=b.pop('labels')
        out=m(**b); loss=lossf(out.logits,lb)
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
        opt.step(); sch.step(); opt.zero_grad(); tot+=loss.item()
    print(f'  epoch {ep+1} loss {tot/len(dl):.4f}',flush=True)
@torch.no_grad()
def pred(X):
    m.eval(); P=[]
    for i in range(0,len(X),64):
        e=tok([x[0] for x in X[i:i+64]],[x[1] for x in X[i:i+64]],truncation=True,
              max_length=a.maxlen,padding=True,return_tensors='pt').to(dev)
        P.append(torch.softmax(m(**e).logits,-1)[:,1].cpu().numpy())
    return np.concatenate(P)
pdv,pte=pred(Xdv),pred(Xte)
od=os.environ.get('EXP_OUTPUT_DIR','.'); os.makedirs(od,exist_ok=True)
np.savez_compressed(f'{od}/{a.out}',p_none_dev=pdv,p_none_test=pte,y_dev=ydv)
# HONEST UNSEEN-TARGET READ: precision@k among dev rows whose GOLD is a stance (the pool we deploy on)
o=np.argsort(-pdv); res={}
for k in [10,20,30,40,60,80]:
    res[f'prec@{k}']=float(ydv[o[:k]].mean())
res['dev_none_base']=float(ydv.mean()); res['auc_proxy']=float(np.mean([ydv[o[:i]].mean() for i in [20,40,80]]))
print('DEV (UNSEEN TARGET) precision@k for None:',json.dumps(res),flush=True)
json.dump(res,open(f'{od}/{a.out}_metrics.json','w'),indent=1)
print(f'[RESULT] none_detector seed={a.seed} prec@20={res["prec@20"]:.3f} prec@40={res["prec@40"]:.3f} base={res["dev_none_base"]:.3f}')
