"""실험 D — 시드 반복. native B / aligned B, 3겹 x 추가 2시드."""
import os, sys, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data, cdld_core as CC, cdld_a as CA
import paths
CD = paths.WORK + os.sep; CK=CD+'ck_seed/'; os.makedirs(CK, exist_ok=True)
BUDGET=float(sys.argv[1]) if len(sys.argv)>1 else 1e9

def auc(y,p):
    o=np.argsort(p); y=y[o]; n1=y.sum(); n0=len(y)-n1
    if n1==0 or n0==0: return float('nan')
    r=np.arange(1,len(y)+1)
    return float((r[y>.5].sum()-n1*(n1+1)/2)/(n1*n0))

def main():
    d=data.prep(None, ward=False); D=data.tensors(d)
    sizes=list(d['sizes']); ncov=d['cov'].shape[1]; y=d['y']
    t0=time.time()
    for kind in ('natB','aliB'):
        for s in (1,2):
            for f in range(3):
                key=f'base_seed{s}_f{f}_{kind}'
                if os.path.exists(CK+key+'.json'): continue
                if time.time()-t0+340>BUDGET: return
                tr=np.flatnonzero(d['fold']!=f); te=np.flatnonzero(d['fold']==f)
                torch.manual_seed(50000+s*1000+f)
                if kind=='natB':
                    m=CC.Finder(sizes,ncov,n_nurse=int(d['n_nurse']),n_drug=int(d['n_drug']),drug_latent=True)
                else:
                    m=CA.AlignedFinder(sizes,ncov,int(d['n_nurse']),int(d['n_drug']),'B')
                t1=time.time()
                CC.fit(m,D,tr,te,cycles=6,sub_epochs=1,bs=8192,seed=42+f,cyclic=False)
                p=CC.predict(m,D,te)
                res=dict(model=kind,seed=s,fold=f,kind=kind,arm=f'seed{s}',tag='base',
                         sse=float(((y[te]-p)**2).sum()),
                         sstot=float(((y[te]-y[tr].mean())**2).sum()),
                         auc=auc(y[te],p),secs=round(time.time()-t1,1))
                json.dump(res,open(CK+key+'.json','w'))
                print(f'  ✓ seed {key}  BSS {1-res["sse"]/res["sstot"]:+.5f}  {res["secs"]:.0f}s',flush=True)
if __name__=='__main__': main()
