import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
"""교정 귀무 기준 전체 집계 — 논문 수치의 단일 출처."""
import json, glob, os, numpy as np, pandas as pd
CK=os.path.join(paths.WORK, 'cdld/ck/'); V1=os.path.join(paths.WORK, 'cdld/ck_v1/')
def load(tag, d=CK):
    r=[json.load(open(f)) for f in glob.glob(f'{d}{tag}_*.json')]
    if not r: return None
    x=pd.DataFrame(r); x['r2']=1-x.sse/x.sstot; return x
def g(x,a,k):
    s=x[(x.arm==a)&(x.kind==k)].sort_values('fold'); return s.r2.to_numpy()
def net(tag, maxns=3, d=CK, label=None):
    x=load(tag,d)
    if x is None: return None
    A,B,C=g(x,'real','A'),g(x,'real','B'),g(x,'real','C')
    if len(A)==0 or len(B)!=len(A) or len(C)!=len(A): return None
    sb=[g(x,f'shuffle{k}','B') for k in range(maxns)]
    sc=[g(x,f'shuffle{k}','C') for k in range(maxns)]
    sb=[v for v in sb if len(v)==len(A)]; sc=[v for v in sc if len(v)==len(A)]
    if not sb or not sc: return None
    SB,SC=np.mean(sb,0),np.mean(sc,0)
    m=(B-A)-(SB-A); l=(C-B)-(SC-SB); t=(C-A)-(SC-A)
    return dict(설정=label or tag, 겹=len(A), 귀무=len(sc),
                A=A.mean(), B=B.mean(), C=C.mean(),
                AUC_C=float(x[(x.arm=='real')&(x.kind=='C')].auc.mean()),
                주효과=m.mean(), 주효과sd=m.std(ddof=1),
                잠재=l.mean(), 잠재sd=l.std(ddof=1),
                총=t.mean(), 총sd=t.std(ddof=1),
                비=l.mean()/m.mean(), S=(SC-SB).mean())
def als(path, label):
    if not os.path.exists(path): return None
    a=pd.read_csv(path).set_index('arm')[['A','B','C']].astype(float)
    s=a.loc[[i for i in a.index if i.startswith('shuffle')]].mean(); r=a.loc['real']
    m=(r.B-r.A)-(s.B-r.A); l=(r.C-r.B)-(s.C-s.B); t=(r.C-r.A)-(s.C-r.A)
    return dict(설정=label, 겹=5, 귀무=2, A=r.A, B=r.B, C=r.C, AUC_C=np.nan,
                주효과=m, 주효과sd=np.nan, 잠재=l, 잠재sd=np.nan,
                총=t, 총sd=np.nan, 비=l/m, S=s.C-s.B)
rows=[net('base',label='CDLD 주판'),
      net('wardX',label='CDLD + 병동·병동×약물'),
      net('temporal',2,label='CDLD 시간 분할'),
      net('cap16',1,label='CDLD 잠재 64→16'),
      net('top100',1,label='CDLD 약물 상위 100종'),
      als(os.path.join(paths.WORK, '결과/07_사전등록_none.csv'),'ALS 주판'),
      als(os.path.join(paths.WORK, '결과/09_병동관문.csv'),'ALS 병동 통제')]
t=pd.DataFrame([r for r in rows if r])
t.to_csv(os.path.join(paths.WORK, 'cdld/out_v2_전체집계.csv'), index=False, float_format='%.5f')
pd.set_option('display.width',200)
print(t[['설정','겹','귀무','A','C','주효과','잠재','총','비','S']].round(4).to_string(index=False))
print()
for _,r in t.iterrows():
    j=lambda v,th: '충족' if v>=th else '★미충족'
    print(f"  {r.설정:24s} R {j(r.총,.010)} · Q2 {j(r.잠재,.005)} · Q2b {j(r.비,1.0)} "
          f"· S {'충족' if r.S<=0 else '★미충족'}")
tt=t.총.values; rr=t.비.values
print(f"\n  총 {tt.min():.4f}~{tt.max():.4f} = {tt.max()/tt.min():.2f}배    비 {rr.min():.2f}~{rr.max():.2f} = {rr.max()/rr.min():.2f}배")
