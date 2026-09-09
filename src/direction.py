import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
"""약물 잠재 «방향»이 누락률의 재진술인가, 아니면 별개 축인가. (CDLD 판)"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import data
from sklearn.cluster import KMeans

d=data.prep(None)
z=np.load(paths.work('uv_interp.npz'), allow_pickle=True)
V=z['V']; dl=np.asarray(z['drug_labels'],dtype=object)
y=d['y']; drg=d['drg']
cnt=np.bincount(drg,minlength=len(dl)).astype(float)
rate=np.bincount(drg,weights=y,minlength=len(dl))/np.maximum(cnt,1)
keep=cnt>=2000
Vk=V[keep]; rk=rate[keep]; ck=cnt[keep]; lk=dl[keep]
N=Vk/np.linalg.norm(Vk,axis=1,keepdims=True)
print(f'대상 약물 {keep.sum()}종 (각 2,000건 이상)')

def cohesion(X, lab):
    s=0.
    for c in np.unique(lab):
        M=X[lab==c]
        if len(M)<2: continue
        G=M@M.T; s+=(G.sum()-len(M))/(len(M)*(len(M)-1))*len(M)
    return s/len(X)

km=KMeans(4, n_init=20, random_state=0).fit(N); lab=km.labels_
coh=cohesion(N,lab)
# 누락률·log건수 방향 제거
Z=np.column_stack([np.ones(len(rk)), rk, np.log(ck)])
R=N - Z@np.linalg.lstsq(Z,N,rcond=None)[0]
R=R/np.maximum(np.linalg.norm(R,axis=1,keepdims=True),1e-9)
km2=KMeans(4, n_init=20, random_state=0).fit(R); lab2=km2.labels_
coh2=cohesion(R,lab2)
rng=np.random.default_rng(0)
null=[cohesion(R, rng.permutation(lab2)) for _ in range(300)]
zsc=(coh2-np.mean(null))/np.std(null)
print(f'원 방향 군집 응집도 {coh:.4f}')
print(f'누락률+log건수 제거 후 응집도 {coh2:.4f}  ({100*coh2/coh:.0f}% 잔존)  순열 z = {zsc:+.1f}')

# 누락률 사분위와의 일치
q=pd.qcut(rk,4,labels=False)
agree=max((pd.crosstab(lab2,q).max(1).sum())/len(lab2), 0)
print(f'누락률 사분위로 잔차군집을 맞히는 비율 {100*agree:.0f}%')

# ★군집 번호를 «관측 비시행률 오름차순»으로 고정한다 (canonicalization).
# KMeans 내부 라벨 0~3 은 실행마다 뒤바뀔 수 있으므로 그대로 쓰지 않는다.
_w=[float(np.average(rk[lab2==c],weights=ck[lab2==c])) for c in range(4)]
_order=np.argsort(_w)                      # 낮은 비시행률부터
_canon=np.empty(4,dtype=int); _canon[_order]=np.arange(4)
lab2=_canon[lab2]                          # 0=가장 낮음 ... 3=가장 높음
lab =_canon[lab] if len(np.unique(lab))==4 else lab

rows=[]
for c in range(4):
    m=lab2==c
    top=lk[m][np.argsort(-ck[m])][:8]
    rows.append(dict(군집=c, 약물수=int(m.sum()), 건수=int(ck[m].sum()),
                     누락률=float(np.average(rk[m],weights=ck[m])),
                     대표약물=' · '.join(map(str,top))))
t=pd.DataFrame(rows)                        # 이미 오름차순이다
assert list(t.누락률)==sorted(t.누락률), '군집 번호가 관측 비시행률 오름차순이 아니다'
pd.set_option('display.width',200,'display.max_colwidth',110)
print('\n'+t.to_string(index=False))
pd.DataFrame(dict(drug=lk, n=ck, rate=rk, cluster=lab, cluster_resid=lab2)).to_csv(
    os.path.join(paths.RESULTS,'v9','out_04_drug_direction_clusters.csv'), index=False)
t.to_csv(os.path.join(paths.RESULTS,'v9','out_05_cluster_summary.csv'), index=False)
print('-> out_04_drug_direction_clusters.csv · out_05_cluster_summary.csv')
