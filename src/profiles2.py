"""제공자 프로파일 — 군집별 배수. 군집에는 번호만 쓴다."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths, data
d=data.prep(None)
cl=pd.read_csv(os.path.join(paths.RESULTS,'v9','out_04_drug_direction_clusters.csv'))
# ★군집에 임상적 이름을 붙이지 않는다 — 번호는 관측 비시행률 오름차순(direction.py).
NAME={c: f'cluster_{c+1}' for c in range(4)}
z=np.load(paths.work('uv_interp.npz'), allow_pickle=True)
nl=np.asarray(z['nurse_labels'],dtype=object); dl=np.asarray(z['drug_labels'],dtype=object)
cmap=dict(zip(cl.drug, cl.cluster_resid))
nur,drg,y=d['nur'],d['drg'],d['y']
grp=np.array([cmap.get(str(x),-1) for x in dl]); gi=grp[drg]
ok=gi>=0
tab=pd.DataFrame({'n':nur[ok],'g':gi[ok],'y':y[ok]})
agg=tab.groupby(['n','g']).agg(k=('y','size'), r=('y','mean')).reset_index()
gbase=tab.groupby('g').y.mean()
agg['ratio']=agg.r/agg.g.map(gbase)
piv=agg.pivot(index='n',columns='g',values='ratio'); pk=agg.pivot(index='n',columns='g',values='k')
cnt=np.bincount(nur).astype(float); rate=np.bincount(nur,weights=y)/cnt
keep=(pk>=300).all(axis=1) & (cnt[piv.index]>=2000)
P=piv[keep].copy(); K=pk[keep]
P['n_all']=cnt[P.index]; P['rate_all']=rate[P.index]
print(f'대상 제공자 {len(P)}명 · 전체 누락률 중앙 {P.rate_all.median():.3f}')
band=P[(P.rate_all>P.rate_all.quantile(.30))&(P.rate_all<P.rate_all.quantile(.70))]
cols=sorted(gbase.index)
M=band[cols].to_numpy(); rr=band.rate_all.to_numpy(); best=None
MAXD=0.005                       # ★전체 누락률 차이 0.5%p 이내인 짝만
for a in range(len(M)):
    for b in range(a+1,len(M)):
        if abs(rr[a]-rr[b])>MAXD: continue
        s=(M[a]-M[b]).max()-(M[a]-M[b]).min()
        if best is None or s>best[0]: best=(s,a,b)
sc,a,b=best
print(f'후보 {len(M)}명 중 전체 누락률 차이 {MAXD*100:.1f}%p 이내로 제한')
print(f'\n★가장 대비되는 짝 (반전 폭 {sc:.2f})')
rows=[]
for who,ii in (('A',a),('B',b)):
    r=band.iloc[ii]; kk=K.loc[band.index[ii]]      # ★위치색인 오정렬 수정
    print(f'\n  제공자 {who}  총 {int(r.n_all):,}건 · 전체 누락률 {100*r.rate_all:.1f}%')
    rec=dict(nurse=who, n=int(r.n_all), overall=float(r.rate_all))
    for g in cols:
        print(f'      {NAME[g]:<14} 평균의 {r[g]:.2f}배  (n={int(kk[g]):,})')
        rec[NAME[g]]=round(float(r[g]),2)
    rows.append(rec)
pd.DataFrame(rows).to_csv(os.path.join(paths.RESULTS,'v9','out_06_provider_profiles.csv'),index=False)
# 제공자 단위 표(out_07)는 배포하지 않는다 — DUA 보수적 해석. 필요하면 로컬에서만 생성.
print('\n군별 전체 평균 누락률:')
for g in cols: print(f'  {NAME[g]:<14} {100*gbase[g]:.1f}%')
