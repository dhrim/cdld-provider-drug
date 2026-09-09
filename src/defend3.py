"""반전 폭 분포를 저장한다 (그림 4 · 표용)."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths, data
d = data.prep(None)
z = np.load(paths.work('uv_interp.npz'), allow_pickle=True)
dl = np.asarray(z['drug_labels'], dtype=object)
nur, drg, y = d['nur'], d['drg'], d['y']; n_nurse = int(d['n_nurse'])
cl = pd.read_csv(os.path.join(paths.RESULTS,'v9','out_04_drug_direction_clusters.csv'))
gmap = dict(zip(cl.drug.astype(str), cl.cluster_resid))
grp = np.array([gmap.get(str(x), -1) for x in dl]); gi = grp[drg]; ok = gi >= 0

def table(lab, min_cell=300, min_tot=2000):
    t = pd.DataFrame({'n': lab[ok], 'g': gi[ok], 'y': y[ok]})
    agg = t.groupby(['n','g']).agg(k=('y','size'), r=('y','mean')).reset_index()
    gb = t.groupby('g').y.mean(); agg['ratio'] = agg.r/agg.g.map(gb)
    piv = agg.pivot(index='n',columns='g',values='ratio'); pk = agg.pivot(index='n',columns='g',values='k')
    cnt = np.bincount(lab,minlength=n_nurse).astype(float)
    rate = np.bincount(lab,weights=y,minlength=n_nurse)/np.maximum(cnt,1)
    keep = (pk>=min_cell).all(axis=1)&(cnt[piv.index]>=min_tot)&piv.notna().all(axis=1)
    P = piv[keep].copy(); P['rate_all']=rate[P.index]; return P

def spans(P, band=(.30,.70), maxdiff=.005):
    b = P[(P.rate_all>P.rate_all.quantile(band[0]))&(P.rate_all<P.rate_all.quantile(band[1]))]
    cols=[c for c in b.columns if c!='rate_all']; M=b[cols].to_numpy(); r=b.rate_all.to_numpy()
    out=[]
    for a in range(len(M)):
        m=np.abs(r[a+1:]-r[a])<=maxdiff
        if m.any():
            D=M[a]-M[a+1:][m]; out.append(D.max(axis=1)-D.min(axis=1))
    return (np.concatenate(out) if out else np.array([])), len(b)

obs,obn = spans(table(nur))
NP=[]; NN=[]
for k in range(20):
    p,n = spans(table(data.shuffle_nurse(d,1000+k)))
    if len(p): NP.append(p); NN.append(n)
nul = np.concatenate(NP); nmax=np.array([p.max() for p in NP]); nmean=np.array([p.mean() for p in NP])
np.savez(os.path.join(paths.RESULTS,'v9','reversal.npz'), obs=obs, null=nul, null_max=nmax, null_mean=nmean,
         obs_band=obn, null_band=np.array(NN))
rows=[]
for q,lab in [(.5,'중앙값'),(.75,'75분위'),(.9,'90분위'),(.95,'95분위'),(.99,'99분위')]:
    rows.append(dict(통계=lab, 관측=np.quantile(obs,q), 귀무=np.quantile(nul,q),
                     배=np.quantile(obs,q)/np.quantile(nul,q)))
rows.append(dict(통계='평균', 관측=obs.mean(), 귀무=nul.mean(), 배=obs.mean()/nul.mean()))
rows.append(dict(통계='최대', 관측=obs.max(), 귀무=nmax.mean(), 배=np.nan))
t=pd.DataFrame(rows); t.to_csv(os.path.join(paths.RESULTS,'v9','out_09_reversal_span.csv'), index=False)
print(t.round(3).to_string(index=False))
print(f'\n관측 대역 {obn}명 · 짝 {len(obs):,}   귀무 대역 {np.mean(NN):.0f}명 · 짝 {len(nul)//len(NP):,}/회')
print(f'평균 z = {(obs.mean()-nmean.mean())/nmean.std(ddof=1):+.1f}')
print(f'최대 z = {(obs.max()-nmax.mean())/nmax.std(ddof=1):+.1f}  (귀무 최대의 최대 {nmax.max():.2f})')
