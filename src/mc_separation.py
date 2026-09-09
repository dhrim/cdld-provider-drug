"""D7 — headline «중앙값»에 대한 복제 수준 몬테카를로 표준화 분리.

defend3.py 의 절차를 그대로 쓰되, 귀무 복제마다 통계량 T 를 «따로» 남긴다.
계획 동결: 20260909_D7_중앙값_몬테카를로분리_사전고정.md
sha256 f3ed990d315b421ce45b298e8a379f29fd2389e02789b65ef0ff5b50a3811afd
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths, data

d = data.prep(None)
z = np.load(paths.work('uv_interp.npz'), allow_pickle=True)
dl = np.asarray(z['drug_labels'], dtype=object)
nur, drg, y = d['nur'], d['drg'], d['y']; n_nurse = int(d['n_nurse'])
cl = pd.read_csv(os.path.join(paths.RESULTS, 'v9', 'out_04_drug_direction_clusters.csv'))
gmap = dict(zip(cl.drug.astype(str), cl.cluster_resid))
grp = np.array([gmap.get(str(x), -1) for x in dl]); gi = grp[drg]; ok = gi >= 0


def table(lab, min_cell=300, min_tot=2000):
    t = pd.DataFrame({'n': lab[ok], 'g': gi[ok], 'y': y[ok]})
    agg = t.groupby(['n', 'g']).agg(k=('y', 'size'), r=('y', 'mean')).reset_index()
    gb = t.groupby('g').y.mean(); agg['ratio'] = agg.r / agg.g.map(gb)
    piv = agg.pivot(index='n', columns='g', values='ratio')
    pk = agg.pivot(index='n', columns='g', values='k')
    cnt = np.bincount(lab, minlength=n_nurse).astype(float)
    rate = np.bincount(lab, weights=y, minlength=n_nurse) / np.maximum(cnt, 1)
    keep = (pk >= min_cell).all(axis=1) & (cnt[piv.index] >= min_tot) & piv.notna().all(axis=1)
    P = piv[keep].copy(); P['rate_all'] = rate[P.index]; return P


def spans(P, band=(.30, .70), maxdiff=.005):
    b = P[(P.rate_all > P.rate_all.quantile(band[0])) & (P.rate_all < P.rate_all.quantile(band[1]))]
    cols = [c for c in b.columns if c != 'rate_all']
    M = b[cols].to_numpy(); r = b.rate_all.to_numpy(); out = []
    for a in range(len(M)):
        m = np.abs(r[a + 1:] - r[a]) <= maxdiff
        if m.any():
            D = M[a] - M[a + 1:][m]; out.append(D.max(axis=1) - D.min(axis=1))
    return np.concatenate(out) if out else np.array([])


obs = spans(table(nur))
T_obs_med, T_obs_mean = float(np.median(obs)), float(obs.mean())

med_b, mean_b = [], []
for k in range(20):
    p = spans(table(data.shuffle_nurse(d, 1000 + k)))
    med_b.append(float(np.median(p))); mean_b.append(float(p.mean()))
    print(f'  복제 {k:2d}  중앙 {med_b[-1]:.4f}  평균 {mean_b[-1]:.4f}', flush=True)

med_b = np.array(med_b); mean_b = np.array(mean_b)
rows = []
for lab, T, B in (('중앙값', T_obs_med, med_b), ('평균', T_obs_mean, mean_b)):
    rows.append(dict(통계량=lab, T_obs=T, 귀무평균=B.mean(), 귀무SD=B.std(ddof=1),
                     표준화분리=(T - B.mean()) / B.std(ddof=1),
                     넘지못한복제=int((B >= T).sum()), 복제수=len(B)))
t = pd.DataFrame(rows)
out = os.path.join(paths.RESULTS, 'v9', 'table16_mc_separation.csv')
t.to_csv(out, index=False)
np.savez(os.path.join(paths.RESULTS, 'v9', 'mc_separation_replicates.npz'),
         med_b=med_b, mean_b=mean_b, T_obs_med=T_obs_med, T_obs_mean=T_obs_mean)
print(t.round(4).to_string(index=False)); print('->', out)
