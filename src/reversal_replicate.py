"""표 2 를 «복제 수준» 통계량으로 통일한다.

기존 defend3.py 는 20회 귀무의 pair span 을 «전부 합친 뒤» 분위수를 냈다(pooled).
복제마다 eligible pair 수가 다르므로 pooled 통계량은 pair 가 많이 나온 복제에
더 큰 가중치를 준다. 반면 D7 의 몬테카를로 표준화 분리는 «복제»를 단위로 본다.
둘을 같은 정의로 맞춘다 — 복제마다 통계량 T_b 를 내고, 그 20개를 요약한다.

pooled 값도 함께 저장해 둘의 차이를 그대로 보인다.
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
    return (np.concatenate(out) if out else np.array([])), len(b)


STATS = [('중앙값', lambda a: float(np.median(a))),
         ('75분위', lambda a: float(np.quantile(a, .75))),
         ('90분위', lambda a: float(np.quantile(a, .90))),
         ('95분위', lambda a: float(np.quantile(a, .95))),
         ('99분위', lambda a: float(np.quantile(a, .99))),
         ('평균',   lambda a: float(a.mean()))]

obs, obs_band = spans(table(nur))
per, pools, bands, nprov = {k: [] for k, _ in STATS}, [], [], []
for k in range(20):
    p, nb = spans(table(data.shuffle_nurse(d, 1000 + k)))
    for lab, f in STATS: per[lab].append(f(p))
    pools.append(p); bands.append(len(p)); nprov.append(nb)
    print(f'  복제 {k:2d}  짝 {len(p):,}  대역 {nb}명  중앙 {per["중앙값"][-1]:.4f}', flush=True)

pooled = np.concatenate(pools)
rows = []
for lab, f in STATS:
    B = np.array(per[lab]); T = f(obs)
    rows.append(dict(통계=lab, 관측=T,
                     귀무_복제평균=B.mean(), 귀무_복제SD=B.std(ddof=1),
                     배=T / B.mean(),
                     표준화분리=(T - B.mean()) / B.std(ddof=1),
                     넘지못한복제=int((B >= T).sum()),
                     귀무_pooled=f(pooled)))
t = pd.DataFrame(rows)
t.to_csv(os.path.join(paths.RESULTS, 'v9', 'table17_reversal_replicate.csv'), index=False)

pc = pd.DataFrame(dict(복제=range(20), eligible_짝=bands, 대역_제공자=nprov))
pc.to_csv(os.path.join(paths.RESULTS, 'v9', 'table18_replicate_pairs.csv'), index=False)

print(t.round(4).to_string(index=False))
print(f'\n관측  대역 {obs_band}명 · 짝 {len(obs):,}')
print(f'귀무 복제당 짝  최소 {min(bands):,} · 중앙 {int(np.median(bands)):,} · 최대 {max(bands):,}'
      f'  (합 {sum(bands):,})')
print(f'귀무 복제당 대역 제공자  최소 {min(nprov)} · 중앙 {int(np.median(nprov))} · 최대 {max(nprov)}')
