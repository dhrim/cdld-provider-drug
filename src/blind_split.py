"""D6 — 제공자를 보지 않는 약물 분할에서 반전폭을 다시 잰다.

defend3.py 와 «절차는 동일»하고 약물 분할만 바꾼다.
  P1  약물별 관측 비시행률 순 -> 사건 수 4분위        (drug, y 만 사용)
  P2  약물별 사건 수 순       -> 사건 수 4분위        (drug 만 사용)
두 분할 모두 제공자 배정 셔플에 «불변»이므로 관측팔과 귀무팔이 같은 분할을 쓴다.
계획 동결: 20260909_D6_제공자무관_약물분할_사전고정.md
sha256 57a7e7f46d3ec88fba3f67d17ce931bcc6505a7d8d4464261857aef65e4ba3fa
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths, data

d = data.prep(None)
nur, drg, y = d['nur'], d['drg'], d['y']
n_nurse, n_drug = int(d['n_nurse']), int(d['n_drug'])

cnt_d = np.bincount(drg, minlength=n_drug).astype(float)
sum_d = np.bincount(drg, weights=y, minlength=n_drug)
rate_d = sum_d / np.maximum(cnt_d, 1)


def quartile_split(key):
    """key 오름차순으로 약물을 정렬하고 «사건 수» 4분위로 4군을 만든다."""
    order = np.argsort(key, kind='stable')
    cum = np.cumsum(cnt_d[order]) / cnt_d.sum()
    g_sorted = np.clip((cum * 4).astype(int), 0, 3)
    g = np.empty(n_drug, dtype=int); g[order] = g_sorted
    return g


SPLITS = {'P1': quartile_split(rate_d), 'P2': quartile_split(cnt_d)}


def table(lab, gi, min_cell=300, min_tot=2000):
    t = pd.DataFrame({'n': lab, 'g': gi, 'y': y})
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


rows = []
for name, g in SPLITS.items():
    gi = g[drg]
    obs, obn = spans(table(nur, gi))
    NP, NN = [], []
    for k in range(20):
        p, nb = spans(table(data.shuffle_nurse(d, 1000 + k), gi))
        if len(p): NP.append(p); NN.append(nb)
    nul = np.concatenate(NP)
    rows.append(dict(분할=name,
                     관측_중앙=np.median(obs), 귀무_중앙=np.median(nul),
                     배_중앙=np.median(obs) / np.median(nul),
                     관측_평균=obs.mean(), 귀무_평균=nul.mean(),
                     배_평균=obs.mean() / nul.mean(),
                     관측_제공자=obn, 관측_짝=len(obs),
                     귀무_짝=len(nul)))
    print(f'{name}  관측 {obn}명 · {len(obs):,}짝   귀무 {len(nul):,}짝', flush=True)

t = pd.DataFrame(rows)
out = os.path.join(paths.RESULTS, 'v9', 'table15_blind_split.csv')
t.to_csv(out, index=False)
print(t.round(3).to_string(index=False))
print('->', out)
