"""민감도 — 논문에 넣을 3종만. 비용을 줄인 판(겹·섞기 축소)을 명시적으로 쓴다.

  temporal   시간 분할 (2017-2019 · 2020-2022 를 시험부로). 겹 1개, 섞기 2회 → 7 단위
  cap16      ★잠재 차원 64 -> 16. «용량을 줄여도 분해가 같은가». 겹 3, 섞기 1 → 9 단위
  top100     상위 100 약물만. 셀 밀도 의존성. 겹 3, 섞기 1 → 9 단위

주판(base)·병동판(wardX)은 겹 5 · 섞기 2 로 «완전»하다. 여기만 축소한다.
"""
import os, sys, json, time, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data, cdld_core as CC

SENS = sys.argv[1]
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 540.
import paths
CD = paths.WORK + os.sep; CK = CD + 'ck/'
CFG = dict(temporal=dict(folds=[0], shuf=2, latent=64),
           cap16   =dict(folds=[0, 1, 2], shuf=1, latent=16),
           top100  =dict(folds=[0, 1, 2], shuf=1, latent=64))[SENS]
CYCLES, SUB, EP_AB, BS = 3, 1, 6, 8192


def prep_sens():
    cp = CD + f'.prep_{SENS}.npz'
    if os.path.exists(cp):
        z = np.load(cp, allow_pickle=True); return {k: z[k] for k in z.files}
    e = data.frame()
    if SENS == 'top100':
        top = set(e.drug.value_counts().head(100).index)
        e = e[e.drug.isin(top)]
        prev = None                       # 셀 가지치기를 사전명시대로 다시 돌린다
        while True:
            n = e.groupby('enter_provider_id').size(); dd = e.groupby('drug').size()
            e = e[e.enter_provider_id.isin(n[n >= 200].index) & e.drug.isin(dd[dd >= 200].index)]
            c = e.groupby(['enter_provider_id', 'drug']).size(); k = set(c[c >= 5].index)
            e = e[[x in k for x in zip(e.enter_provider_id, e.drug)]]
            s = (len(e), e.enter_provider_id.nunique(), e.drug.nunique())
            if s == prev: break
            prev = s
        e = e.reset_index(drop=True)
    d = data.prep(e, cache=False)
    if SENS == 'temporal':                # ★시간 분할 — 나중 시기를 시험부로
        late = e.anchor_year_group.isin(['2017 - 2019', '2020 - 2022']).to_numpy()
        d['fold'] = np.where(late, 0, 1).astype(np.int64)
    np.savez(cp, **d); return d


def auc(y, p):
    o = np.argsort(p); y = y[o]
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float('nan')
    r = np.arange(1, len(y) + 1)
    return float((r[y > .5].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def unit(d, cache, arm, fold, kind):
    key = f'{SENS}_{arm}_f{fold}_{kind}'; fp = CK + key + '.json'
    if os.path.exists(fp): return
    if arm not in cache:
        nu = d['nur'] if arm == 'real' else data.shuffle_nurse(d, int(arm[-1]), 'A')
        cache[arm] = data.tensors(d, nur_override=nu)
    D = cache[arm]
    tr = np.flatnonzero(d['fold'] != fold); te = np.flatnonzero(d['fold'] == fold)
    y = d['y']; kw = dict(n_nurse=int(d['n_nurse']), n_drug=int(d['n_drug']))
    sizes = list(d['sizes']); ncov = d['cov'].shape[1]
    CC.LATENT = CFG['latent']                      # ★용량 축소 판
    if kind == 'A':   m = CC.Finder(sizes, ncov, n_drug=kw['n_drug'], drug_latent=True)
    elif kind == 'B': m = CC.Finder(sizes, ncov, drug_latent=True, **kw)
    else:             m = CC.Finder(sizes, ncov, latent=True, **kw)
    t0 = time.time()
    CC.fit(m, D, tr, te, cycles=(CYCLES if kind == 'C' else EP_AB),
           sub_epochs=(SUB if kind == 'C' else 1), bs=BS, seed=42 + fold,
           cyclic=(kind == 'C'))
    p = CC.predict(m, D, te)
    res = dict(tag=SENS, arm=arm, fold=int(fold), kind=kind, n_te=int(len(te)),
               sse=float(((y[te] - p) ** 2).sum()),
               sstot=float(((y[te] - y[tr].mean()) ** 2).sum()),
               auc=auc(y[te], p), secs=round(time.time() - t0, 1),
               latent=CFG['latent'])
    json.dump(res, open(fp, 'w'))
    print(f'  ✓ {key}  R2 {1-res["sse"]/res["sstot"]:+.4f}  AUC {res["auc"]:.4f}  {res["secs"]:.0f}s', flush=True)


def main():
    d = prep_sens()
    print(f'[{SENS}] {len(d["y"]):,}행 · 간호사 {int(d["n_nurse"])} x 약물 {int(d["n_drug"])} '
          f'· 누락률 {float(d["y"].mean()):.5f} · 잠재 {CFG["latent"]}차원', flush=True)
    arms = ['real'] + [f'shuffle{k}' for k in range(CFG['shuf'])]
    q = [('real', f, 'A') for f in CFG['folds']]
    q += [(a, f, k) for k in ('B', 'C') for a in arms for f in CFG['folds']]
    cache = {}; t0 = time.time()
    for a, f, k in q:
        if os.path.exists(CK + f'{SENS}_{a}_f{f}_{k}.json'): continue
        if time.time() - t0 + {'A': 340., 'B': 350., 'C': 470.}[k] > BUDGET: break
        unit(d, cache, a, f, k)
    left = sum(1 for a, f, k in q if not os.path.exists(CK + f'{SENS}_{a}_f{f}_{k}.json'))
    print(f'\n[{SENS}] 남은 {left} / 전체 {len(q)}', flush=True)


main()
