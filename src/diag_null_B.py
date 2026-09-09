"""대안 귀무(블록 교환)의 «치환 강도» 진단 — table12_null_B_diag.csv 를 만든다.

v7 까지 원고는 `1.6%` 를 「원배정 그대로 남은 입원」이라고 적었다. **틀렸다.**
그 수는 `moved` 에서 나오는데, `moved` 는 «층에 교환 상대가 있었는가»만 세므로

    (a) 교환 상대가 없는 층의 입원          <- 1.6% 가 세는 것
    (b) 최종 배정 벡터가 실제로 원본과 같은 입원

는 다른 양이다. (b) 는 (a) 를 포함하고, 크기 2 인 층에서 무작위 순열이 항등이 될
확률이 1/2 이므로 (a) 보다 «상당히» 크다. 이 스크립트가 (b) 를 직접 센다.

    $ python src/diag_null_B.py            # 결과 CSV 재생성 (MIMIC 파생 캐시 필요)

모형을 적합하지 않는다. 순열만 재현한다.
"""
import os, sys, csv
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

FRAME = os.path.join(paths.DATA, 'frame_none.pkl')
if not os.path.exists(FRAME):
    paths.need_data(f'ALS-arm frame cache ({FRAME})')

e = pd.read_pickle(FRAME)
nur, _ = pd.factorize(e.enter_provider_id)
hadm, _ = pd.factorize(e.hadm_id.fillna(-1))
order = np.argsort(hadm, kind='stable'); h = hadm[order]
bnd = np.flatnonzero(np.r_[True, h[1:] != h[:-1], True])
starts, ends = bnd[:-1], bnd[1:]
nadm = len(starts)

strata = {}
for a, (s0, s1) in enumerate(zip(starts, ends)):
    v = nur[order[s0:s1]]
    strata.setdefault((s1 - s0, len(np.unique(v))), []).append(a)
singletons = {a for m in strata.values() if len(m) < 2 for a in m}

rows = []
for seed in (90, 91):                      # run_prereg.py 의 blockswap0 · blockswap1
    rng = np.random.default_rng(seed)
    out = nur.copy(); eligible = 0
    for members in strata.values():
        if len(members) < 2:
            continue
        m = np.array(members); perm = m[rng.permutation(len(m))]
        src = [nur[order[starts[b]:ends[b]]].copy() for b in perm]
        for a, vec in zip(m, src):
            out[order[starts[a]:ends[a]]] = vec
        eligible += len(m)
    ident = [a for a, (s0, s1) in enumerate(zip(starts, ends))
             if np.array_equal(out[order[s0:s1]], nur[order[s0:s1]])]
    rows.append({
        'seed': seed, '입원': nadm,
        '상대없는층_입원': nadm - eligible,
        '상대없는층_비율': round((nadm - eligible) / nadm, 5),
        '★실제동일_입원': len(ident),
        '★실제동일_비율': round(len(ident) / nadm, 5),
        '그중_교환가능층': sum(1 for a in ident if a not in singletons),
        '행_제공자유지': round(float((out == nur).mean()), 5)})
    print(f"seed {seed}: 상대없음 {rows[-1]['상대없는층_비율']:.2%} · "
          f"★실제동일 {rows[-1]['★실제동일_비율']:.2%} · "
          f"행유지 {rows[-1]['행_제공자유지']:.2%}")

fp = os.path.join(paths.RESULTS, 'v7', 'table12_null_B_diag.csv')
with open(fp, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader()
    [w.writerow(r) for r in rows]
print('저장:', fp)
