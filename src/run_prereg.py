"""사전등록 본측정 — 간호사 x 약물 → 투약 누락.
`20260903_사전등록.md` 3~7절 그대로.

  A : 근무조 + 약물 + 진단챕터 + x'theta        (간호사 없음)
  B : + alpha[간호사]
  C : + delta[간호사,약물] = u_i' v_j            (랭크·릿지·가중 격자 선택)

환자 단위 5겹. 귀무 = 입원 단위 간호사 섞기 2회.
인자: [sens]  none|slide|temporal|n500|top100
"""
import os, sys, json, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths
import als_core as C
import icd as im

HOSP  = paths.HOSP        # admissions·patients·icustays·diagnoses_icd
OUT   = paths.work('results_als')
CACHE = paths.work('cache_arms')
SRC   = paths.EMAR
MIN_NURSE, MIN_DRUG, MIN_CELL = 200, 200, 5
RANKS = [1, 2, 3, 5, 10, 20]
RIDGES = [1., 10., 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
WEIGHTS = ['count', 'reliability']
N_FOLDS, N_INNER, N_SHUF, SEED = 5, 4, 2, 42
os.makedirs(OUT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
SENS = sys.argv[1] if len(sys.argv) > 1 else 'none'
ARM  = sys.argv[2] if len(sys.argv) > 2 else 'all'   # real | shuffle0 | shuffle1 | all

GIVEN = {'Administered', 'Started', 'Applied', 'Restarted',
         'Administered in Other Location', 'Administered Bolus from IV Drip'}
OMIT  = {'Not Given', 'Hold Dose', 'Not Started', 'Not Applied'}
DELAY = {'Delayed Administered', 'Delayed Started', 'Delayed Applied'}
SLIDE = {'Not Given per Sliding Scale'}


def build(sens):
    for _f in (SRC, f'{HOSP}/admissions.csv.gz', f'{HOSP}/patients.csv.gz',
               f'{HOSP}/diagnoses_icd.csv.gz', f'{HOSP}/icustays.csv.gz'):
        if not os.path.exists(_f):
            paths.need_data(f'MIMIC-IV input {_f}')
    em = pd.read_csv(SRC)
    # ★sens == 'nohold' : 'Hold Dose' 를 결과 정의에서 «완전히 제외»한다(0 으로 재분류하지 않는다).
    #   보류는 시행이 아니므로 0 으로 두면 결과변수를 오염시킨다. 행 자체를 뺀다.
    omit_set = OMIT - ({'Hold Dose'} if sens == 'nohold' else set())
    keep = GIVEN | omit_set | DELAY | (SLIDE if sens == 'slide' else set())
    e = em[em.event_txt.isin(keep)].copy()
    e['omit'] = e.event_txt.isin(omit_set | (SLIDE if sens == 'slide' else set())).astype(float)
    e['charttime'] = pd.to_datetime(e.charttime)
    e['shift'] = np.where(e.charttime.dt.hour.between(7, 18), 'day', 'night')
    e['drug'] = e.medication.astype(str).str.strip().str.upper()
    if sens == 'top100':
        top = set(e.drug.value_counts().head(100).index)
        e = e[e.drug.isin(top)]
    mn = 500 if sens == 'n500' else MIN_NURSE
    prev = None
    while True:
        n = e.groupby('enter_provider_id').size(); d = e.groupby('drug').size()
        e = e[e.enter_provider_id.isin(n[n >= mn].index) & e.drug.isin(d[d >= MIN_DRUG].index)]
        c = e.groupby(['enter_provider_id', 'drug']).size()
        k = set(c[c >= MIN_CELL].index)
        e = e[[x in k for x in zip(e.enter_provider_id, e.drug)]]
        s = (len(e), e.enter_provider_id.nunique(), e.drug.nunique())
        if s == prev: break
        prev = s
    e = e.reset_index(drop=True)
    adm = pd.read_csv(f'{HOSP}/admissions.csv.gz',
                      usecols=['hadm_id', 'admission_type', 'insurance'])
    pat = pd.read_csv(f'{HOSP}/patients.csv.gz',
                      usecols=['subject_id', 'gender', 'anchor_age', 'anchor_year',
                               'anchor_year_group'])
    e = e.merge(adm, on='hadm_id', how='left').merge(pat, on='subject_id', how='left')
    e['age'] = (e.anchor_age + (e.charttime.dt.year - e.anchor_year)).clip(18, 95)
    dx = pd.read_csv(f'{HOSP}/diagnoses_icd.csv.gz',
                     usecols=['hadm_id', 'seq_num', 'icd_code', 'icd_version'])
    p1 = dx[dx.seq_num == 1].drop_duplicates('hadm_id')
    e['dx'] = e.hadm_id.map({h: im.icd_chapter(c, v) for h, c, v in
                             zip(p1.hadm_id, p1.icd_code, p1.icd_version)}).fillna('기타')
    # ★중증도 대리 (U0 필수)
    e['n_dx'] = e.hadm_id.map(dx.groupby('hadm_id').size()).fillna(0)
    e['n_drug_adm'] = e.hadm_id.map(e.groupby('hadm_id').drug.nunique()).fillna(0)
    e['n_event_adm'] = e.hadm_id.map(e.groupby('hadm_id').size()).fillna(0)
    icu = pd.read_csv(f'{HOSP}/icustays.csv.gz', usecols=['hadm_id', 'los'])
    e['icu'] = e.hadm_id.isin(set(icu.hadm_id)).astype(float)
    e['icu_los'] = e.hadm_id.map(icu.groupby('hadm_id').los.sum()).fillna(0)
    return e


def design(e):
    cols = ['age', 'n_dx', 'n_drug_adm', 'n_event_adm', 'icu', 'icu_los']
    X = np.column_stack([e[c].fillna(e[c].median()).to_numpy(float) for c in cols])
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    for c in ['gender', 'admission_type', 'insurance']:
        X = np.hstack([X, pd.get_dummies(e[c].astype(str)).to_numpy(float)[:, 1:]])
    return X


def measure(y, base_codes, base_sizes, nur, n_i, drg, n_j, X, fold, tag):
    """A / B / C. C 는 겹마다 내부 4겹 CV 로 (랭크, 릿지, 가중) 선택."""
    cell = nur.astype(np.int64) * n_j + drg
    uq, ci_row = np.unique(cell, return_inverse=True)
    ci = (uq // n_j).astype(int); cj = (uq % n_j).astype(int)
    inner_of_cell = np.random.default_rng(SEED + 1).integers(0, N_INNER, len(uq))
    sse = {k: 0. for k in ['tot', 'A', 'B', 'C']}
    chosen = []
    for k in range(N_FOLDS):
        te = fold == k; tr = ~te
        sse['tot'] += float(((y[te] - y[tr].mean()) ** 2).sum())
        pA = C.fit_baseline(y, base_codes, base_sizes, X, tr)
        sse['A'] += float(((y[te] - pA(te)) ** 2).sum())
        pB = C.fit_baseline(y, base_codes + [nur], base_sizes + [n_i], X, tr)
        bt = pB(te); sse['B'] += float(((y[te] - bt) ** 2).sum())
        res = y[tr] - pB(tr)
        cnt = np.bincount(ci_row[tr], minlength=len(uq)).astype(float)
        s = np.bincount(ci_row[tr], weights=res, minlength=len(uq))
        m = cnt > 0
        r_c = np.zeros(len(uq)); r_c[m] = s[m] / cnt[m]
        sigma2 = res.var()
        between = np.average(r_c[m] ** 2, weights=cnt[m])
        tau2 = max(between - sigma2 * np.average(1./cnt[m], weights=cnt[m]), 1e-9)
        n0 = sigma2 / tau2

        def wv(sel, wn):
            return cnt[sel] if wn == 'count' else cnt[sel] / (cnt[sel] + n0)

        best, bl = None, np.inf
        for wn in WEIGHTS:
            for rg in RIDGES:
                for rank in RANKS:
                    tot = 0.
                    for q in range(N_INNER):
                        kp = m & (inner_of_cell != q); hd = m & (inner_of_cell == q)
                        W, R = C.dense_pair(ci[kp], cj[kp], r_c[kp], wv(kp, wn), n_i, n_j)
                        U, V = C.fit_low_rank(W, R, rank, rg)
                        d = (U[ci[hd]] * V[cj[hd]]).sum(1)
                        tot += float((cnt[hd] * (r_c[hd] - d) ** 2).sum())
                    if tot < bl: bl, best = tot, (wn, rg, rank)
        wn, rg, rank = best
        W, R = C.dense_pair(ci[m], cj[m], r_c[m], wv(m, wn), n_i, n_j)
        U, V = C.fit_low_rank(W, R, rank, rg)
        d = (U[ci] * V[cj]).sum(1)
        sse['C'] += float(((y[te] - bt - d[ci_row[te]]) ** 2).sum())
        chosen.append(f'{wn}/{rg:g}/r{rank}')
    r2 = {k: 1 - sse[k] / sse['tot'] for k in ['A', 'B', 'C']}
    print(f'  {tag:<16} A {r2["A"]:+.4f} · B {r2["B"]:+.4f} · C {r2["C"]:+.4f}   '
          f'(주효과 {r2["B"]-r2["A"]:+.4f} · 잠재 {r2["C"]-r2["B"]:+.4f})  {chosen[0]}',
          flush=True)
    return r2, chosen


if __name__ == '__main__':
    t0 = time.time()
    cp = f'{CACHE}/frame_{SENS}.pkl'
    if os.path.exists(cp):
        e = pd.read_pickle(cp); print(f'[{SENS}] 캐시 사용 {len(e):,}행', flush=True)
    else:
        e = build(SENS); e.to_pickle(cp)
    c = e.groupby(['enter_provider_id', 'drug']).size()
    nN, nD = e.enter_provider_id.nunique(), e.drug.nunique()
    w = c.to_numpy(float)
    print(f'[{SENS}] {len(e):,}행 · 간호사 {nN} x 약물 {nD} · 셀 {len(c):,} · '
          f'밀도 {len(c)/(nN*nD):.2%} · 유효표본 {w.sum()**2/(w**2).sum():,.0f}')
    print(f'  누락률 {e.omit.mean():.2%}  ({time.time()-t0:.0f}s)\n', flush=True)
    y = e.omit.to_numpy(float)
    nur, nl = pd.factorize(e.enter_provider_id); drg, dl = pd.factorize(e.drug)
    shf, sl = pd.factorize(e['shift']); dxc, dxl = pd.factorize(e.dx)
    X = design(e)
    bc = [shf, drg, dxc]; bs = [len(sl), len(dl), len(dxl)]
    if SENS == 'temporal':
        te_mask = e.anchor_year_group.isin(['2017 - 2019', '2020 - 2022']).to_numpy()
        fold = np.where(te_mask, 0, 1)
        globals()['N_FOLDS'] = 1
    else:
        pt, pl = pd.factorize(e.subject_id)
        fold = np.random.default_rng(SEED).integers(0, N_FOLDS, len(pl))[pt]
    hadm, hl = pd.factorize(e.hadm_id.fillna(-1))

    def null_A(seed):
        """★교정 귀무 — 입원별 제공자 «라벨 재배정». 구판(nh[hadm]=nur)은
        입원마다 한 명만 남겨 158,422개 입원을 전부 1인으로 접었다.
        보존: 입원당 행 수·제공자 수·배정 패턴·제공자 빈도 분포."""
        rng = np.random.default_rng(seed)
        order = np.argsort(hadm, kind='stable'); h = hadm[order]
        bnd = np.flatnonzero(np.r_[True, h[1:] != h[:-1], True])
        w = np.bincount(nur, minlength=len(nl)).astype(float); w /= w.sum()
        pool = np.arange(len(nl)); out = nur.copy()
        for s0, s1 in zip(bnd[:-1], bnd[1:]):
            idx = order[s0:s1]
            uq, inv = np.unique(nur[idx], return_inverse=True)
            out[idx] = rng.choice(pool, size=len(uq), replace=False, p=w)[inv]
        return out
    def null_B(seed):
        """대안 귀무 — 크기 층화 «입원 블록 교환». (행 수, 제공자 수)가 같은 입원끼리
        배정 벡터를 통째로 교환한다. 구조를 «정확히» 보존하나 층이 작으면 교환 상대가
        없어 그 입원은 원배정 그대로 남는다 -> 치환 강도가 약하다.
        2차 동결 계획 §3.4 가 약속한 «성능 결과 한 판»을 위해 저랭크 판에서 실행한다.
        대안 귀무는 추정기의 성질이 아니라 «귀무 설계»의 성질이다."""
        rng = np.random.default_rng(seed)
        order = np.argsort(hadm, kind='stable'); h = hadm[order]
        bnd = np.flatnonzero(np.r_[True, h[1:] != h[:-1], True])
        starts, ends = bnd[:-1], bnd[1:]
        key = {}
        for a, (s0, s1) in enumerate(zip(starts, ends)):
            v = nur[order[s0:s1]]
            key.setdefault((s1 - s0, len(np.unique(v))), []).append(a)
        out = nur.copy(); eligible = 0
        for members in key.values():
            if len(members) < 2:
                continue
            m = np.array(members); perm = m[rng.permutation(len(m))]
            src = [nur[order[starts[b]:ends[b]]].copy() for b in perm]
            for a, vec in zip(m, src):
                out[order[starts[a]:ends[a]]] = vec
            eligible += len(m)
        # ★두 수는 «다른 것»을 센다. v7 까지 이 둘을 혼동해 보고했다.
        #   (a) no_partner : 층에 교환 상대가 없어 «구조적으로» 못 움직인 입원
        #   (b) identical  : 최종 배정 벡터가 원본과 «실제로» 같은 입원. (a) 를 포함하며,
        #                    교환 가능한 층에서도 순열의 고정점이나 동일 벡터 교환으로 생긴다.
        no_partner = 1 - eligible / len(starts)
        identical = sum(1 for s0, s1 in zip(starts, ends)
                        if np.array_equal(out[order[s0:s1]], nur[order[s0:s1]])) / len(starts)
        print(f'  [null_B seed {seed}] 입원 {len(starts):,} — '
              f'교환 상대 없음 {no_partner:.2%} · ★배정 벡터가 실제로 동일 {identical:.2%} · '
              f'행 수준 제공자 유지 {(out == nur).mean():.2%}', flush=True)
        return out
    todo = [('real', nur)] if ARM == 'real' else (
           [(ARM, null_B(90+int(ARM[-1])))]
           if ARM.startswith('blockswap') else
           [(ARM, null_A(70+int(ARM[-1])))]
           if ARM.startswith('shuffle') else
           [('real', nur)] + [(f'shuffle{k}', null_A(70+k)) for k in range(N_SHUF)])
    for name, nu in todo:
        ap = f'{CACHE}/arm_{SENS}_{name}.json'
        if os.path.exists(ap):
            print(f'  {name} 완료됨 — 건너뜀', flush=True); continue
        r, ch = measure(y, bc, bs, nu, len(nl), drg, len(dl), X, fold, name)
        json.dump(dict(sens=SENS, arm=name, **r, chosen=ch), open(ap, 'w'))
    parts = [json.load(open(f'{CACHE}/arm_{SENS}_{n}.json'))
             for n in ['real'] + [f'shuffle{k}' for k in range(N_SHUF)]
             if os.path.exists(f'{CACHE}/arm_{SENS}_{n}.json')]
    if len(parts) < 1 + N_SHUF:
        print(f'\n미완 — 완료 {len(parts)}/{1+N_SHUF}. 남은 팔을 이어서 돌린다.'); sys.exit(0)
    t = pd.DataFrame([{k: v for k, v in p.items() if k != 'chosen'} for p in parts])
    t.to_csv(f'{OUT}/07_사전등록_{SENS}.csv', index=False)
    R, N = t.iloc[0], t.iloc[1:].mean(numeric_only=True)
    main = (R.B - R.A) - (N.B - N.A)
    lat = (R.C - R.B) - (N.C - N.B)
    tot = R.C - N.C
    print(f'\n{"="*66}')
    print(f'[{SENS}] 순가치 — 주효과 {main:+.4f} · 잠재 {lat:+.4f} · 총 {tot:+.4f}')
    if SENS == 'none':
        print(f'  S  건전성  섞기 잠재 {N.C-N.B:+.4f} <= 0  '
              f'{"통과" if N.C-N.B <= 0 else "★확인"}')
        print(f'  R  필수    총 {tot:+.4f} >= +0.010  {"통과" if tot >= 0.010 else "탈락"}')
        print(f'  Q2 핵심    잠재 {lat:+.4f} >= +0.005  {"통과" if lat >= 0.005 else "탈락"}')
        if main > 0:
            print(f'  Q2b 핵심   잠재/주효과 {lat/main:.2f} >= 1.0  '
                  f'{"통과" if lat/main >= 1.0 else "탈락"}')
    print(f'\n저장: {OUT}/07_사전등록_{SENS}.csv  ({time.time()-t0:.0f}s)')
