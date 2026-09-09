"""★병동 교란 관문 — careunit 을 넣고도 간호사 x 약물 잠재가 남는가.

사전등록 밖의 민감도. 격자는 사전등록 본측정에서 다섯 겹 모두 선택된
고정점(reliability / ridge 1 / rank 20)을 쓴다 — 내부 4겹 격자탐색은 생략.

참조선 세 판
  base   : 근무조 + 약물 + 진단챕터 + x'theta            (사전등록 그대로)
  ward   : + gamma[careunit]                             (병동 주효과)
  wardX  : + gamma[careunit] + delta_W[careunit, 약물]   (★U0. 축소 셀평균)

각 판에서  A(간호사 없음) -> B(+주효과) -> C(+잠재)  를 내고
입원 단위 간호사 섞기 2회를 귀무로 뺀다.
"""
import os, sys, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import als_core as C
import icd as im

U   = paths.DATA + os.sep      # MIMIC-IV 파생 입력 (emar_slim · hosp 테이블)
W   = paths.DATA + os.sep
RANK, RIDGE, WEIGHT = 20, 1.0, 'reliability'
N_FOLDS, N_SHUF, SEED = 5, 2, 42
MIN_NURSE, MIN_DRUG, MIN_CELL = 200, 200, 5

GIVEN = {'Administered','Started','Applied','Restarted',
         'Administered in Other Location','Administered Bolus from IV Drip'}
OMIT  = {'Not Given','Hold Dose','Not Started','Not Applied'}
DELAY = {'Delayed Administered','Delayed Started','Delayed Applied'}


def build():
    cp = W + '.frame_ward.pkl'
    if os.path.exists(cp):
        return pd.read_pickle(cp)
    for _f in ('emar_slim.csv.gz', 'admissions.csv.gz', 'patients.csv.gz',
               'diagnoses_icd.csv.gz', 'icustays.csv.gz'):
        if not os.path.exists(U + _f):
            paths.need_data(f'MIMIC-IV input {U}{_f}')
    em = pd.read_csv(U+'emar_slim.csv.gz')
    e = em[em.event_txt.isin(GIVEN | OMIT | DELAY)].copy()
    e['omit'] = e.event_txt.isin(OMIT).astype(float)
    e['charttime'] = pd.to_datetime(e.charttime)
    e['shift'] = np.where(e.charttime.dt.hour.between(7, 18), 'day', 'night')
    e['drug'] = e.medication.astype(str).str.strip().str.upper()
    prev = None
    while True:
        n = e.groupby('enter_provider_id').size(); d = e.groupby('drug').size()
        e = e[e.enter_provider_id.isin(n[n >= MIN_NURSE].index)
              & e.drug.isin(d[d >= MIN_DRUG].index)]
        c = e.groupby(['enter_provider_id','drug']).size()
        k = set(c[c >= MIN_CELL].index)
        e = e[[x in k for x in zip(e.enter_provider_id, e.drug)]]
        s = (len(e), e.enter_provider_id.nunique(), e.drug.nunique())
        if s == prev: break
        prev = s
    e = e.reset_index(drop=True)
    # ★병동
    cu = pd.read_csv(W+'emar_careunit.csv.gz')
    e = e.merge(cu, on='emar_id', how='left')
    e['careunit'] = e.careunit.fillna('_MISSING_')
    # 공변량 (사전등록 그대로)
    adm = pd.read_csv(U+'admissions.csv.gz', usecols=['hadm_id','admission_type','insurance'])
    pat = pd.read_csv(U+'patients.csv.gz',
                      usecols=['subject_id','gender','anchor_age','anchor_year','anchor_year_group'])
    e = e.merge(adm, on='hadm_id', how='left').merge(pat, on='subject_id', how='left')
    e['age'] = (e.anchor_age + (e.charttime.dt.year - e.anchor_year)).clip(18, 95)
    dx = pd.read_csv(U+'diagnoses_icd.csv.gz',
                     usecols=['hadm_id','seq_num','icd_code','icd_version'])
    p1 = dx[dx.seq_num == 1].drop_duplicates('hadm_id')
    e['dx'] = e.hadm_id.map({h: im.icd_chapter(c, v) for h, c, v in
                             zip(p1.hadm_id, p1.icd_code, p1.icd_version)}).fillna('기타')
    e['n_dx'] = e.hadm_id.map(dx.groupby('hadm_id').size()).fillna(0)
    e['n_drug_adm'] = e.hadm_id.map(e.groupby('hadm_id').drug.nunique()).fillna(0)
    e['n_event_adm'] = e.hadm_id.map(e.groupby('hadm_id').size()).fillna(0)
    icu = pd.read_csv(U+'icustays.csv.gz', usecols=['hadm_id','los'])
    e['icu'] = e.hadm_id.isin(set(icu.hadm_id)).astype(float)
    e['icu_los'] = e.hadm_id.map(icu.groupby('hadm_id').los.sum()).fillna(0)
    e.to_pickle(cp)
    return e


def design(e):
    cols = ['age','n_dx','n_drug_adm','n_event_adm','icu','icu_los']
    X = np.column_stack([e[c].fillna(e[c].median()).to_numpy(float) for c in cols])
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    for c in ['gender','admission_type','insurance']:
        X = np.hstack([X, pd.get_dummies(e[c].astype(str)).to_numpy(float)[:, 1:]])
    return X


def shrunk_cell(res_tr, cell_tr, cell_all, ncell):
    """훈련 잔차의 축소 셀평균. 보지 못한 셀은 0."""
    cnt = np.bincount(cell_tr, minlength=ncell).astype(float)
    s   = np.bincount(cell_tr, weights=res_tr, minlength=ncell)
    m = cnt > 0
    r = np.zeros(ncell); r[m] = s[m] / cnt[m]
    sig2 = res_tr.var()
    betw = np.average(r[m]**2, weights=cnt[m])
    tau2 = max(betw - sig2*np.average(1./cnt[m], weights=cnt[m]), 1e-9)
    n0 = sig2 / tau2
    r = r * (cnt / (cnt + n0))
    return r[cell_all]


def run(y, bc, bs, X, nur, n_i, drg, n_j, fold, wcode, n_w, mode, tag):
    cell = nur.astype(np.int64)*n_j + drg
    uq, ci_row = np.unique(cell, return_inverse=True)
    ci = (uq // n_j).astype(int); cj = (uq % n_j).astype(int)
    wd_cell = wcode.astype(np.int64)*n_j + drg          # (careunit, 약물)
    sse = dict(tot=0., A=0., B=0., C=0.)
    codes, sizes = list(bc), list(bs)
    if mode in ('ward','wardX'):
        codes = codes + [wcode]; sizes = sizes + [n_w]
    for k in range(N_FOLDS):
        te = fold == k; tr = ~te
        sse['tot'] += float(((y[te] - y[tr].mean())**2).sum())
        pA = C.fit_baseline(y, codes, sizes, X, tr)
        aTr, aTe = pA(tr), pA(te)
        if mode == 'wardX':                              # ★U0 : careunit x 약물
            wd = shrunk_cell(y[tr]-aTr, wd_cell[tr], wd_cell, n_w*n_j)
            aTr = aTr + wd[tr]; aTe = aTe + wd[te]
            yy = y - wd                                  # 이후 단계는 U0 제거 후 잔차에
        else:
            yy = y
        sse['A'] += float(((y[te] - aTe)**2).sum())
        pB = C.fit_baseline(yy, codes + [nur], sizes + [n_i], X, tr)
        bTe = pB(te) + (wd[te] if mode == 'wardX' else 0.)
        sse['B'] += float(((y[te] - bTe)**2).sum())
        res = yy[tr] - pB(tr)
        cnt = np.bincount(ci_row[tr], minlength=len(uq)).astype(float)
        s   = np.bincount(ci_row[tr], weights=res, minlength=len(uq))
        m = cnt > 0
        r_c = np.zeros(len(uq)); r_c[m] = s[m]/cnt[m]
        sig2 = res.var()
        betw = np.average(r_c[m]**2, weights=cnt[m])
        tau2 = max(betw - sig2*np.average(1./cnt[m], weights=cnt[m]), 1e-9)
        n0 = sig2/tau2
        wv = cnt[m]/(cnt[m]+n0) if WEIGHT == 'reliability' else cnt[m]
        Wm, Rm = C.dense_pair(ci[m], cj[m], r_c[m], wv, n_i, n_j)
        Uf, Vf = C.fit_low_rank(Wm, Rm, RANK, RIDGE)
        d = (Uf[ci]*Vf[cj]).sum(1)
        sse['C'] += float(((y[te] - bTe - d[ci_row[te]])**2).sum())
    r2 = {k: 1 - sse[k]/sse['tot'] for k in ['A','B','C']}
    print(f'  {tag:<22} A {r2["A"]:+.4f} · B {r2["B"]:+.4f} · C {r2["C"]:+.4f}   '
          f'(주효과 {r2["B"]-r2["A"]:+.4f} · 잠재 {r2["C"]-r2["B"]:+.4f})', flush=True)
    return r2


if __name__ == '__main__':
    t0 = time.time()
    e = build()
    print(f'{len(e):,}행 · 간호사 {e.enter_provider_id.nunique()} x '
          f'약물 {e.drug.nunique()} · 병동 {e.careunit.nunique()} · '
          f'누락률 {e.omit.mean():.2%}  ({time.time()-t0:.0f}s)', flush=True)
    print(f'  병동 미배정 {100*(e.careunit=="_MISSING_").mean():.1f}%\n', flush=True)
    y = e.omit.to_numpy(float)
    nur, nl = pd.factorize(e.enter_provider_id); drg, dl = pd.factorize(e.drug)
    shf, sl = pd.factorize(e['shift']);          dxc, dxl = pd.factorize(e.dx)
    wcd, wl = pd.factorize(e.careunit)
    X = design(e)
    bc, bs = [shf, drg, dxc], [len(sl), len(dl), len(dxl)]
    pt, pl = pd.factorize(e.subject_id)
    fold = np.random.default_rng(SEED).integers(0, N_FOLDS, len(pl))[pt]
    hadm, hl = pd.factorize(e.hadm_id.fillna(-1))

    def null_A(seed):
        """★교정 귀무 — 입원별 제공자 «라벨 재배정» (구판은 입원을 1인으로 접었다)."""
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

    arms = [('real', nur)] + [(f'shuffle{k}', null_A(70+k)) for k in range(N_SHUF)]
    rows = []
    for mode in ['base','ward','wardX']:
        print(f'[{mode}]', flush=True)
        for name, nu in arms:
            f = f'{W}.w_{mode}_{name}.json'
            if os.path.exists(f):
                r = json.load(open(f))
                print(f'  {name:<22} 완료됨', flush=True)
            else:
                r = run(y, bc, bs, X, nu, len(nl), drg, len(dl), fold,
                        wcd, len(wl), mode, name)
                json.dump(r, open(f,'w'))
            rows.append(dict(mode=mode, arm=name, **r))
        print('', flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(W+'09_병동관문.csv', index=False)
    print('='*70)
    print(f'{"판":<8}{"주효과":>10}{"잠재":>10}{"총":>10}{"비":>8}{"섞기잠재":>10}')
    for mode in ['base','ward','wardX']:
        g = t[t['mode']==mode]
        R = g.iloc[0]; N = g.iloc[1:].mean(numeric_only=True)
        main = (R.B-R.A)-(N.B-N.A); lat = (R.C-R.B)-(N.C-N.B); tot = R.C-N.C
        ratio = lat/main if main > 0 else float('nan')
        print(f'{mode:<8}{main:>+10.4f}{lat:>+10.4f}{tot:>+10.4f}{ratio:>8.2f}'
              f'{N.C-N.B:>+10.4f}')
    print(f'\n({time.time()-t0:.0f}s)')
