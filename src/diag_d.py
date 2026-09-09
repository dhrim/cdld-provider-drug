"""native B 비가법성 진단 — 사전 문서 §6.

고정 context x 에서 (제공자 x 약물) 격자 위 함수 eta_B(i,j;x) 를 만들고
  D        : 이차 차분 (비가법성 대비)
  M        : 제공자 주대비 (약물에 대해 평균)
  sd(D)/sd(M) : 무차원 눈금
  ANOVA2   : 격자 위 함수의 이원 분산분해 -> 상호작용 분율
를 계산한다.

support : 격자의 «모든» 칸 (i,j) 가 학습셋에서 >= N 사건으로 관측된 부분격자만 쓴다.
          주 분석 N=20, 민감도 N in {5, 50}.  D_all 은 support 무시 무작위 격자.
"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data, cdld_core as CC, cdld_a as CA

import paths
CD = paths.WORK + os.sep; CKB = CD + 'ckpt/'
OUT = CD + 'diag/'; os.makedirs(OUT, exist_ok=True)
N_X = 20          # 고정 context 표본 수
MAX_I, MAX_J = 40, 20


def load_B(fold, d):
    ck = torch.load(CKB + f'B_f{fold}.pt', map_location='cpu', weights_only=False)
    m = CC.Finder(ck['sizes'], ck['ncov'], n_nurse=ck['n_nurse'],
                  n_drug=ck['n_drug'], drug_latent=True)
    m.load_state_dict(ck['state']); m.eval()
    return m


def counts(d, fold):
    """학습셋의 (제공자 x 약물) 사건수 희소 계수."""
    tr = np.flatnonzero(d['fold'] != fold)
    nur, drg = d['nur'][tr], d['drg'][tr]
    n_n, n_d = int(d['n_nurse']), int(d['n_drug'])
    C = np.zeros((n_n, n_d), dtype=np.int32)
    np.add.at(C, (nur, drg), 1)
    return C, tr


def pick_grid(C, N, max_i=MAX_I, max_j=MAX_J):
    """모든 칸이 >= N 인 부분격자를 탐욕적으로 찾는다."""
    drug_tot = C.sum(0)
    best = None
    for K in (max_j, 15, 12, 10, 8, 6, 5, 4, 3):
        J = np.argsort(-drug_tot)[:K]
        ok = np.flatnonzero(C[:, J].min(1) >= N)
        if len(ok) >= 6:
            prov_tot = C.sum(1)[ok]
            I = ok[np.argsort(-prov_tot)[:max_i]]
            if best is None or len(I) * K > best[2]:
                best = (I, J, len(I) * K)
    if best is None:
        return None, None
    return best[0], best[1]


def rand_grid(C, n_i, n_j, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.choice(C.shape[0], size=n_i, replace=False),
            rng.choice(C.shape[1], size=n_j, replace=False))


@torch.no_grad()
def eta_on_grid(m, d, row, I, J):
    """행 `row` 의 context 를 고정하고 (I x J) 격자에서 eta 를 계산한다.

    ★약물은 «가법 코드»(codes[:,1])와 «잠재 인덱스»(drg) 두 곳에 들어간다.
      둘을 함께 바꿔야 한다.
    """
    ni, nj = len(I), len(J)
    codes = np.repeat(d['codes'][row][None, :], ni * nj, axis=0)
    cov = np.repeat(d['cov'][row][None, :], ni * nj, axis=0)
    nur = np.repeat(np.asarray(I), nj)
    drg = np.tile(np.asarray(J), ni)
    codes[:, 1] = drg                                   # 약물 가법 코드 동기화
    out = m(torch.from_numpy(codes), torch.from_numpy(cov),
            torch.from_numpy(nur), torch.from_numpy(drg))
    return out.numpy().reshape(ni, nj)


def stats_from_grid(E, w_i=None, w_j=None):
    E = np.asarray(E, float)
    d1 = E[:, None, :] - E[None, :, :]                  # (i,i',j) 제공자 대비
    M = d1.mean(2)                                      # 약물 평균 -> 주대비
    D = d1[:, :, :, None] - d1[:, :, None, :]           # (i,i',j,j')
    iu = np.triu_indices(E.shape[0], 1)
    ju = np.triu_indices(E.shape[1], 1)
    Dv = D[iu[0], iu[1]][:, ju[0], ju[1]].ravel()
    Mv = M[iu].ravel()
    an = CA.anova2(E, w_i, w_j)
    return dict(sd_D=float(Dv.std()), sd_M=float(Mv.std()),
                ratio_sdD_sdM=float(Dv.std() / Mv.std()) if Mv.std() > 0 else float('nan'),
                mean_abs_D=float(np.abs(Dv).mean()), max_abs_D=float(np.abs(Dv).max()),
                **{k: float(v) for k, v in an.items()})


def agg(rows):
    ks = [k for k in rows[0] if isinstance(rows[0][k], float)]
    return {k: float(np.mean([r[k] for r in rows])) for k in ks}


def run_fold(fold, d, C, rows_x):
    m = load_B(fold, d)
    res = {}
    for N in (20, 5, 50):
        I, J = pick_grid(C, N)
        if I is None:
            res[f'supported_N{N}'] = dict(note='no grid'); continue
        w_i = C[np.ix_(I, J)].sum(1).astype(float)
        w_j = C[np.ix_(I, J)].sum(0).astype(float)
        per_x = [stats_from_grid(eta_on_grid(m, d, r, I, J), w_i, w_j) for r in rows_x]
        res[f'supported_N{N}'] = dict(n_provider=int(len(I)), n_drug=int(len(J)),
                                      min_cell=int(C[np.ix_(I, J)].min()),
                                      prov_events_med=float(np.median(C.sum(1)[I])),
                                      **agg(per_x))
        if N == 5:      # 가장 넓은 격자에서 제공자 사건수 삼분위 층화
            vol = C.sum(1)[I]
            q = np.quantile(vol, [1/3, 2/3])
            for t, sel in enumerate([vol <= q[0], (vol > q[0]) & (vol <= q[1]), vol > q[1]]):
                if sel.sum() < 3: continue
                It = I[sel]
                wt = C[np.ix_(It, J)].sum(1).astype(float)
                pt = [stats_from_grid(eta_on_grid(m, d, r, It, J), wt, w_j) for r in rows_x]
                res[f'tercile{t+1}_N5'] = dict(n_provider=int(len(It)),
                                               prov_events_med=float(np.median(vol[sel])),
                                               **agg(pt))
    # D_all — support 무시
    I0, J0 = pick_grid(C, 20)
    ni, nj = (len(I0), len(J0)) if I0 is not None else (20, 10)
    Ir, Jr = rand_grid(C, ni, nj, seed=fold)
    per_x = [stats_from_grid(eta_on_grid(m, d, r, Ir, Jr)) for r in rows_x]
    res['all_random'] = dict(n_provider=ni, n_drug=nj,
                             min_cell=int(C[np.ix_(Ir, Jr)].min()), **agg(per_x))
    return res


def main():
    d = data.prep(None, ward=False)
    rng = np.random.default_rng(7)
    rows_x = rng.choice(len(d['y']), size=N_X, replace=False)
    allres = {}
    for fold in range(5):
        if not os.path.exists(CKB + f'B_f{fold}.pt'):
            print(f'fold {fold} 체크포인트 없음 — 건너뜀'); continue
        C, _ = counts(d, fold)
        allres[f'fold{fold}'] = run_fold(fold, d, C, rows_x)
        print(f'fold {fold} 완료', flush=True)
    json.dump(allres, open(OUT + 'D_native_B.json', 'w'), indent=1)

    # 요약
    keys = sorted({k for v in allres.values() for k in v})
    print('\n%-16s %5s %5s %9s %9s %8s %8s' %
          ('격자', '제공', '약물', 'sd(D)', 'sd(M)', 'D/M', 'frac_int'))
    for k in keys:
        rr = [v[k] for v in allres.values() if k in v and 'sd_D' in v[k]]
        if not rr: continue
        f = lambda z: float(np.mean([r[z] for r in rr]))
        print('%-16s %5d %5d %9.5f %9.5f %8.4f %8.4f' %
              (k, rr[0].get('n_provider', 0), rr[0].get('n_drug', 0),
               f('sd_D'), f('sd_M'), f('ratio_sdD_sdM'), f('frac_inter')))


if __name__ == '__main__':
    main()
