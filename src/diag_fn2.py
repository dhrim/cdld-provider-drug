"""함수 수준 분해 — 보완판 (사전 지표 복구 · x 분포 · 상호작용 성분 상관).

사전 고정 지표를 primary 로 되돌린다.
  primary   frac_inter = Var(c)/[Var(a)+Var(b)+Var(c)]  ·  sd(D)/sd(M)
  secondary Var(c)/Var(a)   — Q2b 와 형식이 대응되나 «사후» 지표임을 명시

x 20개를 «분포» 로 보고한다 (현재는 평균만 저장했다).

★ corr(c_native, c_aligned) — 상호작용 «성분 자체» 의 상관.
   raw eta 격자의 상관은 쓰지 않는다. 격자 분산의 90% 이상이 약물 주효과라
   두 모형 모두 거의 1 이 나와, 「비슷한 함수」라는 과잉 주장을 수치로
   뒷받침하는 모양이 되기 때문이다.
"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data, cdld_core as CC, cdld_a as CA, diag_d as DD
import paths
CD = paths.WORK + os.sep; CKB = CD + 'ckpt/'; CKA2 = CD + 'ck_a2/'; OUT = CD + 'diag/'


def two_way(E, wi, wj):
    """격자 위 함수의 이원 분해. wi, wj 는 «주변» 빈도이며 곱 가중 wi*wj 를 쓴다."""
    E = np.asarray(E, float)
    wi = np.asarray(wi, float) / np.sum(wi); wj = np.asarray(wj, float) / np.sum(wj)
    mu = (wi[:, None] * wj[None, :] * E).sum()
    a = (wj[None, :] * E).sum(1) - mu
    b = (wi[:, None] * E).sum(0) - mu
    c = E - mu - a[:, None] - b[None, :]
    Va = (wi * a ** 2).sum(); Vb = (wj * b ** 2).sum()
    Vc = (wi[:, None] * wj[None, :] * c ** 2).sum()
    return dict(Va=Va, Vb=Vb, Vc=Vc, frac_inter=Vc / (Va + Vb + Vc),
                inter_over_prov=Vc / Va), c, (wi, wj)


def dm_stats(E):
    E = np.asarray(E, float)
    d1 = E[:, None, :] - E[None, :, :]
    M = d1.mean(2)
    D = d1[:, :, :, None] - d1[:, :, None, :]
    iu = np.triu_indices(E.shape[0], 1); ju = np.triu_indices(E.shape[1], 1)
    Dv = D[iu[0], iu[1]][:, ju[0], ju[1]].ravel(); Mv = M[iu].ravel()
    return Dv.std(), Mv.std()


def wcorr(x, y, w):
    x = x.ravel(); y = y.ravel(); w = w.ravel() / w.sum()
    mx = (w * x).sum(); my = (w * y).sum()
    cxy = (w * (x - mx) * (y - my)).sum()
    return cxy / np.sqrt((w * (x - mx) ** 2).sum() * (w * (y - my) ** 2).sum())


def q(v):
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()), median=float(np.median(v)),
                q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)),
                min=float(v.min()), max=float(v.max()), n=int(len(v)))


def main():
    d = data.prep(None, ward=False)
    rng = np.random.default_rng(7)
    rows_x = rng.choice(len(d['y']), size=DD.N_X, replace=False)
    acc = {}
    for fold in range(5):
        if not os.path.exists(CKB + f'C_f{fold}.pt'): continue
        ck = torch.load(CKB + f'C_f{fold}.pt', map_location='cpu', weights_only=False)
        mn = CC.Finder(ck['sizes'], ck['ncov'], n_nurse=ck['n_nurse'],
                       n_drug=ck['n_drug'], latent=True)
        mn.load_state_dict(ck['state']); mn.eval()
        ma = CA.AlignedFinder(list(d['sizes']), d['cov'].shape[1],
                              int(d['n_nurse']), int(d['n_drug']), 'C')
        ma.load_state_dict(torch.load(CKA2 + f'w_base_real_f{fold}_C.pt',
                                      map_location='cpu', weights_only=False)); ma.eval()
        C, _ = DD.counts(d, fold)
        I, J = DD.pick_grid(C, 20)
        sub = C[np.ix_(I, J)]
        wi = sub.sum(1).astype(float); wj = sub.sum(0).astype(float)
        for r in rows_x:
            En = DD.eta_on_grid(mn, d, r, I, J)
            Ea = DD.eta_on_grid(ma, d, r, I, J)
            sn, cn, (Wi, Wj) = two_way(En, wi, wj)
            sa, ca, _ = two_way(Ea, wi, wj)
            sdDn, sdMn = dm_stats(En); sdDa, sdMa = dm_stats(Ea)
            W = Wi[:, None] * Wj[None, :]
            acc.setdefault('native_frac_inter', []).append(sn['frac_inter'])
            acc.setdefault('aligned_frac_inter', []).append(sa['frac_inter'])
            acc.setdefault('native_sdD_sdM', []).append(sdDn / sdMn)
            acc.setdefault('aligned_sdD_sdM', []).append(sdDa / sdMa)
            acc.setdefault('native_inter_over_prov', []).append(sn['inter_over_prov'])
            acc.setdefault('aligned_inter_over_prov', []).append(sa['inter_over_prov'])
            acc.setdefault('corr_c', []).append(wcorr(cn, ca, W))
            acc.setdefault('corr_eta_raw', []).append(wcorr(En, Ea, W))
        print(f'fold {fold} 완료', flush=True)
    out = {k: q(v) for k, v in acc.items()}
    out['_note'] = ('weighting = product of provider and drug marginal frequencies '
                    'within the selected grid; grid = 40 providers x 20 drugs, '
                    'high-support subset of 936 x 522; 20 fixed event contexts x 5 folds')
    json.dump(out, open(OUT + 'fn_decomp2.json', 'w'), indent=1)

    def show(k, lab):
        s = out[k]
        print(f'  {lab:34s} {s["mean"]:7.4f}  중앙 {s["median"]:7.4f}  '
              f'IQR [{s["q25"]:.4f}, {s["q75"]:.4f}]  범위 [{s["min"]:.4f}, {s["max"]:.4f}]')
    print('\n=== 사전 고정 primary ===')
    show('native_frac_inter', 'native   frac_inter')
    show('aligned_frac_inter', 'CDLD-A   frac_inter')
    show('native_sdD_sdM', 'native   sd(D)/sd(M)')
    show('aligned_sdD_sdM', 'CDLD-A   sd(D)/sd(M)')
    print('\n=== 사후 secondary (Q2b 대응) ===')
    show('native_inter_over_prov', 'native   Var(c)/Var(a)')
    show('aligned_inter_over_prov', 'CDLD-A   Var(c)/Var(a)')
    print('\n=== 상호작용 성분의 일치 ===')
    show('corr_c', '★corr(c_native, c_aligned)')
    show('corr_eta_raw', '(참고) corr(eta_native, eta_aligned)')


if __name__ == '__main__':
    main()
