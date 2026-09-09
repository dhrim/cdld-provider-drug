"""함수 수준 이원 분산분해 비교 — native C vs CDLD-A2 C.

같은 고정 context x, 같은 (제공자 x 약물) 격자, 같은 가중에서
  eta(i,j;x) = mu + a_i + b_j + c_ij
의 분산 분해를 두 모형에 «동일한 절차»로 적용한다.

핵심 물음 : 예측 기술의 분해(Q2b)는 두 모형에서 갈렸다(0.59 vs 2.34).
            «함수»의 분해도 갈리는가?
"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data, cdld_core as CC, cdld_a as CA, diag_d as DD

import paths
CD = paths.WORK + os.sep; CKB = CD + 'ckpt/'; CKA2 = CD + 'ck_a2/'
OUT = CD + 'diag/'


def load_native_C(fold):
    ck = torch.load(CKB + f'C_f{fold}.pt', map_location='cpu', weights_only=False)
    m = CC.Finder(ck['sizes'], ck['ncov'], n_nurse=ck['n_nurse'],
                  n_drug=ck['n_drug'], latent=True)
    m.load_state_dict(ck['state']); m.eval(); return m


def load_a2_C(fold, d):
    sd = torch.load(CKA2 + f'w_base_real_f{fold}_C.pt', map_location='cpu',
                    weights_only=False)
    m = CA.AlignedFinder(list(d['sizes']), d['cov'].shape[1],
                         int(d['n_nurse']), int(d['n_drug']), 'C')
    m.load_state_dict(sd); m.eval(); return m


def main():
    d = data.prep(None, ward=False)
    rng = np.random.default_rng(7)
    rows_x = rng.choice(len(d['y']), size=DD.N_X, replace=False)
    res = {}
    for fold in range(5):
        if not os.path.exists(CKB + f'C_f{fold}.pt'): continue
        C, _ = DD.counts(d, fold)
        I, J = DD.pick_grid(C, 20)
        w_i = C[np.ix_(I, J)].sum(1).astype(float)
        w_j = C[np.ix_(I, J)].sum(0).astype(float)
        for name, m in (('native_C', load_native_C(fold)),
                        ('cdldA2_C', load_a2_C(fold, d))):
            per_x = [DD.stats_from_grid(DD.eta_on_grid(m, d, r, I, J), w_i, w_j)
                     for r in rows_x]
            res.setdefault(name, []).append(DD.agg(per_x))
        print(f'fold {fold} 완료', flush=True)
    summ = {}
    print('\n%-12s %9s %9s %8s %10s %10s %10s %12s' %
          ('모형', 'sd(D)', 'sd(M)', 'D/M', 'Var(a)', 'Var(b)', 'Var(c)', 'Var(c)/Var(a)'))
    for name, rows in res.items():
        f = lambda z: float(np.mean([r[z] for r in rows]))
        s = dict(sd_D=f('sd_D'), sd_M=f('sd_M'), ratio_sdD_sdM=f('ratio_sdD_sdM'),
                 V_provider=f('V_provider'), V_drug=f('V_drug'), V_inter=f('V_inter'),
                 inter_over_provider=f('V_inter') / f('V_provider'),
                 frac_inter=f('frac_inter'))
        summ[name] = s
        print('%-12s %9.5f %9.5f %8.4f %10.5f %10.5f %10.5f %12.4f' %
              (name, s['sd_D'], s['sd_M'], s['ratio_sdD_sdM'],
               s['V_provider'], s['V_drug'], s['V_inter'], s['inter_over_provider']))
    json.dump(summ, open(OUT + 'fn_decomp.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
