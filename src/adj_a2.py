"""CDLD-A2 판정 — 사전 문서 §9 의 귀속 분해 규칙을 «그대로» 실행한다.

  gamma* = Delta_main,native - Delta_main,aligned      (A 공유로 잔차 없음)
  kappa  = Delta_total,aligned - Delta_total,native
  항등식 : Delta_latent,aligned = l + gamma* + kappa
  r_A    = (l + gamma* + kappa) / (m - gamma*)

  귀속 : r(gamma*,0) = (l+gamma*)/(m-gamma*)   readout 몫
         r(0,kappa)  = (l+kappa)/m             총신호 이동 몫
         잔차       = dr - (두 몫의 합)

★ native 와 CDLD-A2 를 «같은 귀무 팔»(shuffle0, shuffle1)로 맞춰 비교한다.
  본실험 발표값(귀무 3회)은 참고로 함께 낸다.
"""
import os, sys, json, glob, numpy as np
import paths
CD = paths.WORK + os.sep
DELTA = 0.0025            # fit gate margin
R_THRESH = 0.010          # R 기준
NULLS = ['shuffle0', 'shuffle1']


def load(dirname, tag='base'):
    out = {}
    for fp in glob.glob(CD + dirname + f'/{tag}_*_f?_?.json'):
        r = json.load(open(fp))
        out[(r['arm'], r['fold'], r['kind'])] = 1 - r['sse'] / r['sstot']
    return out


def ladder(bss, nulls, folds=range(5)):
    def mean(arm, kind):
        v = [bss[(arm, f, kind)] for f in folds if (arm, f, kind) in bss]
        return float(np.mean(v)) if v else float('nan')

    def mean_null(kind):
        v = [bss[(a, f, kind)] for a in nulls for f in folds if (a, f, kind) in bss]
        return float(np.mean(v)) if v else float('nan')

    A, B, C = mean('real', 'A'), mean('real', 'B'), mean('real', 'C')
    Bn, Cn = mean_null('B'), mean_null('C')
    d_main = B - Bn                       # A 는 팔 무관 -> 상쇄
    d_total = C - Cn
    d_latent = d_total - d_main
    return dict(A=A, B=B, C=C, B_null=Bn, C_null=Cn,
                d_main=d_main, d_latent=d_latent, d_total=d_total,
                ratio=d_latent / d_main if d_main else float('nan'))


def main():
    nat = load('ck'); ali = load('ck_a2')
    n_ali = len([k for k in ali if k[2] in 'BC'])
    print(f'CDLD-A2 완료 단위 {n_ali}/30 (B,C x 3팔 x 5겹)\n')

    L_nat3 = ladder(nat, ['shuffle0', 'shuffle1', 'shuffle2'])
    L_nat = ladder(nat, NULLS)
    L_ali = ladder(ali, NULLS)

    def show(name, L):
        print(f'{name:22s} A {L["A"]:.5f}  B {L["B"]:.5f}  C {L["C"]:.5f}  | '
              f'main {L["d_main"]:+.5f}  latent {L["d_latent"]:+.5f}  '
              f'total {L["d_total"]:+.5f}  비 {L["ratio"]:.4f}')
    show('native (귀무 3, 발표)', L_nat3)
    show('native (귀무 2, 정합)', L_nat)
    show('CDLD-A2 (귀무 2)', L_ali)

    m, l, r_nat = L_nat['d_main'], L_nat['d_latent'], L_nat['ratio']
    gamma = L_nat['d_main'] - L_ali['d_main']
    kappa = L_ali['d_total'] - L_nat['d_total']
    r_A = L_ali['ratio']
    ident = l + gamma + kappa
    print(f'\ngamma*  = {gamma:+.5f}   (native B - CDLD-A2 B, 귀무 보정)')
    print(f'kappa   = {kappa:+.5f}   (총 포함 효과 이동)')
    print(f'항등식 검산 : Delta_latent,A 관측 {L_ali["d_latent"]:+.5f} vs '
          f'l+g+k {ident:+.5f}  오차 {L_ali["d_latent"]-ident:+.2e}')

    dr = r_A - r_nat
    r_g = (l + gamma) / (m - gamma) if (m - gamma) else float('nan')
    r_k = (l + kappa) / m
    sh_g, sh_k = r_g - r_nat, r_k - r_nat
    resid = dr - sh_g - sh_k
    print(f'\n== 귀속 분해 ==')
    print(f'  r_native {r_nat:.4f} -> r_aligned {r_A:.4f}   (dr {dr:+.4f})')
    print(f'  readout 제약 몫  r(g,0)={r_g:.4f}   기여 {sh_g:+.4f}  ({sh_g/dr*100 if dr else float("nan"):+.0f}%)')
    print(f'  총신호 이동 몫   r(0,k)={r_k:.4f}   기여 {sh_k:+.4f}  ({sh_k/dr*100 if dr else float("nan"):+.0f}%)')
    print(f'  잔차(교호)                        {resid:+.4f}  ({resid/dr*100 if dr else float("nan"):+.0f}%)')

    print(f'\n== 판정 ==')
    R_ok = L_ali['d_total'] >= R_THRESH
    print(f'  R (Delta_total,A) {L_ali["d_total"]:+.5f}  vs 기준 {R_THRESH:+.4f}  -> '
          f'{"유지" if R_ok else "★실패"}')
    print(f'  kappa 임계 {-(L_nat["d_total"]-R_THRESH):+.5f} -> kappa {kappa:+.5f} '
          f'{"안전" if kappa > -(L_nat["d_total"]-R_THRESH) else "★초과"}')
    print(f'  Q2b(CDLD-A) 비 {r_A:.4f} vs 1.0 -> {"통과" if r_A >= 1 else "실패"}')
    print(f'  C 단계 BSS  native {L_nat["C"]:.5f}  CDLD-A2 {L_ali["C"]:.5f}  '
          f'차 {L_ali["C"]-L_nat["C"]:+.5f}  (관문 margin -{DELTA})  '
          f'-> {"점추정 통과" if L_ali["C"]-L_nat["C"] > -DELTA else "★점추정 실패"} '
          f'(부트스트랩 CI 로 확정)')

    json.dump(dict(native3=L_nat3, native2=L_nat, aligned=L_ali,
                   gamma=gamma, kappa=kappa, r_nat=r_nat, r_A=r_A, dr=dr,
                   share_readout=sh_g, share_kappa=sh_k, resid=resid,
                   R_ok=bool(R_ok), delta=DELTA),
              open(CD + 'diag/adj_a2.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
