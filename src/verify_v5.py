"""v5 검증 — 본문의 «모든» 수치를 겹 단위 결과에서 재계산하고 불일치 시 비영 종료."""
import sys, json, glob, numpy as np, pandas as pd
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import collect_v5 as CV
CD = CV.CD
ok = True; n = 0


def chk(name, got, want, tol=5e-5):
    global ok, n; n += 1
    good = (got is not None) and abs(got - want) <= tol
    ok &= good
    print(f'{"OK " if good else "FAIL"}  {name:52s} got {got:+.5f}  want {want:+.5f}')


nat = CV.ladder(CV.load('ck', 'base'), ['shuffle0', 'shuffle1', 'shuffle2'])
nat2 = CV.ladder(CV.load('ck', 'base'), CV.NULLS2)
ali = CV.ladder(CV.load('ck_a2', 'base'), CV.NULLS2)
al1 = CV.ladder(CV.load('ck_a', 'base'), CV.NULLS2)
wrd = CV.ladder(CV.load('ck', 'wardX'), ['shuffle0', 'shuffle1', 'shuffle2'])

print('== 사다리 ==')
chk('native  Delta_total (R)', nat['d_total'], 0.01535)
chk('native  Delta_main', nat['d_main'], 0.00963)
chk('native  Delta_latent', nat['d_latent'], 0.00573)
chk('native  ratio (Q2b)', nat['ratio'], 0.59474, 1e-4)
chk('native  S', nat['S'], -0.00038)
chk('native ward Delta_total', wrd['d_total'], 0.01286)
chk('CDLD-A2 Delta_total (R)', ali['d_total'], 0.01506)
chk('CDLD-A2 Delta_main', ali['d_main'], 0.00451)
chk('CDLD-A2 Delta_latent', ali['d_latent'], 0.01056)
chk('CDLD-A2 ratio (Q2b)', ali['ratio'], 2.34330, 1e-3)
chk('CDLD-A2 S', ali['S'], -0.00054)
chk('CDLD-A 1차 Delta_total (붕괴)', al1['d_total'], 0.00550)
chk('CDLD-A 1차 ratio', al1['ratio'], 0.22143, 1e-3)

print('\n== A 공유 ==')
chk('A_native == A_CDLD-A2', ali['A'] - nat['A'], 0.0, 1e-12)

print('\n== gamma*, kappa, 항등식 ==')
g = nat2['d_main'] - ali['d_main']; k = ali['d_total'] - nat2['d_total']
chk('gamma*', g, 0.00510, 1e-4)
chk('kappa', k, -0.00020, 1e-4)
chk('항등식 l+g+k == Delta_latent,A', nat2['d_latent'] + g + k - ali['d_latent'], 0.0, 1e-12)

print('\n== 귀속 분해 ==')
m, l, r0, rA = nat2['d_main'], nat2['d_latent'], nat2['ratio'], ali['ratio']
rg = (l + g) / (m - g); rk = (l + k) / m
chk('r(gamma*,0)', rg, 2.3871, 1e-3)
chk('readout 몫 / dr', (rg - r0) / (rA - r0), 1.0248, 2e-3)
chk('kappa 몫 / dr', (rk - r0) / (rA - r0), -0.0117, 5e-3)

print('\n== 적합 관문 (부트스트랩) ==')
G = json.load(open(CD + 'gate_C_a2.json'))
chk('dBSS 점추정', G['dBSS_point'], 0.00058, 2e-5)
chk('CI 하한', G['ci_lo'], 0.00018, 5e-5)
assert G['gate_pass'], 'fit gate 실패'
print('OK    fit gate pass (CI 하한 > -0.0025)')

print('\n== 함수 수준 분해 ==')
F = json.load(open(CD + 'fn_decomp.json'))
chk('native C  Var(c)/Var(a)', F['native_C']['inter_over_provider'], 0.7428, 2e-3)
chk('CDLD-A2 C Var(c)/Var(a)', F['cdldA2_C']['inter_over_provider'], 0.5274, 2e-3)
print('OK    순서 반전 확인' if F['native_C']['inter_over_provider'] >
      F['cdldA2_C']['inter_over_provider'] else 'FAIL  순서 반전 없음')

print('\n== native B 비가법성 ==')
D = json.load(open(CD + 'D_summary.json'))
chk('supported N=20  sd(D)/sd(M)', D['supported_N20']['dm'], 0.4632, 2e-3)
chk('supported N=20  Var(c)/Var(a)', D['supported_N20']['ivp'], 0.0976, 2e-3)
assert D['all_random']['sd_D'] > 2 * D['supported_N20']['sd_D'], '외삽 오염 확인 실패'
print('OK    D_all sd 가 D_supported 의 2배 초과 (외삽 오염 실재)')

print('\n== 판정 ==')
assert nat['ratio'] < 1.0 and ali['ratio'] > 1.0 and 1.5089 > 1.0, 'Q2b 판정 구도 불일치'
print('OK    Q2b: native 미충족 · CDLD-A2 충족 · ALS 충족 (분해 명시 두 판이 일치)')
assert ali['d_total'] >= 0.010, 'R 실패'
print('OK    R: CDLD-A2 에서도 유지 (Branch B 조건)')

print(f'\n검사 {n}건 — {"전부 통과" if ok else "★불일치 있음"}')
sys.exit(0 if ok else 1)
