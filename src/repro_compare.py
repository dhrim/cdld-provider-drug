"""D8 — 원본 대 재실행(fresh reproducibility run) 대조.

계획 동결: 20260909_D8_층B_재현_사전고정.md
sha256 c417b6d7d2b06b045f29d818abc4b99480ccf0a32ea3985ff5b099291f46f79f
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
R = os.path.join(paths.RESULTS, 'v9'); O = os.path.join(R, '_orig')


def canon(df):
    """cluster_resid 를 «사건 수 가중 관측률 오름차순»으로 다시 매긴다."""
    w = df.groupby('cluster_resid').apply(
        lambda g: np.average(g.rate, weights=g.n), include_groups=False)
    order = list(w.sort_values().index)
    m = {c: i for i, c in enumerate(order)}
    out = df.copy(); out['canon'] = out.cluster_resid.map(m); return out


a = canon(pd.read_csv(os.path.join(O, 'out_04_drug_direction_clusters.csv')))
b = canon(pd.read_csv(os.path.join(R, 'out_04_drug_direction_clusters.csv')))
j = a[['drug', 'n', 'rate', 'canon']].merge(
    b[['drug', 'canon']], on='drug', how='inner', suffixes=('_orig', '_new'))

agree_w = float((j.n * (j.canon_orig == j.canon_new)).sum() / j.n.sum())
agree_u = float((j.canon_orig == j.canon_new).mean())

print(f'① 공통 약물 {len(j)}종 (원본 {len(a)} · 재실행 {len(b)})')
print(f'② 사건 수 가중 멤버십 일치도  {agree_w:.3f}   (문턱 0.70)  '
      f'{"PASS" if agree_w >= .70 else "FAIL"}')
print(f'   (약물 수 기준 일치도       {agree_u:.3f})')

s_o = pd.read_csv(os.path.join(O, 'out_05_cluster_summary.csv')).sort_values('누락률')
s_n = pd.read_csv(os.path.join(R, 'out_05_cluster_summary.csv')).sort_values('누락률')
cmp = pd.DataFrame({
    'Cluster': [1, 2, 3, 4],
    '약물수_원본': s_o.약물수.values, '약물수_재실행': s_n.약물수.values,
    '건수_원본': s_o.건수.values, '건수_재실행': s_n.건수.values,
    '관측률_원본': (s_o.누락률.values * 100).round(1),
    '관측률_재실행': (s_n.누락률.values * 100).round(1)})
print('\n' + cmp.to_string(index=False))
sp_o = s_o.누락률.max() / s_o.누락률.min(); sp_n = s_n.누락률.max() / s_n.누락률.min()
print(f'\n③ 최대/최소 비  원본 {sp_o:.2f} · 재실행 {sp_n:.2f}  (문턱 2.0)  '
      f'{"PASS" if sp_n >= 2.0 else "FAIL"}')
print(f'   각 군집 약물 수 >= 40 : {"PASS" if s_n.약물수.min() >= 40 else "FAIL"}'
      f' (최소 {int(s_n.약물수.min())})')

p_o = pd.read_csv(os.path.join(O, 'out_06_provider_profiles.csv'))
p_n = pd.read_csv(os.path.join(R, 'out_06_provider_profiles.csv'))
print('\n⑤ argmax 짝'); print('  원본  :', p_o.to_dict('records'))
print('  재실행:', p_n.to_dict('records'))
same = (sorted(p_o.n.tolist()) == sorted(p_n.n.tolist()))
print(f'  같은 두 제공자인가(n 으로 판정): {"예" if same else "★아니오"}')

t_o = pd.read_csv(os.path.join(O, 'table17_reversal_replicate.csv'))
t_n = pd.read_csv(os.path.join(R, 'table17_reversal_replicate.csv'))
r_o = float(t_o[t_o.통계 == '중앙값'].배.iloc[0]); r_n = float(t_n[t_n.통계 == '중앙값'].배.iloc[0])
print(f'\n④ 반전폭 중앙값 배  원본 {r_o:.2f} · 재실행 {r_n:.2f}  (문턱 1.3)  '
      f'{"PASS" if r_n >= 1.3 else "FAIL"}')

pd.DataFrame([dict(가중일치도=agree_w, 약물수일치도=agree_u,
                   최대최소비_원본=sp_o, 최대최소비_재실행=sp_n,
                   반전폭배_원본=r_o, 반전폭배_재실행=r_n,
                   argmax_동일=same)]).to_csv(
    os.path.join(R, 'table19_repro_check.csv'), index=False)
print('\n-> table19_repro_check.csv')
