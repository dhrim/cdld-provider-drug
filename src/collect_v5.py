"""v5 결과 집계 — 모든 표를 겹 단위 결과에서 «재계산»해 CSV 로 낸다."""
import os, sys, json, glob, numpy as np, pandas as pd
import os
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CD = os.path.join(_HERE, 'results', 'v5') + os.sep    # 저장소 자립 경로
OUT = os.path.join(_HERE, 'results', 'v5') + os.sep
NULLS2 = ['shuffle0', 'shuffle1']


def load(dirname, tag='base'):
    """겹 단위 결과 json 을 읽는다. 저장소에서는 results/v5/fold_json/<dir>__<name>.json 이다."""
    rows = []
    pats = [CD + f'fold_json/{dirname}__{tag}_*_f?_?.json',   # 저장소 배치
            CD + dirname + f'/{tag}_*_f?_?.json']             # 원 작업 배치
    for fp in [f for p in pats for f in glob.glob(p)]:
        r = json.load(open(fp)); r['bss'] = 1 - r['sse'] / r['sstot']; rows.append(r)
    return pd.DataFrame(rows)


def ladder(df, nulls):
    g = lambda arm, k: df[(df.arm == arm) & (df.kind == k)].bss.mean()
    gn = lambda k: df[df.arm.isin(nulls) & (df.kind == k)].bss.mean()
    A, B, C, Bn, Cn = g('real', 'A'), g('real', 'B'), g('real', 'C'), gn('B'), gn('C')
    dm, dt = B - Bn, C - Cn
    return dict(A=A, B=B, C=C, d_main=dm, d_latent=dt - dm, d_total=dt,
                ratio=(dt - dm) / dm, S=Cn - Bn,
                auc_C=df[(df.arm == 'real') & (df.kind == 'C')].auc.mean())


def main():
    rows = []
    specs = [('순환 이중 잠재 (비제약 읽기층)', 'ck', 'base', ['shuffle0', 'shuffle1', 'shuffle2']),
             ('순환 이중 잠재 — 병동 통제', 'ck', 'wardX', ['shuffle0', 'shuffle1', 'shuffle2']),
             ('순환 이중 잠재 (귀무2, 정합비교)', 'ck', 'base', NULLS2),
             ('분해 정합 변종 CDLD-A 1차(붕괴)', 'ck_a', 'base', NULLS2),
             ('분해 정합 변종 CDLD-A', 'ck_a2', 'base', NULLS2),
             ('분해 정합 변종 — 병동 통제', 'ck_a2w', 'wardX', NULLS2)]
    for name, d, tag, nl in specs:
        df = load(d, tag)
        if df.empty or df[df.kind == 'C'].empty: continue
        n_units = len(df)
        r = ladder(df, nl); r['설정'] = name; r['단위수'] = n_units
        rows.append(r)
    # ALS (발표값)
    for name, v in (('저랭크 이중선형', dict(A=.12861, B=.13526, C=.14577, d_main=.00751,
                                        d_latent=.01134, d_total=.01885, ratio=1.5089,
                                        S=-.00082, auc_C=np.nan, 단위수=np.nan)),
                    ('저랭크 이중선형 — 병동 통제', dict(A=.13808, B=.14257, C=.14859,
                                                d_main=.00522, d_latent=.00770,
                                                d_total=.01293, ratio=1.4752,
                                                S=-.00169, auc_C=np.nan, 단위수=np.nan))):
        v['설정'] = name; rows.append(v)
    t = pd.DataFrame(rows)[['설정', '단위수', 'A', 'B', 'C', 'auc_C', 'd_main', 'd_latent',
                            'd_total', 'ratio', 'S']]
    t['R기준'] = np.where(t.d_total >= .010, '충족', '미충족')
    t['Q2'] = np.where(t.d_latent >= .005, '충족', '미충족')
    t['Q2b'] = np.where(t.ratio >= 1.0, '충족', '미충족')
    t['S기준'] = np.where(t.S <= 0, '충족', '미충족')
    t.to_csv(OUT + 'table1_ladders.csv', index=False, float_format='%.5f')
    print(t.to_string(index=False, float_format=lambda z: f'{z:+.5f}'))

    # 겹 단위 원자료
    frames = []
    for name, d, tag, _ in specs:
        df = load(d, tag)
        if df.empty: continue
        df['설정'] = name; frames.append(df)
    if frames:
        pd.concat(frames).to_csv(OUT + 'fold_level_v5.csv', index=False)
    # 진단 표
    for src, dst in (('diag/adj_a2.json', 'table2_attribution.json'),
                     ('diag/gate_C_a2.json', 'table3_fitgate.json'),
                     ('diag/fn_decomp.json', 'table4_function_decomp.json'),
                     ('diag/D_summary.json', 'table5_D_native_B.json')):
        if os.path.exists(CD + src):
            json.dump(json.load(open(CD + src)), open(OUT + dst, 'w'), indent=1)
    print('\n저장:', OUT)


if __name__ == '__main__':
    main()
