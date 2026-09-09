"""D3 집중도 지표 — «실행 전에 고정한» 네 값만 계산한다.

사전 고정 : 20260908_D3_출력량_사전고정.md      (sha256 c10c9ae2…)
            20260908_D3_부록1_출처정책_사전고정.md (sha256 dc067157…)

정의(고정)
    추정기   native CDLD 주분석판 (비제약 읽기층 · 잠재 L2 1e-4)
    순위     «겹 안에서» 예측확률 내림차순. 동점은 원 행 순서로 깬다.
             겹마다 정확히 상위 10%(내림)를 취하고 그 뒤 합친다
    지표     capture  = 상위 10% 안의 documented non-administration 수
                       / 전체 documented non-administration 수          (%)
             obsrate  = 상위 10% 안의 documented non-administration 수
                       / 상위 10% 사건 수 × 1000                        (per 1,000)
    귀무     shuffle0 · shuffle1 · shuffle2 «3팔 평균»
    A 의 팔  A 에는 제공자 항이 없어 팔에 «구조상» 불변 -> real 값을 모든 팔에 쓴다
    출처     A · C_real · C_null 전부 «진단용 복제 계열» (부록 1 참조)

새 지표를 추가하지 않는다. 네 값이 전부다.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data

CD = os.path.dirname(os.path.abspath(__file__)) + '/'
CONC = CD + 'ck_conc/'; CKPT = CD + 'ckpt/'
NULLS = ['shuffle0', 'shuffle1', 'shuffle2']
TOPQ = 0.10


def load(path, key_te='te', key_p='p'):
    z = np.load(path)
    return z[key_te].astype(np.int64), z[key_p].astype(np.float64)


def topdecile(y, te, p):
    """겹 하나. 반환 (상위10% 양성수, 상위10% 사건수, 전체 양성수)."""
    n = len(te); k = int(np.floor(n * TOPQ))
    # 내림차순 정렬. 동점은 «원 행 순서»로 깬다 -> (-p, te) 사전순
    order = np.lexsort((te, -p))
    sel = order[:k]
    yy = y[te]
    return float(yy[sel].sum()), int(k), float(yy.sum())


def arm_metrics(pred_of_fold, y):
    hit = tot_ev = tot_pos = 0.0
    for f in range(5):
        te, p = pred_of_fold(f)
        h, k, pos = topdecile(y, te, p)
        hit += h; tot_ev += k; tot_pos += pos
    return dict(capture=100.0 * hit / tot_pos,           # %
                obsrate=1000.0 * hit / tot_ev,           # per 1,000
                top_events=int(tot_ev), top_positives=int(hit),
                all_positives=int(tot_pos))


def main():
    d = data.prep(None, ward=False); y = d['y'].astype(float)
    A = arm_metrics(lambda f: load(CONC + f'pred_real_f{f}_A.npz'), y)
    C_real = arm_metrics(lambda f: load(CKPT + f'pred_C_f{f}.npz'), y)
    C_nulls = {a: arm_metrics(lambda f, a=a: load(CONC + f'pred_{a}_f{f}_C.npz'), y)
               for a in NULLS}
    C_null = {k: float(np.mean([C_nulls[a][k] for a in NULLS])) for k in ('capture', 'obsrate')}

    rows = []
    for metric, unit in (('capture', '%'), ('obsrate', 'per 1,000 events')):
        a, c, cn = A[metric], C_real[metric], C_null[metric]
        rows.append({
            '지표': ('top-10% capture rate' if metric == 'capture'
                   else 'top-10% observed documented non-administration rate'),
            '단위': unit,
            'A (제공자 없음)': round(a, 1),
            'C (제공자 잠재 포함)': round(c, 1),
            'real incremental gain (C−A)': round(c - a, 1),
            'provider-shuffle-adjusted incremental gain': round(c - cn, 1),
            'C 귀무 3팔 평균': round(cn, 1),
            **{f'C {a_}': round(C_nulls[a_][metric], 1) for a_ in NULLS}})

    import csv
    fp = CD + 'table13_concentration.csv'
    with open(fp, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader()
        [w.writerow(r) for r in rows]

    # 복제 계열의 기록값 일치 (§4.13 형식)
    diffs = {'A': [], 'C_null': []}
    for f in range(5):
        diffs['A'].append(json.load(open(CONC + f'real_f{f}_A.json'))['diff'])
    for a in NULLS:
        for f in range(5):
            diffs['C_null'].append(json.load(open(CONC + f'{a}_f{f}_C.json'))['diff'])
    Cr = [json.load(open(CKPT + f'C_f{f}.json'))['diff'] for f in range(5)]
    agree = {k: dict(n=len(v), mean=float(np.mean(v)), maxabs=float(np.max(np.abs(v))))
             for k, v in (('A', diffs['A']), ('C_real', Cr), ('C_null', diffs['C_null']))}
    json.dump(dict(top_quantile=TOPQ, nulls=NULLS, A=A, C_real=C_real,
                   C_nulls=C_nulls, C_null_mean=C_null, agreement_vs_recorded=agree),
              open(CD + 'table13_concentration.json', 'w'), ensure_ascii=False, indent=1)

    print(f"상위 {TOPQ:.0%} · 겹 안 정렬 후 합산 · 전체 양성 {A['all_positives']:,}건 "
          f"· 상위구간 {A['top_events']:,}건\n")
    hdr = ['지표', 'A', 'C', 'C−A', '귀무보정 증분', 'C 귀무평균']
    print(f"{hdr[0]:<52}{hdr[1]:>8}{hdr[2]:>8}{hdr[3]:>8}{hdr[4]:>14}{hdr[5]:>12}")
    for r, m in zip(rows, ('capture', 'obsrate')):
        print(f"{r['지표'][:50]:<52}{A[m]:>8.1f}{C_real[m]:>8.1f}"
              f"{C_real[m]-A[m]:>8.1f}{C_real[m]-C_null[m]:>14.1f}{C_null[m]:>12.1f}")
    print('\n복제 계열 대 기록 BSS 일치')
    for k, v in agree.items():
        print(f"  {k:<8} n={v['n']:>2}  평균 {v['mean']:+.5f}  최대|차이| {v['maxabs']:.5f}")
    print(f'\n저장: {fp}')


if __name__ == '__main__':
    main()
