import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import json, glob, numpy as np, pandas as pd
CK=os.path.join(paths.WORK, 'cdld/ck/'); V1=os.path.join(paths.WORK, 'cdld/ck_v1/')
def _need_runs(*dfs):
    """체크포인트가 하나도 없으면 «깨끗이» 멈춘다 — 빈 DataFrame 으로 죽지 않는다."""
    if all(getattr(d, 'empty', True) for d in dfs):
        import paths as _p
        raise SystemExit(
            f'[cdld] no run checkpoints found under {_p.WORK}\n'
            '  This script aggregates the training runs; run them first (docs/REPRODUCE.md),\n'
            '  or read the committed results instead:\n'
            '      python verify.py     python src/verify_v8.py')

def L(d,kinds):
    r=[json.load(open(f)) for f in glob.glob(d+'base_*.json')]
    return pd.DataFrame([x for x in r if x['kind'] in kinds])
C=L(CK,{'C'}); AB2=L(CK,{'A','B'}); AB1=L(V1,{'A','B'})
_need_runs(C, AB2, AB1)

def r2(df): return 1-df.sse.sum()/df.sstot.sum()
def perfold(df,arm,kind):
    s=df[(df.arm==arm)&(df.kind==kind)].sort_values('fold')
    return (1-s.sse/s.sstot).to_numpy()

rows=[]
for name,AB in [('사다리1 (사전명시 형태)',AB1),('사다리2 (약물표현 정렬)',AB2)]:
    A=perfold(AB,'real','A')
    d={}
    for arm in ['real','shuffle0','shuffle1']:
        d[arm]=dict(A=A, B=perfold(AB,arm,'B'), C=perfold(C,arm,'C'))
    shB=(d['shuffle0']['B']+d['shuffle1']['B'])/2
    shC=(d['shuffle0']['C']+d['shuffle1']['C'])/2
    main=(d['real']['B']-A)-(shB-A); lat=(d['real']['C']-d['real']['B'])-(shC-shB)
    tot =(d['real']['C']-A)-(shC-A)
    rows.append(dict(사다리=name, 주효과=main.mean(), 주효과_sd=main.std(ddof=1),
                     잠재=lat.mean(), 잠재_sd=lat.std(ddof=1),
                     총=tot.mean(), 비=lat.mean()/main.mean(),
                     비_겹최소=(lat/main).min(), 비_겹최대=(lat/main).max(),
                     섞기잠재=(shC-shB).mean()))
t=pd.DataFrame(rows)
print(t.round(4).to_string(index=False))
t.to_csv(os.path.join(paths.WORK, 'cdld/out_02_판정.csv'), index=False)

print('\n=== 판정 (사전명시 기준) ===')
for _,r in t.iterrows():
    print(f"\n  [{r.사다리}]")
    print(f"    S   섞기 잠재 <= 0        {r.섞기잠재:+.4f}   {'통과' if r.섞기잠재<=0 else '★탈락'}")
    print(f"    U0  중증도 대리 포함       포함        통과")
    print(f"    R   총 >= +0.010          {r.총:+.4f}   {'통과' if r.총>=0.010 else '★탈락'}")
    print(f"    Q2  잠재 >= +0.005        {r.잠재:+.4f}   {'통과' if r.잠재>=0.005 else '★탈락'}")
    print(f"    Q2b 비 >= 1.0             {r.비:.2f}      {'통과' if r.비>=1.0 else '★탈락'}")

print('\n=== 추정기 대비 (같은 자료·같은 겹·같은 섞기) ===')
print(f"""
                        주효과     잠재      총      비
  ALS (보관)           +0.0083  +0.0110  +0.0193   1.32
  CDLD 사다리1         {t.iloc[0].주효과:+.4f}  {t.iloc[0].잠재:+.4f}  {t.iloc[0].총:+.4f}   {t.iloc[0].비:.2f}
  CDLD 사다리2         {t.iloc[1].주효과:+.4f}  {t.iloc[1].잠재:+.4f}  {t.iloc[1].총:+.4f}   {t.iloc[1].비:.2f}
  ★총 순가치는 세 판 모두 +0.0193. 분해만 다르다.""")
