import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import json, glob, numpy as np, pandas as pd
CK=os.path.join(paths.WORK, 'cdld/ck/')
def _need_runs(*dfs):
    """체크포인트가 하나도 없으면 «깨끗이» 멈춘다 — 빈 DataFrame 으로 죽지 않는다."""
    if all(getattr(d, 'empty', True) for d in dfs):
        raise SystemExit(
            f'[cdld] no run checkpoints found under {paths.WORK}\n'
            '  This script aggregates the training runs; run them first (docs/REPRODUCE.md),\n'
            '  or read the committed results instead:\n'
            '      python verify.py     python src/verify_v8.py')

def L(tag,kinds):
    r=[json.load(open(f)) for f in glob.glob(CK+f'{tag}_*.json')]
    return pd.DataFrame([x for x in r if x['kind'] in kinds])
def pf(df,arm,kind):
    s=df[(df.arm==arm)&(df.kind==kind)].sort_values('fold'); return (1-s.sse/s.sstot).to_numpy()
def au(df,arm,kind):
    return float(df[(df.arm==arm)&(df.kind==kind)].auc.mean())

rows=[]
for tag,label in [('base','기본판 (사다리2)'),('wardX','★병동+병동x약물 통제')]:
    D=L(tag,{'A','B','C'})
    _need_runs(D)
    A=pf(D,'real','A'); B=pf(D,'real','B'); C=pf(D,'real','C')
    shB=(pf(D,'shuffle0','B')+pf(D,'shuffle1','B'))/2
    shC=(pf(D,'shuffle0','C')+pf(D,'shuffle1','C'))/2
    main=(B-A)-(shB-A); lat=(C-B)-(shC-shB); tot=(C-A)-(shC-A)
    rows.append(dict(판=label, A=A.mean(), B=B.mean(), C=C.mean(), AUC_C=au(D,'real','C'),
                     주효과=main.mean(), 잠재=lat.mean(), 총=tot.mean(),
                     비=lat.mean()/main.mean(), 섞기잠재=(shC-shB).mean(),
                     주효과_sd=main.std(ddof=1), 잠재_sd=lat.std(ddof=1),
                     비_최소=(lat/main).min(), 비_최대=(lat/main).max()))
t=pd.DataFrame(rows)
print(t[['판','A','B','C','AUC_C']].round(4).to_string(index=False))
print()
print(t[['판','주효과','잠재','총','비','섞기잠재']].round(4).to_string(index=False))
t.to_csv(os.path.join(paths.WORK, 'cdld/out_03_병동관문.csv'), index=False)

print('\n=== 판정 ===')
for _,r in t.iterrows():
    print(f"\n  [{r['판']}]")
    for nm,val,ok in [('S   섞기잠재<=0', r.섞기잠재, r.섞기잠재<=0),
                      ('R   총>=+0.010 ', r.총, r.총>=0.010),
                      ('Q2  잠재>=+0.005', r.잠재, r.잠재>=0.005),
                      ('Q2b 비>=1.0    ', r.비, r.비>=1.0)]:
        print(f"    {nm}  {val:+.4f}   {'통과' if ok else '★탈락'}")

b,w=t.iloc[0],t.iloc[1]
print(f"""
=== 병동이 흡수한 몫 ===
  주효과  {b.주효과:+.4f} -> {w.주효과:+.4f}   잔존 {100*w.주효과/b.주효과:.0f}%
  잠재    {b.잠재:+.4f} -> {w.잠재:+.4f}   잔존 {100*w.잠재/b.잠재:.0f}%
  총      {b.총:+.4f} -> {w.총:+.4f}   잔존 {100*w.총/b.총:.0f}%
  (ALS 판: 총 +0.0193 -> +0.0128, 잔존 66% · 비 1.32 -> 1.14)""")
