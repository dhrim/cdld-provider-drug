import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
"""체크포인트를 사다리 두 벌로 집계한다."""
import json, glob, os, numpy as np, pandas as pd
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

def load(d, kinds):
    r=[json.load(open(f)) for f in glob.glob(d+'base_*.json')]
    return pd.DataFrame([x for x in r if x['kind'] in kinds])
C  = load(CK, {'C'})
AB2= load(CK, {'A','B'})          # 사다리 2 — A·B 에 약물 잠재 있음
AB1= load(V1, {'A','B'})          # 사다리 1 — 사전명시 형태(약물 주효과만)
_need_runs(C, AB2, AB1)
rows=[]
for name, AB in [('사다리1_사전명시형태', AB1), ('사다리2_약물표현정렬', AB2)]:
    if AB.empty: continue
    A=AB[AB.kind=='A']
    for arm in ['real','shuffle0','shuffle1']:
        rec={'ladder':name,'arm':arm}
        for k,src in [('A',A),('B',AB[(AB.arm==arm)&(AB.kind=='B')]),
                      ('C',C[(C.arm==arm)&(C.kind=='C')])]:
            rec[k]=(1-src.sse.sum()/src.sstot.sum()) if len(src)==5 else np.nan
            rec[k+'_auc']=float(src.auc.mean()) if len(src)==5 else np.nan
            rec[k+'_n']=len(src)
        rows.append(rec)
t=pd.DataFrame(rows)
t.to_csv(os.path.join(paths.WORK, 'cdld/out_01_사다리.csv'), index=False)
print(t.round(4).to_string(index=False))
print()
for name in t.ladder.unique():
    s=t[t.ladder==name].set_index('arm')
    if s[['A','B','C']].notna().all().all():
        sh=s.loc[['shuffle0','shuffle1'],['A','B','C']].mean()
        main=(s.loc['real','B']-s.loc['real','A'])-(sh['B']-sh['A'])
        lat =(s.loc['real','C']-s.loc['real','B'])-(sh['C']-sh['B'])
        tot =(s.loc['real','C']-s.loc['real','A'])-(sh['C']-sh['A'])
        print(f"  [{name}] 주효과 {main:+.4f} · 잠재 {lat:+.4f} · 총 {tot:+.4f} · ★비 {lat/main:.2f}"
              f"  (섞기 잠재 {sh['C']-sh['B']:+.4f})")
    else:
        print(f"  [{name}] 미완 — " + ', '.join(f"{k}:{int(s[k+'_n'].max())}/5" for k in 'ABC'))
