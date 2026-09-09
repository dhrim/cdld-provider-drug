"""보완 실험 진행률 — 52단위."""
import os, glob, json
import paths
CD = paths.WORK + os.sep
SPEC=[('A  native C 감쇠제외','ck_natnd',[('base',a,f,'C') for a in ('real','shuffle0','shuffle1') for f in range(5)]),
      ('B  CDLD-A C L2 1e-5','ck_a2mid',[('base',a,f,'C') for a in ('real','shuffle0','shuffle1') for f in range(5)]),
      ('C  CDLD-A 3번째 귀무','ck_a2',[('base','shuffle2',f,k) for k in ('B','C') for f in range(5)]),
      ('D  시드 반복','ck_seed',[('base',f'seed{s}',f,k) for k in ('natB','aliB') for s in (1,2) for f in range(3)])]
tot=done=0; lines=[]
for name,d,units in SPEC:
    n=sum(1 for t,a,f,k in units if os.path.exists(CD+d+f'/{t}_{a}_f{f}_{k}.json'))
    tot+=len(units); done+=n
    lines.append(f'  {name:24s} {n:2d}/{len(units):2d}' + ('  ✅' if n==len(units) else ''))
print('\n'.join(lines))
print(f'  {"합계":24s} {done:2d}/{tot:2d}   ({done/tot*100:.0f}%)')
