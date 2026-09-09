"""v6 집계 — 3-null 통일 · 2x2 · 함수 분해 · 귀무 빈도 표."""
import os, sys, json, glob, numpy as np, pandas as pd
import paths
CD = paths.WORK + os.sep; OUT=paths.RESULTS + '/v6/'
N3=['shuffle0','shuffle1','shuffle2']; N2=['shuffle0','shuffle1']

def load(d,tag='base'):
    o={}
    for fp in glob.glob(CD+d+f'/{tag}_*_f?_?.json'):
        r=json.load(open(fp)); o[(r['arm'],r['fold'],r['kind'])]=(1-r['sse']/r['sstot'], r.get('auc'))
    return o

def lad(Asrc,Bsrc,Csrc,N):
    g=lambda s,a,f,k: s[(a,f,k)][0]
    need=lambda s,k: all((a,f,k) in s for a in N for f in range(5))
    if not (need(Bsrc,'B') and need(Csrc,'C')): return None
    A=np.mean([g(Asrc,'real',f,'A') for f in range(5)])
    B=np.mean([g(Bsrc,'real',f,'B') for f in range(5)])
    Bn=np.mean([g(Bsrc,a,f,'B') for a in N for f in range(5)])
    C=np.mean([g(Csrc,'real',f,'C') for f in range(5)])
    Cn=np.mean([g(Csrc,a,f,'C') for a in N for f in range(5)])
    dm=B-Bn; dt=C-Cn
    auc=np.mean([Csrc[('real',f,'C')][1] for f in range(5) if Csrc[('real',f,'C')][1]==Csrc[('real',f,'C')][1]])
    return dict(A=A,B=B,C=C,auc_C=auc,d_main=dm,d_latent=dt-dm,d_total=dt,
                ratio=(dt-dm)/dm,S=Cn-Bn,n_null=len(N))

def main():
    S={k:load(k) for k in ('ck','ck_natnd','ck_a','ck_a2','ck_a2mid','ck_a2w')}
    W=load('ck','wardX'); Ww=load('ck_a2w','wardX')
    rows=[]
    def add(name,readout,l2,Asrc,Bsrc,Csrc,N):
        L=lad(Asrc,Bsrc,Csrc,N)
        if L: rows.append(dict(설정=name, readout=readout, latent_L2=l2, **L))
    add('순환 이중 잠재 (native)','비제약','1e-4',S['ck'],S['ck'],S['ck'],N3)
    add('native — 잠재표 감쇠 제외','비제약','0',S['ck'],S['ck'],S['ck_natnd'],N2)
    add('CDLD-A 1차 (붕괴)','정합','1e-4',S['ck'],S['ck_a'],S['ck_a'],N2)
    add('CDLD-A — 잠재표 L2 1e-5','정합','1e-5',S['ck'],S['ck_a2'],S['ck_a2mid'],N2)
    add('CDLD-A (본판)','정합','0',S['ck'],S['ck_a2'],S['ck_a2'],N3)
    Lw=lad(W,W,W,N3); Lwa=lad(W,Ww,Ww,N2)
    if Lw: rows.append(dict(설정='native — 병동 통제',readout='비제약',latent_L2='1e-4',**Lw))
    if Lwa: rows.append(dict(설정='CDLD-A — 병동 통제',readout='정합',latent_L2='0',**Lwa))
    for nm,v in (('저랭크 이중선형 (ALS)',dict(A=.12861,B=.13526,C=.14577,auc_C=np.nan,
                    d_main=.00751,d_latent=.01124,d_total=.01875,ratio=1.498,S=-.00073,n_null=3)),
                 ('ALS — 병동 통제',dict(A=.13808,B=.14257,C=.14859,auc_C=np.nan,
                    d_main=.00522,d_latent=.00770,d_total=.01293,ratio=1.475,S=-.00169,n_null=2)),
                 ('ALS — Hold Dose 제외',dict(A=.13052,B=.13680,C=.14740,auc_C=np.nan,
                    d_main=.00714,d_latent=.01132,d_total=.01846,ratio=1.586,S=-.00072,n_null=2)),
                 ('ALS — Sliding Scale 포함',dict(A=.18042,B=.18560,C=.19530,auc_C=np.nan,
                    d_main=.00592,d_latent=.01085,d_total=.01677,ratio=1.833,S=-.00115,n_null=2))):
        rows.append(dict(설정=nm,readout='정합',latent_L2='—',**v))
    t=pd.DataFrame(rows)
    t['R기준']=np.where(t.d_total>=.010,'충족','미충족'); t['Q2']=np.where(t.d_latent>=.005,'충족','미충족')
    t['Q2b']=np.where(t.ratio>=1.0,'충족','미충족'); t['S기준']=np.where(t.S<=0,'충족','미충족')
    t=t[['설정','readout','latent_L2','n_null','A','B','C','auc_C','d_main','d_latent',
         'd_total','ratio','S','R기준','Q2','Q2b','S기준']]
    t.to_csv(OUT+'table1_ladders_v6.csv',index=False,float_format='%.5f')
    print(t.to_string(index=False,float_format=lambda z:f'{z:+.5f}'))
    # 귀무 빈도 표
    sys.path.insert(0,CD); import data
    d=data.prep(None,ward=False); real=np.bincount(d['nur'],minlength=int(d['n_nurse']))
    nr=[dict(팔='실제',중앙=float(np.median(real)),평균=float(real.mean()),
             최대=int(real.max()),최소=int(real.min()),순위상관=np.nan)]
    for k in range(3):
        c=np.bincount(data.shuffle_nurse(d,k,'A'),minlength=int(d['n_nurse']))
        nr.append(dict(팔=f'귀무{k}',중앙=float(np.median(c)),평균=float(c.mean()),
                       최대=int(c.max()),최소=int(c.min()),
                       순위상관=float(np.corrcoef(np.argsort(np.argsort(real)),
                                                np.argsort(np.argsort(c)))[0,1])))
    pd.DataFrame(nr).to_csv(OUT+'table8_null_frequency.csv',index=False,float_format='%.4f')
    print('\n', pd.DataFrame(nr).to_string(index=False))
    for src,dst in (('diag/fn_decomp2.json','table9_function_decomp_v6.json'),):
        json.dump(json.load(open(CD+src)),open(OUT+dst,'w'),indent=1)
    print('\n저장:',OUT)

if __name__=='__main__': main()
