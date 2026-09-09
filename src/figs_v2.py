import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
"""교정 귀무 기준 그림 4장. 축: 「같은 기준이 추정기에 따라 뒤집힌다」."""
import json, glob, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'font.size':9,'axes.linewidth':.8,'axes.spines.top':False,
                 'axes.spines.right':False,'figure.dpi':200,
                 'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42})
OUT=os.path.join(paths.WORK, 'repo_v2/figures/'); os.makedirs(OUT,exist_ok=True)
CD, CA, GREY = '#1f4e79','#b45f06','#8c8c8c'
fold=pd.read_csv(os.path.join(paths.WORK, 'repo_v2/results/fold_level.csv'))
T=pd.read_csv(os.path.join(paths.WORK, 'repo_v2/results/table3_settings.csv'))
def r2(x,a,k):
    s=x[(x.arm==a)&(x.kind==k)].sort_values('fold'); return s.r2.to_numpy()
def lad(tag,n=3):
    x=fold[fold.tag==tag]
    A,B,C=r2(x,'real','A'),r2(x,'real','B'),r2(x,'real','C')
    SB=np.mean([r2(x,f'shuffle{k}','B') for k in range(n)],0)
    SC=np.mean([r2(x,f'shuffle{k}','C') for k in range(n)],0)
    return A,B,C,SB,SC
A,B,C,SB,SC=lad('base')
als=pd.read_csv(os.path.join(paths.WORK, '결과/07_사전등록_none.csv')).set_index('arm')[['A','B','C']].astype(float)
aS=als.loc[[i for i in als.index if i.startswith('shuffle')]].mean(); ar=als.loc['real']

# ── 그림 1 : 사다리 — 같은 총, 다른 나눔, 그리고 판정이 갈린다 ─────────
fig,(ax,bx)=plt.subplots(1,2,figsize=(7.8,3.6),gridspec_kw=dict(width_ratios=[1.45,1],wspace=.34))
x=np.array([0,1,2])
cd=np.array([0,(B-A).mean(),(C-A).mean()]); cds=np.array([0,(B-A).std(ddof=1),(C-A).std(ddof=1)])
cs=np.array([0,(SB-A).mean(),(SC-A).mean()])
ad=np.array([0,ar.B-ar.A,ar.C-ar.A]); as_=np.array([0,aS.B-ar.A,aS.C-ar.A])
ax.fill_between(x,cd-cds,cd+cds,color=CD,alpha=.10,lw=0)
ax.plot(x,cd,'-o',color=CD,ms=5,lw=1.8,label='Cyclic dual latent discovery')
ax.plot(x,cs,'--o',color=CD,ms=4,lw=1.3,mfc='white')
ax.plot(x,ad,'-s',color=CA,ms=5,lw=1.8,label='Low-rank bilinear')
ax.plot(x,as_,'--s',color=CA,ms=4,lw=1.3,mfc='white')
ax.axhline(0,color='k',lw=.6)
for xs,o,n,c,v in [(2.14,cd[2],cs[2],CD,.0154),(2.42,ad[2],as_[2],CA,.0189)]:
    ax.annotate('',xy=(xs,o),xytext=(xs,n),arrowprops=dict(arrowstyle='<->',color=c,lw=1.2))
    ax.plot([2,xs],[o,o],color=c,lw=.6,ls=':'); ax.plot([2,xs],[n,n],color=c,lw=.6,ls=':')
    ax.text(xs+.06,(o+n)/2,f'+{v:.4f}',fontsize=8.5,va='center',color=c,fontweight='bold')
ax.text(1.0,-.0088,'filled = observed     open + dashed = provider-shuffled null',
        fontsize=7,ha='center',color=GREY)
ax.set_xticks(x); ax.set_xticklabels(['A\nno provider','B\n+ provider\nidentity','C\n+ provider\nlatent'])
ax.set_ylabel('out-of-sample $\\Delta R^2$ vs own baseline A')
ax.set_xlim(-.25,3.05); ax.set_ylim(-.0100,.0215)
ax.legend(frameon=False,fontsize=7.6,loc='upper left')
ax.set_title('a   The ladder under two estimator families',fontsize=10,loc='left')

lab=['latent net gain\n(C $-$ B)','main-effect net gain\n(B $-$ A)']
w=.34; y=np.array([1,0])
bx.barh(y+w/2,[ (C-B).mean()-(SC-SB).mean(), (B-A).mean()-(SB-A).mean() ],w,color=CD,label='Cyclic dual latent')
bx.barh(y-w/2,[ (ar.C-ar.B)-(aS.C-aS.B), (ar.B-ar.A)-(aS.B-ar.A) ],w,color=CA,label='Low-rank bilinear')
for yy,v,c in [(1+w/2,.0057,CD),(0+w/2,.0096,CD),(1-w/2,.0113,CA),(0-w/2,.0075,CA)]:
    bx.text(v+.0003,yy,f'{v:+.4f}',va='center',fontsize=7.8,color=c,fontweight='bold')
bx.set_yticks(y); bx.set_yticklabels(lab,fontsize=8)
bx.set_xlim(0,.0180); bx.set_ylim(-.55,1.72); bx.set_xlabel('net $\\Delta R^2$',fontsize=8)
bx.legend(frameon=False,fontsize=7.6,loc='upper right',bbox_to_anchor=(1.02,1.02))
bx.set_title('b   Where the gain comes from — reversed',fontsize=10,loc='left')
fig.savefig(OUT+'fig1_ladder_two_estimators.pdf',bbox_inches='tight')
fig.savefig(OUT+'fig1_ladder_two_estimators.png',bbox_inches='tight'); plt.close(fig)

# ── 그림 2 : 판정이 갈린다 ─────────────────────────────────────────
fig,(ax,bx)=plt.subplots(2,1,figsize=(7.2,4.6),sharex=True,gridspec_kw=dict(height_ratios=[1,1.35],hspace=.15))
lb=[s.replace('Cyclic dual latent — ','CDLD\n').replace('Low-rank bilinear — ','Low-rank\n')
    for s in T.setting]
col=[CA if 'Low-rank' in s else CD for s in T.setting]
ax.bar(range(len(T)),T.total,.5,color=col,alpha=.85)
for i,v in enumerate(T.total): ax.text(i,v+.0006,f'{v:.4f}',ha='center',fontsize=7.6)
ax.axhline(.010,color='k',ls=':',lw=1); ax.set_ylim(0,.0235)
ax.text(.02,.0106,'threshold R = +0.010',fontsize=7,ha='left',color=GREY)
ax.set_ylabel('entity net value\n(C $-$ A)',fontsize=8.5)
ax.set_title('Every setting clears R.  Q2b does not survive the change of estimator.',
             fontsize=10.5,loc='left')
bx.bar(range(len(T)),T.ratio,.5,color=col,alpha=.85)
for i,v in enumerate(T.ratio):
    bx.text(i,v+.035,f'{v:.2f}',ha='center',fontsize=8.6,fontweight='bold',
            color=(CA if v>=1 else CD))
bx.axhline(1.0,color='k',ls=':',lw=1.2)
bx.text(len(T)-.45,1.04,'prespecified conclusion threshold $Q_{2b}$ = 1.0',fontsize=7.5,ha='right')
bx.set_ylim(0,1.78); bx.set_xticks(range(len(T))); bx.set_xticklabels(lb,fontsize=7.4)
bx.set_ylabel('latent / main-effect net gain',fontsize=8.5)
fig.savefig(OUT+'fig2_criterion_flips.pdf',bbox_inches='tight')
fig.savefig(OUT+'fig2_criterion_flips.png',bbox_inches='tight'); plt.close(fig)
print('fig1, fig2 저장')
