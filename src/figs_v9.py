"""v9 그림 — substantive result 를 앞에 둔다. 배포된 결과 파일만 읽는다.

    Fig 1  (a) matched-pair reversal 분포 관측 대 귀무   (b) 그 분포의 argmax 짝
    Fig 2  Cluster 1~4 — 해석 이름 «없이» 대표약물·관측률
    Fig 3  제공자 정보의 표본외 증분 — 사다리 + top-10% 집중도
    Fig 4  병동·맥락 강건성
"""
import os, sys, json, csv
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.dpi':150,'savefig.bbox':'tight'})
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
CD=os.path.dirname(os.path.abspath(__file__))+os.sep
R6=os.path.join(paths.RESULTS,'v6')+os.sep
R9=os.path.join(paths.RESULTS,'v9')+os.sep
OUT=os.path.join(paths.ROOT,'figures')+os.sep; os.makedirs(OUT,exist_ok=True)
C1,C2,C3,CG='#3d5a80','#ee6c4d','#7f9c6a','#9aa5b1'

# ── 군집 : 관측 비시행률 오름차순으로 Cluster 1~4. 해석 이름을 «쓰지 않는다» ──
CL=pd.read_csv(R9+'out_05_cluster_summary.csv').sort_values('누락률').reset_index(drop=True)
CL['label']=[f'Cluster {i+1}' for i in range(len(CL))]
# out_06 의 열은 이미 canonical(관측률 오름차순) cluster_1..4 다.
COLS=[f'cluster_{i+1}' for i in range(len(CL))]
ID2LAB=dict(zip(CL.군집, CL.label))

# ── Fig 1 ────────────────────────────────────────────────────────────────
z=np.load(R9+'reversal.npz'); obs,nul=z['obs'],z['null']
P=pd.read_csv(R9+'out_06_provider_profiles.csv')
cols=COLS                                        # 관측률 오름차순
A=P.iloc[0]; B=P.iloc[1]
span=float(max(A[c]-B[c] for c in cols)-min(A[c]-B[c] for c in cols))

fig,ax=plt.subplots(1,2,figsize=(9.6,3.4)); fig.subplots_adjust(wspace=.30)
bins=np.linspace(0,4.6,60)
ax[0].hist(nul,bins=bins,density=True,color=CG,alpha=.75,label='provider-shuffled null')
ax[0].hist(obs,bins=bins,density=True,histtype='step',color=C1,lw=1.8,label='observed')
for v,c,ls in ((np.median(nul),CG,'--'),(np.median(obs),C1,'--')):
    ax[0].axvline(v,color=c,ls=ls,lw=1.1)
ax[0].annotate(f'median {np.median(nul):.2f}',xy=(np.median(nul),1.02),xytext=(np.median(nul)-.05,1.62),
               fontsize=7.2,color='#6b7280',ha='center')
ax[0].annotate(f'median {np.median(obs):.2f}',xy=(np.median(obs),1.02),xytext=(np.median(obs)+.14,1.40),
               fontsize=7.2,color=C1,ha='left')
ax[0].plot([span],[0.03],marker='v',color=C2,ms=9,clip_on=False,zorder=5)
ax[0].text(span,0.13,'pair in (b)',ha='right',fontsize=7.2,color=C2)
ax[0].set_xlabel('reversal span between two providers\nmatched on overall non-administration rate')
ax[0].set_ylabel('density'); ax[0].set_xlim(0,4.6); ax[0].set_ylim(0,1.78)
ax[0].legend(frameon=False,fontsize=7.6,loc='upper right')
ax[0].set_title('(a) all eligible matched pairs',fontsize=9)

x=np.arange(len(cols)); w=.36
ax[1].bar(x-w/2,[A[c] for c in cols],w,color=C1,label=f'Provider A  ({100*A.overall:.1f}%)')
ax[1].bar(x+w/2,[B[c] for c in cols],w,color=C2,label=f'Provider B  ({100*B.overall:.1f}%)')
ax[1].axhline(1,color='k',ls=':',lw=.9)
ax[1].text(3.52,0.86,'cluster mean',fontsize=7,color='#555',ha='right')
for i,c in enumerate(cols):
    ax[1].text(i-w/2,A[c]+.06,f'{A[c]:.2f}',ha='center',fontsize=7.2)
    ax[1].text(i+w/2,B[c]+.06,f'{B[c]:.2f}',ha='center',fontsize=7.2)
ax[1].set_xticks(x); ax[1].set_xticklabels([c.replace('cluster_','C') for c in cols])
ax[1].set_ylabel('ratio to cluster mean\nnon-administration rate')
ax[1].set_ylim(0,3.95); ax[1].legend(frameon=False,fontsize=7.6,loc='upper right')
ax[1].set_title('(b) maximum-reversal pair among all eligible pairs in (a)',fontsize=9)
fig.text(.5,-.17,'Overall rates differ by 0.32 percentage points, within the 0.5-point matching window '
         'used to define eligible pairs in this exploratory analysis.\nThese identifiers represent '
         'medication-recording providers, not necessarily the individuals who performed or decided on '
         'medication administration.',ha='center',fontsize=7,color='#444')
fig.savefig(OUT+'fig1_heterogeneity_v9.png'); fig.savefig(OUT+'fig1_heterogeneity_v9.pdf'); plt.close(fig)
print(f'Fig 1  span {span:.2f} · obs n={len(obs):,} · null n={len(nul):,}')

# ── Fig 2 : 군집 — 해석 이름 없이 ─────────────────────────────────────────
fig,ax=plt.subplots(figsize=(8.6,3.2))
y=np.arange(len(CL))[::-1]
ax.barh(y,100*CL.누락률,color=[CG,CG,CG,C2],height=.6)
for i,(yy,r) in enumerate(zip(y,CL.itertuples())):
    ax.text(100*r.누락률+.25,yy,f'{100*r.누락률:.1f}%',va='center',fontsize=8,fontweight='bold')
    ax.text(-.4,yy,f'{r.label}',va='center',ha='right',fontsize=8.5,fontweight='bold')
    ax.text(-.4,yy-.30,f'{r.약물수} drugs · {r.건수:,} events',va='center',ha='right',
            fontsize=6.8,color='#666')
    ax.text(19.6,yy,' · '.join(r.대표약물.split(' · ')[:4]),va='center',fontsize=6.6,color='#333')
ax.set_yticks([]); ax.set_xlim(0,35); ax.set_xlabel('observed documented non-administration rate (%)'); ax.set_xticks(range(0,18,2))
ax.axvline(9.285,color='k',ls=':',lw=.9,ymax=.93); ax.text(9.45,3.42,'cohort 9.3%',fontsize=7,color='#555')
ax.set_title('Exploratory medication-pattern clusters (author-assigned numbering by observed rate)',fontsize=9)
fig.savefig(OUT+'fig2_clusters_v9.png',bbox_inches='tight')
fig.savefig(OUT+'fig2_clusters_v9.pdf',bbox_inches='tight'); plt.close(fig)
print('Fig 2  군집 4개')

# ── Fig 3 : 제공자 정보의 표본외 증분 ────────────────────────────────────
T=pd.read_csv(R6+'table1_ladders_v6.csv')
nat=T[T.설정=='순환 이중 잠재 (native)'].iloc[0]
K={r['지표']:r for r in csv.DictReader(open(R9+'table13_concentration.csv'))}
cap=[k for k in K if k.startswith('top-10% capture')][0]
obr=[k for k in K if 'observed' in k][0]
capA,capC,capN=(float(K[cap]['A (제공자 없음)']),float(K[cap]['C (제공자 잠재 포함)']),
                float(K[cap]['C 귀무 3팔 평균']))

fig,ax=plt.subplots(1,2,figsize=(9.2,3.2)); fig.subplots_adjust(wspace=.34)
# (a) 사다리
A_,B_,C_=nat.A,nat.B,nat.C
Bn,Cn=B_-nat.d_main, C_-nat.d_total    # 귀무 참조선
ax[0].plot([0,1,2],[A_,B_,C_],'o-',color=C1,lw=2,ms=7,label='observed provider assignment')
ax[0].plot([0,1,2],[A_,Bn,Cn],'o--',color=CG,lw=1.6,ms=6,label='provider-shuffled null')
ax[0].annotate('',xy=(2,C_),xytext=(2,Cn),arrowprops=dict(arrowstyle='<->',color=C2,lw=1.4))
ax[0].text(2.06,(C_+Cn)/2,f'+{nat.d_total:.4f}\nBSS',fontsize=7.6,color=C2,va='center')
ax[0].set_xticks([0,1,2]); ax[0].set_xticklabels(['A\nno provider','B\n+ identity','C\n+ provider latent'])
ax[0].set_ylabel('Brier skill score (out-of-sample)')
ax[0].legend(frameon=False,fontsize=7.4,loc='upper left'); ax[0].set_xlim(-.25,2.55)
ax[0].set_title('(a) provider information adds out-of-sample signal',fontsize=9)
# (b) 집중도
xb=np.arange(3); v=[capA,capN,capC]; cs=[C1,CG,C2]
ax[1].bar(xb,v,.55,color=cs)
for i,q in enumerate(v): ax[1].text(i,q+.6,f'{q:.1f}%',ha='center',fontsize=8.5,fontweight='bold')
ax[1].axhline(10,color='k',ls=':',lw=.9); ax[1].text(2.42,11.0,'chance 10%',ha='right',fontsize=7)
ax[1].set_xticks(xb); ax[1].set_xticklabels(['A\nno provider','C\nshuffled null','C\nobserved'])
ax[1].set_ylabel('% of all documented non-administration\nevents in the top-ranked 10%')
ax[1].set_ylim(0,49)
ax[1].set_title('(b) concentration in the top-ranked decile',fontsize=9)
fig.text(.5,-.13,'Each model ranks the held-out events within its own fold; the top 10% of events is taken per fold. '
         'Sets selected by A and by C are not identical.\nThis is not a prospective triage evaluation: some covariates '
         'are aggregated over the whole admission.',ha='center',fontsize=7,color='#444')
fig.savefig(OUT+'fig3_incremental_v9.png'); fig.savefig(OUT+'fig3_incremental_v9.pdf'); plt.close(fig)
print(f'Fig 3  R {nat.d_total:+.5f} · capture {capA:.1f} / {capN:.1f} / {capC:.1f}')

# ── Fig 4 : 병동·맥락 강건성 ─────────────────────────────────────────────
rows=[('Cyclic dual latent','순환 이중 잠재 (native)','native — 병동 통제'),
      ('CDLD-A (aligned)','CDLD-A (본판)','CDLD-A — 병동 통제'),
      ('Low-rank bilinear','저랭크 이중선형 (ALS)','ALS — 병동 통제')]
fig,ax=plt.subplots(figsize=(6.6,3.0))
xb=np.arange(3); w=.34
base=[T[T.설정==b].iloc[0].d_total for _,b,_ in rows]
ward=[T[T.설정==w_].iloc[0].d_total for _,_,w_ in rows]
ax.bar(xb-w/2,base,w,color=C1,label='base')
ax.bar(xb+w/2,ward,w,color=C3,label='ward and ward x drug controlled')
for i,(b,wd) in enumerate(zip(base,ward)):
    ax.text(i-w/2,b+.0004,f'{b:+.4f}',ha='center',fontsize=7.2)
    ax.text(i+w/2,wd+.0004,f'{wd:+.4f}',ha='center',fontsize=7.2)
    ax.text(i+w/2,wd/2,f'{100*wd/b:.0f}%',ha='center',fontsize=8,color='white',fontweight='bold')
ax.axhline(.010,color='k',ls=':',lw=.9); ax.text(2.45,.0104,'prespecified R threshold',ha='right',fontsize=7)
ax.set_xticks(xb); ax.set_xticklabels([r[0] for r in rows],fontsize=8)
ax.set_ylabel('null-adjusted total increment (BSS)'); ax.set_ylim(0,.0215)
ax.legend(frameon=False,fontsize=7.6,loc='upper left')
ax.set_title('Provider signal after ward and ward x drug control',fontsize=9)
fig.savefig(OUT+'fig4_ward_v9.png'); fig.savefig(OUT+'fig4_ward_v9.pdf'); plt.close(fig)
print('Fig 4  잔존', ' · '.join(f'{100*w_/b:.0f}%' for b,w_ in zip(base,ward)))
