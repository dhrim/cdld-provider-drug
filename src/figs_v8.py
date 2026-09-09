"""v8 그림 — 배포된 결과 파일만 읽는다. MIMIC 자료도 모형 가중치도 필요 없다.

    입력   results/v6/table1_ladders_v6.csv · table9_function_decomp_v6.json
           results/v7/table11_two_by_two_v7.csv · table7_latent_norms_v7.json
    출력   figures/fig{1..4}_*_v8.{png,pdf}
"""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.dpi':150,'savefig.bbox':'tight'})
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
R6  = os.path.join(paths.RESULTS, 'v6') + os.sep
R7  = os.path.join(paths.RESULTS, 'v7') + os.sep
OUT = os.path.join(paths.ROOT, 'figures') + os.sep
os.makedirs(OUT, exist_ok=True)
T=pd.read_csv(R6+'table1_ladders_v6.csv')
F=json.load(open(R6+'table9_function_decomp_v6.json'))
C1,C2,C3,CG='#3d5a80','#ee6c4d','#7f9c6a','#9aa5b1'
row=lambda s: T[T.설정==s].iloc[0]
nat=row('순환 이중 잠재 (native)'); ali=row('CDLD-A (본판)'); als=row('저랭크 이중선형 (ALS)')
natnd=row('native — 잠재표 감쇠 제외'); mid=row('CDLD-A — 잠재표 L2 1e-5'); a1=row('CDLD-A 1차 (붕괴)')

# ---- Fig 1 : skill vs function decomposition (사전 지표를 primary 로) ----
fig,ax=plt.subplots(1,3,figsize=(9.8,3.1))
fig.subplots_adjust(wspace=.34)
v=[nat.ratio,ali.ratio,als.ratio]
ax[0].bar(range(3),v,color=[C1,C2,C3],width=.6)
ax[0].axhline(1,color='k',ls='--',lw=.9); ax[0].text(-0.45,1.06,'Q2b threshold',ha='left',fontsize=7)
for i,x in enumerate(v): ax[0].text(i,x+.07,f'{x:.2f}',ha='center',fontweight='bold')
ax[0].set_ylim(0,2.9); ax[0].set_xticks(range(3))
ax[0].set_xticklabels(['native\nCDLD','CDLD-A','ALS'],fontsize=8)
ax[0].set_ylabel('latent / main increment')
ax[0].set_title('(a) decomposition of\npredictive skill (Q2b)',fontsize=9)

def box(a,keys,labs,cols,ylab,title,thresh=None):
    data=[]
    for k in keys:
        s=F[k]; data.append([s['min'],s['q25'],s['median'],s['q75'],s['max']])
    for i,(d,c) in enumerate(zip(data,cols)):
        a.plot([i,i],[d[0],d[4]],color=c,lw=1)
        a.add_patch(plt.Rectangle((i-.22,d[1]),.44,d[3]-d[1],facecolor=c,alpha=.35,edgecolor=c))
        a.plot([i-.22,i+.22],[d[2],d[2]],color=c,lw=2)
        a.text(i,d[4]+ (d[4]-d[0])*.10,f'{F[keys[i]]["mean"]:.3f}',ha='center',fontsize=8,fontweight='bold')
    a.set_xticks(range(len(labs))); a.set_xticklabels(labs,fontsize=8)
    a.set_ylabel(ylab); a.set_title(title,fontsize=9)
    if thresh is not None: a.axhline(thresh,color='k',ls='--',lw=.9)

box(ax[1],['native_frac_inter','aligned_frac_inter'],['native\nCDLD','CDLD-A'],[C1,C2],
    'Var(interaction) / total','(b) decomposition of the fitted\nfunction — prespecified')
ax[1].set_ylim(0,.075)
box(ax[2],['corr_c','corr_eta_raw'],['interaction\ncomponent $c_{ij}$','raw $\\eta$ grid\n(not used)'],
    [C2,CG],'weighted correlation','(c) do the two models find the\nsame interaction structure?')
ax[2].set_ylim(0,1.05)
fig.text(.5,-.06,'(b)(c) 40 providers x 20 drugs, high-support subset of 936 x 522; '
         '20 fixed contexts x 5 folds. Box = IQR, whisker = range, bar = median.',
         ha='center',fontsize=7,color='#555')
fig.savefig(OUT+'fig1_skill_vs_function_v8.png'); fig.savefig(OUT+'fig1_skill_vs_function_v8.pdf'); plt.close(fig)

# ---- Fig 2 : 귀무 보정 «누적 증분» 축 ----
fig,ax=plt.subplots(figsize=(5.8,3.3))
for nm,r,c in (('native CDLD',nat,C1),('CDLD-A',ali,C2),('ALS low-rank',als,C3)):
    ax.plot([0,1,2],[0,r.d_main,r.d_total],'o-',color=c,label=nm,lw=1.8,ms=6)
g=nat.d_main-ali.d_main
ax.annotate('',xy=(1,ali.d_main),xytext=(1,nat.d_main),
            arrowprops=dict(arrowstyle='<->',color='#444',lw=1.1))
ax.text(1.06,(nat.d_main+ali.d_main)/2,
        f'$\\gamma^*$ = {g:+.5f}\nnull-adjusted B-stage gap',fontsize=7,va='center')
ax.axhline(0,color='#bbb',lw=.8)
ax.set_xticks([0,1,2]); ax.set_xticklabels(['A\nno provider','B\n+ provider identity','C\n+ provider latent'])
ax.set_ylabel('null-adjusted cumulative increment (BSS)')
ax.legend(frameon=False,fontsize=8,loc='upper left')
ax.set_title('null-adjusted ladder — the arrow is on the same scale as the axis',fontsize=9)
fig.savefig(OUT+'fig2_ladder_v8.png'); fig.savefig(OUT+'fig2_ladder_v8.pdf'); plt.close(fig)

# ---- Fig 3 : 2x2 (귀무 2회로 «칸을 맞춘» 표만 쓴다 — table11) ----
X=pd.read_csv(R7+'table11_two_by_two_v7.csv',dtype={'latent_L2':str})
g=lambda r,l: X[(X.readout==r)&(X.latent_L2==l)].iloc[0]
u4,u0 = g('비제약','1e-4'), g('비제약','0')
a4,a5,a0 = g('정합','1e-4'), g('정합','1e-5'), g('정합','0')
fig,ax=plt.subplots(figsize=(6.2,3.4))
ax.plot([0,2],[u4.ratio,u0.ratio],'o-',color=C1,lw=1.8,ms=7,label='unconstrained readout')
ax.plot([1,2],[a5.ratio,a0.ratio],'o-',color=C2,lw=1.8,ms=7,label='aligned readout')
ax.plot([0,1],[a4.ratio,a5.ratio],ls=':',color=C2,lw=1.2)
ax.plot([0],[a4.ratio],'o',mfc='white',mec=C2,mew=1.8,ms=9)
for x,y in ((0,u4.ratio),(2,u0.ratio),(1,a5.ratio),(2,a0.ratio)):
    ax.text(x,y+.10,f'{y:.2f}',ha='center',fontsize=8,fontweight='bold',color='k')
ax.text(-0.16,a4.ratio,f'{a4.ratio:.2f}',ha='right',va='center',fontsize=8,color='#c0392b')
ax.annotate('latent tables collapsed;\nfit gate failed',xy=(0.06,a4.ratio+.04),
            xytext=(0.34,1.40),fontsize=7.2,color='#c0392b',va='center',
            arrowprops=dict(arrowstyle='->',color='#c0392b',lw=.9,
                            connectionstyle='arc3,rad=-0.2'))
ax.axhline(1,color='k',ls='--',lw=.9); ax.text(1.95,1.07,'Q2b threshold',ha='right',fontsize=7.2)
ax.set_xticks([0,1,2]); ax.set_xticklabels(['1e-4','1e-5','0'])
ax.set_xlim(-.40,2.25)
ax.set_xlabel('weight decay on the latent tables')
ax.set_ylabel('latent / main increment')
ax.set_ylim(0,2.9); ax.legend(frameon=False,fontsize=8,loc='upper left')
ax.set_title('The ratio classification remains separated after varying latent-table regularisation',fontsize=9)
ax.text(2.24,-0.62,'all cells: 2 null repetitions (matched)',ha='right',fontsize=6.8,color='#666')
fig.savefig(OUT+'fig3_two_by_two_v8.png'); fig.savefig(OUT+'fig3_two_by_two_v8.pdf'); plt.close(fig)

# ---- Fig 4 : 붕괴 — ★배포된 표에서 읽는다 (모형 가중치 재배포하지 않으므로) ----
LN=json.load(open(R7+'table7_latent_norms_v7.json'))
g=lambda k:(LN[k]['u_norm'],LN[k]['v_norm'],LN[k]['sd_uv_logit'],LN[k]['sd_alpha_logit'])
n1,n2,n3=g('aligned_L2_1e-4_run1'),g('aligned_L2_1e-5'),g('aligned_L2_0_run2')
fig,ax=plt.subplots(1,2,figsize=(7.2,2.9)); fig.subplots_adjust(wspace=.42)
x=np.arange(2); w=.26
for i,(n,lab,c) in enumerate(((n1,'L2 = 1e-4','#cccccc'),(n2,'L2 = 1e-5','#f0b8a8'),(n3,'L2 = 0',C2))):
    ax[0].bar(x+(i-1)*w,[n[0],n[1]],w,label=lab,color=c)
    ax[1].bar(x+(i-1)*w,[n[2],n[3]],w,color=c)
ax[0].axhline(.8,color='k',ls=':',lw=.9); ax[0].text(-.45,.83,'at initialisation',fontsize=7.2)
ax[0].set_xticks(x); ax[0].set_xticklabels(['||u|| provider','||v|| drug'])
ax[0].set_ylabel('mean embedding norm'); ax[0].legend(frameon=False,fontsize=7.2)
ax[0].set_title('(a) latent tables collapse under L2',fontsize=9)
ax[1].set_xticks(x); ax[1].set_xticklabels(['sd(u.v)\ninteraction','sd(alpha)\nmain effect'])
ax[1].set_ylabel('logit-scale contribution'); ax[1].set_title('(b) the interaction term vanishes',fontsize=9)
fig.savefig(OUT+'fig4_collapse_v8.png'); fig.savefig(OUT+'fig4_collapse_v8.pdf'); plt.close(fig)
print(f'v8 그림 4점 작성 -> {OUT}')
