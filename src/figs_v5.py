import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
"""v5 그림 — 영문 라벨 (컨테이너 폰트에 한글 글리프 없음)."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 150, 'savefig.bbox': 'tight'})
CD = paths.WORK + os.sep; OUT = paths.RESULTS + '/v6/'
A2 = json.load(open(CD + 'diag/adj_a2.json'))
FN = json.load(open(CD + 'diag/fn_decomp.json'))
G = json.load(open(CD + 'diag/gate_C_a2.json'))
C1, C2, C3 = '#3d5a80', '#ee6c4d', '#7f9c6a'

# ---------- Fig 1 : skill decomposition vs function decomposition ----------
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.2))
mods = ['native CDLD\n(unconstrained\nreadout)', 'CDLD-A\n(aligned\nreadout)', 'ALS low-rank\n(aligned\nreadout)']
skill = [A2['native2']['ratio'], A2['aligned']['ratio'], 1.5089]
cols = [C1, C2, C3]
b = ax[0].bar(range(3), skill, color=cols, width=.6)
ax[0].axhline(1.0, color='k', ls='--', lw=.9)
ax[0].text(2.42, 1.05, 'Q2b threshold', ha='right', fontsize=7.5)
for i, v in enumerate(skill):
    ax[0].text(i, v + .07, f'{v:.2f}', ha='center', fontweight='bold')
ax[0].set_ylim(0, 2.75); ax[0].set_xticks(range(3))
ax[0].set_xticklabels(mods, fontsize=7.2)
ax[0].set_ylabel('latent / main increment')
ax[0].set_title('(a) decomposition of predictive skill\n(the prespecified Q2b)', fontsize=9)

fn = [FN['native_C']['inter_over_provider'], FN['cdldA2_C']['inter_over_provider']]
ax[1].bar(range(2), fn, color=[C1, C2], width=.45)
for i, v in enumerate(fn):
    ax[1].text(i, v + .02, f'{v:.2f}', ha='center', fontweight='bold')
ax[1].set_ylim(0, 2.75); ax[1].set_xticks(range(2))
ax[1].set_xticklabels(mods[:2], fontsize=7.2)
ax[1].set_ylabel('Var(interaction) / Var(provider)')
ax[1].set_title('(b) decomposition of the fitted function\n(provider x drug grid, fixed context)', fontsize=9)
ax[1].plot([0, 0, 1, 1], [.86, 1.15, 1.15, .65], color='k', lw=.8)
ax[1].text(.5, 1.22, 'same order of magnitude —\nand the ordering reverses',
           ha='center', va='bottom', fontsize=7.5)
ax[1].set_xlim(-.6, 1.6)
fig.savefig(OUT + 'fig1_skill_vs_function.png'); fig.savefig(OUT + 'fig1_skill_vs_function.pdf')
plt.close(fig)

# ---------- Fig 2 : ladders ----------
fig, ax = plt.subplots(figsize=(5.6, 3.2))
L = [('native CDLD', A2['native2'], C1), ('CDLD-A', A2['aligned'], C2)]
als = dict(A=.12861, B=.13526, C=.14577)
for i, (nm, d, c) in enumerate(L + [('ALS low-rank', als, C3)]):
    ax.plot([0, 1, 2], [d['A'], d['B'], d['C']], 'o-', color=c, label=nm, lw=1.8, ms=6)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['A\nno provider', 'B\n+ provider identity', 'C\n+ provider latent'])
ax.set_ylabel('Brier skill score (out-of-sample)')
ax.legend(frameon=False, fontsize=8, loc='upper left')
ax.annotate('', xy=(1, .14029), xytext=(1, .14431),
            arrowprops=dict(arrowstyle='<->', color='#444', lw=1.1))
ax.text(1.05, .1423, 'gamma* = +0.0051\nthe non-additive provider-dependent\ncontribution absorbed at stage B',
        fontsize=7, va='center')
ax.text(-.06, .1348, 'A is shared\nby construction', fontsize=7, ha='left')
ax.set_title('the same ladder, three readouts:\nthe two deep models differ only at rung B', fontsize=9)
fig.savefig(OUT + 'fig2_ladder.png'); fig.savefig(OUT + 'fig2_ladder.pdf'); plt.close(fig)

# ---------- Fig 3 : attribution ----------
fig, ax = plt.subplots(figsize=(5.6, 2.9))
r0, rA = A2['r_nat'], A2['r_A']
parts = [('native\nCDLD', r0, '#bbbbbb'), ('+ readout\nconstraint', A2['share_readout'], C2),
         ('+ signal\nshift', A2['share_kappa'], '#9aa5b1'),
         ('+ residual', A2['resid'], '#d9d9d9')]
run = 0
for i, (nm, v, c) in enumerate(parts):
    if i == 0:
        ax.bar(i, v, color=c, width=.6); run = v
        ax.text(i, v + .06, f'{v:.2f}', ha='center', fontsize=8)
    else:
        ax.bar(i, v, bottom=run, color=c, width=.6)
        ax.text(i, run + v + (.06 if v > 0 else -.16), f'{v:+.2f}', ha='center', fontsize=8)
        run += v
ax.bar(4, rA, color=C2, width=.6); ax.text(4, rA + .06, f'{rA:.2f}', ha='center', fontweight='bold')
ax.axhline(1.0, color='k', ls='--', lw=.9)
ax.set_xticks(range(5)); ax.set_xticklabels([p[0] for p in parts] + ['CDLD-A'], fontsize=7.6)
ax.text(1, 2.62, 'gamma* = +0.0051', ha='center', fontsize=7)
ax.text(2, 2.62, 'kappa = -0.0002', ha='center', fontsize=7)
ax.set_ylabel('latent / main increment'); ax.set_ylim(0, 2.75)
ax.set_title('the readout constraint accounts for 102% of the flip', fontsize=9)
fig.savefig(OUT + 'fig3_attribution.png'); fig.savefig(OUT + 'fig3_attribution.pdf'); plt.close(fig)

# ---------- Fig 4 : latent collapse ----------
import torch
def norms(d, f=0):
    sd = torch.load(CD + d + f'/w_base_real_f{f}_C.pt', map_location='cpu', weights_only=False)
    U = sd['u_tab.weight'].numpy(); V = sd['v2_tab.weight'].numpy()
    a = sd['alpha.weight'].numpy().ravel()
    rng = np.random.default_rng(0); ii = rng.integers(0, len(U), 20000); jj = rng.integers(0, len(V), 20000)
    return np.linalg.norm(U, axis=1).mean(), np.linalg.norm(V, axis=1).mean(), \
        (U[ii] * V[jj]).sum(1).std(), a.std()
n1, n2 = norms('ck_a'), norms('ck_a2')
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
fig.subplots_adjust(wspace=.42)
x = np.arange(2); w = .35
ax[0].bar(x - w/2, [n1[0], n1[1]], w, label='run 1 (L2 on latent tables)', color='#cccccc')
ax[0].bar(x + w/2, [n2[0], n2[1]], w, label='run 2 (L2 excluded)', color=C2)
ax[0].axhline(0.8, color='k', ls=':', lw=.9)
ax[0].text(-.45, .82, 'at initialisation', ha='left', fontsize=7.2)
ax[0].set_xticks(x); ax[0].set_xticklabels(['||u|| provider', '||v|| drug'])
ax[0].set_ylabel('mean embedding norm'); ax[0].legend(frameon=False, fontsize=7.2)
ax[0].set_title('(a) latent tables collapse under L2', fontsize=9)
ax[1].bar(x - w/2, [n1[2], n1[3]], w, color='#cccccc')
ax[1].bar(x + w/2, [n2[2], n2[3]], w, color=C2)
ax[1].set_xticks(x); ax[1].set_xticklabels(['sd(u.v)\ninteraction', 'sd(alpha)\nmain effect'])
ax[1].set_ylabel('logit-scale contribution')
ax[1].set_title('(b) the interaction term vanishes', fontsize=9)
fig.savefig(OUT + 'fig4_collapse.png'); fig.savefig(OUT + 'fig4_collapse.pdf'); plt.close(fig)
print('figs written')
for k in ('fig1_skill_vs_function', 'fig2_ladder', 'fig3_attribution', 'fig4_collapse'):
    print(' ', k)
