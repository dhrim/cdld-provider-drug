import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
"""심사자용 재현 노트북 생성 — MIMIC 접근 없이 논문의 모든 표·그림을 재계산."""
import nbformat as nbf, io, os
nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def co(s): C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Reproducing every number and figure in the manuscript

**No MIMIC-IV access is required.** This notebook reads only the per-fold
out-of-sample results committed in `results/` and recomputes the paper's
tables, figures and prespecified adjudication. It runs in a few seconds.

To re-run the experiments themselves (~12 CPU-hours, credentialed MIMIC-IV
access required) see `docs/REPRODUCE.md`.

---

### What the reader should check

| | |
|---|---|
| §1 | the cohort as analysed |
| §2 | **the null design** — and why the first version of it was wrong |
| §3 | the A→B→C ladder under two estimator families |
| §4 | **the prespecified criteria — and the one that fails** |
| §5 | robustness: ward control and sensitivity settings |
| §6 | latent direction clusters |
| §7 | assertions: every number quoted in the manuscript |
""")

co("""
import json, glob, os
import numpy as np, pandas as pd
import matplotlib; import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'font.size': 9, 'figure.dpi': 110, 'axes.spines.top': False,
                 'axes.spines.right': False, 'font.family': 'DejaVu Sans'})
R = 'results/'
CD, CA, GREY = '#1f4e79', '#b45f06', '#8c8c8c'
print('pandas', pd.__version__, '| numpy', np.__version__)
""")

md("""
## 1. The cohort

Row counts and provider statistics as analysed. These are descriptive facts
about the extraction, reported in Table 1 of the manuscript.
""")
co("""
pd.read_csv(R + 'table1_cohort.csv')
""")

md("""
## 2. The null design

Every reported quantity is a **net increment**: the observed gain minus the gain
obtained when provider identity is destroyed. The null therefore decides the
answer, and its construction is reported here in full.

**The null must destroy provider identity while preserving the design** —
rows per admission, distinct providers per admission, the within-admission
assignment pattern, and the provider frequency distribution.

### 2.1 The first implementation was wrong, and we report it

Our initial implementation used

```python
nh = np.zeros(hadm.max() + 1, dtype=np.int64)
nh[hadm] = nur                 # fancy-index assignment: only the LAST write survives
shuffled = rng.permutation(nh)[hadm]
```

Numpy fancy-index assignment keeps only the last write per index, so every
admission collapsed to a **single** provider. The table below contrasts the
observed data, that broken null, and the corrected null.
""")
co("""
pd.read_csv(R + 'table_null_design.csv')
""")
md("""
The corrected null (`design='A'`) redraws each admission's *k* distinct providers
from the global pool with frequency weights and keeps the within-admission
assignment pattern. It reproduces the observed structure exactly
(3.02 providers per admission, median 2, max 635) while retaining only
0.26% of the original assignments.

**This mattered, and asymmetrically.** Correcting the null moved the
cyclic-dual-latent estimate by 20% and the low-rank estimate by 2%
(§5 below). A null that is adequate for one estimator can be inadequate
for another.
""")

md("""
## 3. The ladder, under two estimator families

`A` baseline without the provider · `B` + provider identity ·
`C` + provider latent. Net increments subtract the shuffled-provider null.
""")
co("""
fold = pd.read_csv(R + 'fold_level.csv')
def r2(x, arm, kind):
    s = x[(x.arm == arm) & (x.kind == kind)].sort_values('fold')
    return s.r2.to_numpy()

def ladder(tag, n_null=3):
    x = fold[fold.tag == tag]
    A, B, C = r2(x,'real','A'), r2(x,'real','B'), r2(x,'real','C')
    SB = np.mean([r2(x,f'shuffle{k}','B') for k in range(n_null)], axis=0)
    SC = np.mean([r2(x,f'shuffle{k}','C') for k in range(n_null)], axis=0)
    return dict(A=A, B=B, C=C, SB=SB, SC=SC,
                main=(B-A)-(SB-A), latent=(C-B)-(SC-SB), total=(C-A)-(SC-A))

base = ladder('base')
for k in ('A','B','C','SB','SC'):
    print(f'  {k:3s} {base[k].mean():+.4f}   folds {np.round(base[k],4)}')
""")
co("""
tbl = pd.read_csv(R + 'table2_two_estimators.csv')
tbl
""")
md("""
> **The two estimator families agree that including the provider helps, and
> disagree about where the help comes from.**
""")

md("""
## 4. The prespecified criteria

Thresholds were fixed before the reported confirmatory runs (after exploratory
analysis on the same database — see `protocol/`). `Q2b` is the conclusion
criterion.
""")
co("""
adj = pd.read_csv(R + 'table2b_adjudication.csv')
adj
""")
co("""
b = ladder('base')
ratio_folds = b['latent'] / b['main']
print('per-fold latent/main ratio (cyclic dual latent):', np.round(ratio_folds, 2))
print('every fold below the threshold of 1.0:', bool((ratio_folds < 1.0).all()))
""")
md("""
**Q2b is not met under the cyclic dual latent estimator and is met under the
low-rank bilinear estimator, on the same data, folds and null.**
The same prespecified criterion returns opposite verdicts.
""")

md("""
## 5. Robustness — ward control and sensitivity settings
""")
co("""
sens = pd.read_csv(R + 'table3_settings.csv')
sens
""")
co("""
tot = sens['total'].to_numpy(); rat = sens['ratio'].to_numpy()
print(f'total  {tot.min():+.4f} .. {tot.max():+.4f}   ({tot.max()/tot.min():.2f}x)'
      f'   -- every setting clears the +0.010 threshold: {bool((tot>=0.010).all())}')
print(f'ratio  {rat.min():.2f} .. {rat.max():.2f}   ({rat.max()/rat.min():.2f}x)'
      f'   -- crosses the 1.0 threshold: {bool((rat<1).any() and (rat>=1).any())}')
""")

md("## 6. Figures")
co("""
FIG = 'figures/'
from IPython.display import Image, display
for f in sorted(os.listdir(FIG)):
    if f.endswith('.png'):
        print(f); display(Image(FIG + f))
""")

md("""
## 7. Assertions — every number quoted in the manuscript
""")
co("""
import subprocess, sys
print(subprocess.run([sys.executable, 'verify.py'], capture_output=True, text=True).stdout)
""")
nb['cells'] = C
nb.metadata['kernelspec'] = dict(name='python3', display_name='Python 3')
nbf.write(nb, os.path.join(paths.WORK, 'repo_v2_reproduce.ipynb'))
print('노트북 뼈대 생성')
