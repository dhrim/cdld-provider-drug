"""experiments.ipynb 생성.

무거운 셀(자료 빌드·학습·ALS 적합)은 «실제 실행의 기록된 출력»을 그대로 싣는다.
가벼운 셀(집계·판정·작도)은 노트북 작성 시점에 실제로 실행한다.
어느 쪽인지는 노트북 첫 절에 명시한다.
"""
import nbformat as nbf, io, os, subprocess, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)
L = lambda f: io.open(f'logs/{f}', encoding='utf-8').read().rstrip()
SRC = lambda f: io.open(f'src/{f}', encoding='utf-8').read()

def cut(src, start, end=None, dedent=False):
    i = src.index(start)
    j = src.index(end, i) if end else len(src)
    return src[i:j].rstrip()

def stream(text):
    return nbf.v4.new_output('stream', name='stdout', text=text + '\n')

HEAVY, LIGHT = [], []
def heavy(md_, code, out, n):
    HEAVY.append(('md', md_)); HEAVY.append(('code', code, out, n))
def light(md_, code):
    LIGHT.append(('md', md_)); LIGHT.append(('code', code))

# ─────────────────────────────────────────────────────────────────────
INTRO = """
# The experiments, end to end

This notebook shows **the code that produced every number in the manuscript,
together with its output.**

### How to read it

| cells | outputs |
|---|---|
| §4–§5 — training and the comparator fit | **the verbatim stdout of the actual runs**, shipped unedited under `logs/`. Roughly 12 CPU-hours; not re-executed when this notebook is built. |
| §1–§3 — data build, the null, the model | the code path with **the values it produces**, taken from those same runs. Each is independently re-derivable from `results/`. |
| §6–§10 — aggregation, adjudication, figures | **executed when this notebook was written**, from the committed per-fold results. A few seconds. |

Nothing here needs MIMIC-IV access to *read*. Re-running §1–§5 does
(see `docs/REPRODUCE.md`). `verify.py` re-checks §6–§9 independently.

---
"""

# ── §1 자료 ──────────────────────────────────────────────────────────
heavy("""
## 1. The cohort

`data.prep` applies the prespecified pruning to convergence: providers with
≥200 events, drugs with ≥200 events, provider×drug cells with ≥5 events.
`Not Given per Sliding Scale` is excluded from the primary outcome.
""",
"""import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd, data

e = data.frame()                 # MIMIC-IV emar + admissions/patients/dx/icu/transfers
d = data.prep(e)                 # codes, covariates, patient-disjoint folds

print(f"rows              {len(d['y']):,}")
print(f"providers         {int(d['n_nurse'])}")
print(f"drugs             {int(d['n_drug'])}")
print(f"non-administration rate  {float(d['y'].mean()):.5f}")
print(f"folds (patient-disjoint) {np.bincount(d['fold'])}")""",
"""rows              4,362,548
providers         936
drugs             522
non-administration rate  0.09285
folds (patient-disjoint) [869183 870947 867192 898456 856770]""", 1)

# ── §2 귀무 ──────────────────────────────────────────────────────────
heavy("""
## 2. The null — and the correction

Every reported quantity is a **net increment**: the observed gain minus the gain
obtained when provider identity is destroyed. The null therefore decides the answer.

Our first implementation was wrong. `nh[hadm] = nur` is a numpy fancy-index
assignment, which keeps only the **last** write per index, so every admission
collapsed to a single provider. The corrected design redraws each admission's
*k* distinct providers from the global pool with frequency weights and keeps the
within-admission assignment pattern.
""",
'''print(data.shuffle_nurse.__doc__)

hadm, nur = d['hadm'], d['nur']
def diag(lab, name):
    g = pd.DataFrame({'h': hadm, 'n': lab}).groupby('h').n.nunique()
    c = np.bincount(lab, minlength=936); c = c[c > 0]
    print(f'{name:14} providers/admission mean {g.mean():.2f} median {g.median():.0f} '
          f'max {g.max():4d} | provider events median {np.median(c):6.0f} '
          f'max {c.max():7,} | unchanged {(lab == nur).mean():.4f}')

diag(nur, 'observed')
for k in (0, 1):
    diag(data.shuffle_nurse(d, k, 'A'), f'null A #{k}')''',
'''★귀무 — 제공자 «정체성»만 파괴하고 설계 구조는 보존한다.

    구판(v1)은 `nh[hadm] = nur` 의 팬시 인덱싱 때문에 입원마다 «마지막 한 명»만
    남아, 158,422개 입원 전부가 제공자 1명으로 접혔다. 실제 자료는 입원당
    평균 3.02명(중앙 2, 최대 635)이고 행의 90.7%가 다제공자 입원에 속한다.
    구판은 다른 귀무였다.

    design='A' (주판) 입원별 «제공자 라벨 재배정»
        각 입원의 서로 다른 제공자 k명을 전역 제공자 풀에서 빈도 가중으로
        비복원 추출한 k명으로 «갈아끼운다». 어느 행이 그중 누구에게 가는지의
        패턴(다중도)은 그대로 둔다.
        보존 : 입원당 행 수 · 입원당 제공자 수 · 입원 내 배정 패턴 · 제공자 빈도 분포
        파괴 : 제공자–환자/약물 결합

    design='B' (민감도) 크기 층화 «입원 블록» 교환
        (행 수, 제공자 수)가 같은 입원끼리 배정 벡터를 통째로 교환한다.
        구조를 «정확히» 보존하나 층이 작으면 교환 상대가 없다(그런 입원은 유지).
    

observed       providers/admission mean 3.02 median 2 max  635 | provider events median   2654 max  46,637 | unchanged 1.0000
null A #0      providers/admission mean 3.02 median 2 max  635 | provider events median   2754 max  42,465 | unchanged 0.0026
null A #1      providers/admission mean 3.02 median 2 max  635 | provider events median   2724 max  42,833 | unchanged 0.0025''', 2)

heavy("""
For contrast, this is what the broken null produced. It is kept in the
repository under `results/fold_json_v1_broken_null/` because the manuscript
reports the correction.
""",
"""# the superseded implementation, for the record
def shuffle_v1_broken(d, k):
    nh = np.zeros(d['hadm'].max() + 1, dtype=np.int64)
    nh[d['hadm']] = d['nur']                       # only the LAST write survives
    return np.random.default_rng(70 + k).permutation(nh)[d['hadm']]

diag(shuffle_v1_broken(d, 0), 'null v1')""",
"""null v1        providers/admission mean 1.00 median 1 max    1 | provider events median   2505 max 182,967 | unchanged 0.0020""", 3)

# ── §3 모형 ──────────────────────────────────────────────────────────
heavy("""
## 3. The model

Two trainable latent tables (providers 936×64, drugs 522×64). Training alternates
**turns**: on a provider turn the drug table is frozen and only the provider table
is updated; on a drug turn the reverse. The partner table is synchronised at the
start of each turn.

`A` and `B` also carry the drug latent — without that alignment `C − B` mixes the
provider latent with extra drug-side capacity (§4.5 of the manuscript).
""",
"""import cdld_core as CC
print(CC.__doc__)

sizes, ncov = list(d['sizes']), d['cov'].shape[1]
kw = dict(n_nurse=int(d['n_nurse']), n_drug=int(d['n_drug']))
mA = CC.Finder(sizes, ncov, n_drug=kw['n_drug'], drug_latent=True)
mB = CC.Finder(sizes, ncov, drug_latent=True, **kw)
mC = CC.Finder(sizes, ncov, latent=True, **kw)
for n, m in (('A', mA), ('B', mB), ('C', mC)):
    tot = sum(p.numel() for p in m.parameters())
    emb = sum(p.numel() for k_, p in m.named_parameters() if 'tab' in k_ or 'add' in k_)
    print(f'  {n}  parameters {tot:,}  (embedding tables {emb:,}, network {tot-emb:,})')""",
"""CDLD — Cyclic Dual Latent Discovery. 간호사 × 약물 투약 누락.

정본(`cdld_tutorial_fixed.ipynb` build_finder_model)의 구조를 그대로 옮긴다.
프레임워크만 Keras -> PyTorch (컨테이너에 TF 없음, CPU 2코어).

  스트림   Dense(96|64|96, swish) -> BatchNorm -> Dropout
  결합     concat -> BN -> 256 -> BN -> Drop.3 -> 128 -> BN -> Drop.3
                        -> 64 -> BN -> Drop.2 -> Dense(1, sigmoid)
  손실     BCE          최적화 Adam(5e-4)        LATENT_SIZE 64
  순환     ULD 턴(간호사 잠재만 학습, 약물 잠재 동결)
        <-> ILD 턴(약물 잠재만 학습, 간호사 잠재 동결)
           턴마다 «상대 표를 동기화»한 뒤 학습한다. 이것이 CDLD 다.


  A  parameters 117,254  (embedding tables 33,952, network 83,302)
  B  parameters 118,448  (embedding tables 34,888, network 83,560)
  C  parameters 209,552  (embedding tables 94,840, network 114,712)""", 4)

# ── §4 학습 ──────────────────────────────────────────────────────────
heavy("""
## 4. Training

One unit of work is one `(arm, fold, model level)`. Every unit writes a JSON
checkpoint, so runs are interruptible and resumable. Arms are `real` plus three
provider-shuffled nulls.

Below is the **recorded stdout of the main grid** (30 units re-run after the null
was corrected; the 15 `real` units were unaffected by the null and were reused).
""",
"""!python src/run_cdld.py 100000            # main grid""",
L('run_main.log'), 5)

heavy("""
The ward-controlled grid puts a 64-dimensional care-unit latent into all three
levels so that ward × drug interaction is representable.
""",
"""!python src/run_cdld.py 100000 ward       # ward and ward-by-drug controlled""",
L('run_ward.log'), 6)

heavy("""
Sensitivity arms. These use reduced folds / null replicates; the reduction is
stated in the manuscript's setting table.
""",
"""!python src/run_sens.py temporal 100000   # later years held out
!python src/run_sens.py cap16    100000   # latent 64 -> 16
!python src/run_sens.py top100   100000   # top-100 drugs only""",
L('run_temporal.log') + '\n\n' + L('run_cap16.log') + '\n\n' + L('run_top100.log'), 7)

# ── §5 대조 추정기 ───────────────────────────────────────────────────
heavy("""
## 5. The comparator — low-rank bilinear, alternating least squares

The same ladder, fitted with **no neural network, no gradient descent and an
inner-product interaction**: a Gauss–Seidel demeaned additive baseline plus a
low-rank bilinear term fitted by alternating least squares. Rank, ridge and
weighting are chosen by 4-fold cross-validation *inside* the training folds.

It uses the same data, the same fold assignment and the same null seeds.
""",
"""!python src/run_prereg.py none all        # low-rank bilinear, main
!python src/run_ward.py                   # low-rank bilinear, ward gate""",
L('run_als_main.log') + '\n\n' + L('run_als_ward.log'), 8)

# ── §6~ 가벼운 셀 (실제 실행) ─────────────────────────────────────────
light("""
---

## 6. Per-fold results

Everything below runs in seconds from the committed outputs.
""",
"""import pandas as pd, numpy as np
fold = pd.read_csv('results/fold_level.csv')
print(f'{len(fold)} units of work')
fold.groupby(['tag','kind','arm']).r2.agg(['mean','count']).round(4).head(12)""")

light("""
## 7. Net increments and the ladder
""",
"""def r2(x, arm, kind):
    s = x[(x.arm == arm) & (x.kind == kind)].sort_values('fold'); return s.r2.to_numpy()

def ladder(tag, n_null=3):
    x = fold[fold.tag == tag]
    A, B, C = r2(x,'real','A'), r2(x,'real','B'), r2(x,'real','C')
    SB = np.mean([r2(x,f'shuffle{k}','B') for k in range(n_null)], axis=0)
    SC = np.mean([r2(x,f'shuffle{k}','C') for k in range(n_null)], axis=0)
    return dict(A=A, B=B, C=C, main=(B-A)-(SB-A),
                latent=(C-B)-(SC-SB), total=(C-A)-(SC-A), shuf=(SC-SB))

b = ladder('base')
for k in ('A','B','C'):
    print(f'  observed {k}   {b[k].mean():+.4f}   folds {np.round(b[k],4)}')
print()
for k in ('main','latent','total'):
    print(f'  net {k:7s}  {b[k].mean():+.4f}  +/- {b[k].std(ddof=1):.4f}')
print(f'  shuffled-arm latent {b["shuf"].mean():+.5f}   (a correct null should give ~0)')""")

light("""
## 8. The two estimator families, and the adjudication
""",
"""pd.read_csv('results/table2_two_estimators.csv')""")

light("", """pd.read_csv('results/table2b_adjudication.csv')""")

light("""
> **The same prespecified criterion returns opposite verdicts under the two
> estimator families, on the same data, folds and null.**
""",
"""t = pd.read_csv('results/table3_settings.csv')
t['R']   = np.where(t.total  >= .010, 'met', 'NOT MET')
t['Q2']  = np.where(t.latent >= .005, 'met', 'NOT MET')
t['Q2b'] = np.where(t.ratio  >= 1.0,  'met', 'NOT MET')
print(t[['setting','folds','nulls','main_effect','latent','total','ratio',
         'R','Q2','Q2b']].to_string(index=False))
print()
for c in ('R','Q2','Q2b'):
    print(f'  {c:4s} met in {(t[c]==\"met\").sum()}/{len(t)} settings')
print()
print('  Only the entity total (R) survives every setting.')
print('  Q2b separates exactly by estimator family.')""")

light("""
## 9. Figures
""",
"""from IPython.display import Image, display
import os
for f in sorted(os.listdir('figures')):
    if f.endswith('.png'):
        print(f); display(Image('figures/' + f))""")

light("""
## 10. Independent re-check
""",
"""import subprocess, sys
print(subprocess.run([sys.executable, 'verify.py'], capture_output=True, text=True).stdout)""")

# ── 조립: 가벼운 셀만 먼저 실행 ───────────────────────────────────────
lnb = nbf.v4.new_notebook()
lc = []
for item in LIGHT:
    if item[0] == 'md':
        if item[1].strip(): lc.append(nbf.v4.new_markdown_cell(item[1].strip()))
    else:
        lc.append(nbf.v4.new_code_cell(item[1].strip()))
lnb['cells'] = lc
lnb.metadata['kernelspec'] = dict(name='python3', display_name='Python 3')
nbf.write(lnb, '.light.ipynb')
subprocess.run(['jupyter','nbconvert','--to','notebook','--execute','--inplace',
                '.light.ipynb','--ExecutePreprocessor.timeout=300'], check=True,
               capture_output=True)
executed = nbf.read('.light.ipynb', as_version=4).cells

# ── 조립: 무거운 셀 + 실행된 가벼운 셀 ────────────────────────────────
cells = [nbf.v4.new_markdown_cell(INTRO.strip())]
for item in HEAVY:
    if item[0] == 'md':
        cells.append(nbf.v4.new_markdown_cell(item[1].strip()))
    else:
        _, code, out, n = item
        c = nbf.v4.new_code_cell(code.strip())
        c.execution_count = n
        c.outputs = [stream(out)]
        cells.append(c)
cells += executed
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata['kernelspec'] = dict(name='python3', display_name='Python 3')
nbf.write(nb, 'experiments.ipynb')
os.remove('.light.ipynb')
nh = sum(1 for i in HEAVY if i[0] == 'code')
ne = sum(1 for c in executed if c.cell_type == 'code')
print(f'experiments.ipynb  cells {len(cells)}  (recorded {nh}, executed {ne})')
