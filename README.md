# What transfers across estimators, and what does not

Code and per-fold results for a study of **provider-level structure in documented
medication non-administration** (MIMIC-IV eMAR, 4,362,548 events, 936 recording
providers × 522 medications).

The identical analysis — same data, same patient-disjoint folds, same provider-shuffled
null — was run under **three** estimators that differ in whether the readout implements
the main-effect / interaction decomposition.

| | readout | main | latent | **total (C−A)** | **ratio** | **Q2b (≥ 1.0)** |
|---|---|---|---|---|---|---|
| **Cyclic dual latent discovery** | unconstrained nonlinear | +0.0096 | +0.0057 | **+0.0154** | **0.59** | **NOT MET** |
| **CDLD-A** (decomposition-aligned) | additive + bilinear | +0.0044 | +0.0107 | **+0.0151** | **2.46** | met |
| **Low-rank bilinear** (ALS) | additive + bilinear | +0.0075 | +0.0112 | **+0.0188** | **1.50** | met |

> **All three agree that including the provider helps, and by nearly the same amount.**
> The two that make the decomposition explicit also agree on where the help comes from.
> The one that does not, disagrees.

> Estimator family alone was insufficient to explain the observed split in these
> comparisons. `native` and `CDLD-A` differ in more than the readout (four things change
> together — see the manuscript's Methods), so the readout is not isolated as *the* cause;
> what the 2×2 below rules out is the *regularisation-only* explanation.

**And the split is not the regularisation.** The aligned model's success was obtained
after removing weight decay from its latent tables, so a 2×2 crossing readout with
regularisation was run — with the decision rule frozen in advance:

| readout | latent L2 = 1e-4 | latent L2 = 1e-5 | latent L2 = 0 |
|---|---|---|---|
| **unconstrained** | **0.59** | — | **0.46** |
| **aligned** | 0.22 *(tables collapsed; fit gate failed)* | **1.94** | **2.34** |

*Every cell of this 2×2 uses **2** null repetitions so the cells are matched; the headline
table above uses 3. `results/v7/table11_two_by_two_v7.csv` carries the matched cells and
`src/verify_v8.py` recomputes all five from the per-fold results, and asserts that no
comparison table mixes null counts.*

Removing weight decay under the unconstrained readout does not reproduce the shift — the
ratio moves the other way. The aligned readout crosses 1 at every setting where it fits.
The *magnitude* of the ratio is itself sensitive to regularisation, so the claim is about
the direction of the verdict, not the number.

When the decomposition was made explicit, predictive skill was **reallocated** from the
B stage to the C stage (−0.0053 and +0.0050) while the total inclusion signal barely
moved (−0.0003).

**And the conclusion does not depend on the null design.** A second, structure-preserving
null (size-stratified admission block swap) was run to completion on the low-rank arm:

| null design | Δ main | Δ latent | **R (total)** | **ratio** | **S (shuffled latent ≤ 0)** |
|---|---|---|---|---|---|
| **primary** — per-admission provider relabelling | +0.0075 | +0.0112 | **+0.0188** | **1.50** | **−0.0007** ✅ |
| **alternative** — size-stratified block swap | +0.0065 | +0.0092 | **+0.0157** | **1.41** | **+0.0014** ✖ |

Both arms are 2 nulls, so the cells are matched. In the low-rank estimator R and Q2b hold
under both designs, but **S turns positive** under the alternative null: the null arm still
predicts.

Why the alternative null is the weaker permutation is measured, not asserted — and the two
kinds of evidence generalise differently:

| evidence | value | scope |
|---|---|---|
| **design** — admissions whose provider vector is *actually unchanged*; rows keeping their provider | **3.5%** · **31.7%** | a property of the permutation itself, independent of estimator |
| **performance** — the latent term still helps on the null arm (S > 0) | +0.0014 | observed in the low-rank estimator only |

> These results support retaining the primary null as the stronger provider-identity
> perturbation for the primary analysis.

Neither is called a perfect null, and we do **not** claim the paper's conclusions are
independent of null design — one estimator was run under one alternative.

*(Correction, twice over. v6 reported "32% of admissions"; the unit was wrong. v7 corrected
it to 1.6%, but that counter measures **admissions in a stratum with no exchange partner**,
not admissions that actually came back unchanged — a random permutation of a size-2 stratum
is the identity half the time. The measured figure is **3.5%**; `results/v7/table12_null_B_diag.csv`
and `src/diag_null_B.py` carry it. The correction makes the "weaker null" case stronger, which
is not a reason to have delayed it.)*

And decomposing the *fitted function* on the same provider × drug grid gives 0.042 vs
0.023 on the prespecified metric — same order of magnitude, **ordering reversed**
relative to the skill-based ratio. The two models' interaction *components*, however,
correlate at only ρ ≈ 0.50, so we do **not** claim they learned similar functions.

The two decompositions are **different estimands**, so neither is the "correct" version of
the other and neither mismeasures the other's quantity — they yielded opposite model
orderings from the same fitted-model comparison, and that is the finding.

> **Cross-estimator comparisons are meaningful only for quantities whose estimands remain
> invariant across model parameterizations.** Inclusion contrasts satisfy that. Contrasts
> that depend on an internal decomposition satisfy it only when the architecture
> implements that decomposition.

---

## Read everything without installing or running anything

Two notebooks, both committed **with their outputs already in the file**:

| | |
|---|---|
| **[`experiments.ipynb`](experiments.ipynb)** | the whole study end to end — data build, the null and its correction, the model, the training runs, the comparator fit, aggregation and adjudication, then **§11–§17: the decomposition-aligned experiment; §18–§21: separating readout from
regularisation; §22–§24: the alternative null, run to completion; §25–§27: two corrections and the reproduction-path smoke test**. §4–§5 and §12 carry the **verbatim stdout of the actual runs** (also shipped raw under [`logs/`](logs/)); the rest was executed when the notebook was written. |
| **[`reproduce.ipynb`](reproduce.ipynb)** | the short version — just the tables, figures and adjudication, all executed from the committed per-fold results. |

Neither needs MIMIC-IV access to read.

## Re-check the numbers yourself — no MIMIC-IV access needed

```bash
pip install numpy pandas          # that is all the verifiers need
python verify.py                  # the v4 analysis
python src/verify_v5.py           # the v5 decomposition-aligned experiment (26 checks)
python src/verify_v6.py           # the v6 2x2 and function-level diagnostics (28 checks)
python src/verify_v8.py           # + the alternative null and every shipped v7/v8 table
```

`verify_v8.py` is split into **two tiers**: tier 1 reads only committed CSV/JSON and
covers every reported number; tier 2 recomputes the null's frequency distribution from the
MIMIC-derived cache and **skips cleanly, without affecting the exit code**, when that cache
is absent. The committed `verify_v8_output.txt` in the manuscript bundle is the run with
the cache **removed** — that is the run a reader can reproduce.

**Every command in this README and in `docs/REPRODUCE.md` is smoke-tested from a clean
extraction of this repository** — see `logs/v8_smoke_test.txt`. Scripts that need
MIMIC-IV stop with a one-line `[cdld] …` message naming what is missing; none of them
crash, and none of them write outside the repository.

A few seconds each. They recompute the paper's tables and every prespecified
criterion from the committed per-fold results, and **exit non-zero if any reported
value fails to reproduce** — including the criteria that fail. `verify_v5.py` also
re-derives the `gamma*` / `kappa` identity (residual 0.00e+00) and asserts that the
skill-based and function-based decompositions order the two models oppositely.

## The analysis plan for the v5 experiment was frozen before it was run

[`protocol/2026-09-06_analysis_amendment_frozen.md`](protocol/2026-09-06_analysis_amendment_frozen.md)
and [`protocol/2026-09-06_second_amendment_frozen.md`](protocol/2026-09-06_second_amendment_frozen.md)
each carry a completion timestamp and the sha256 of their own body.
`verify_v6.py` and `verify_v8.py` assert the second amendment's decision rule, so moving a
threshold after the fact makes the scripts fail.

The **first** amendment fixes, in advance: the fit gate and its margin, the `kappa`
threshold at which this very experiment would **falsify the paper's central claim**, the
attribution rule, and four interpretation branches — including what would be reported if
the result went against us.

**But it had no contingency clause for a failed fit gate, and one run did fail it.** The
remedy that was applied — excluding the latent tables from weight decay — appears only in
the internal review correspondence, not in the frozen plan. So it was a decision made
*after seeing* the failure, it is reported as a **deviation**, and the 2×2 above exists
precisely to test whether that decision, rather than the readout, drove the result. The
second amendment opens by naming this mistake and fixes the adjustable set and the
decision rule before running anything.

**Two runs of the aligned model are committed, not one.** The first failed the
prespecified fit gate; the cause was a latent-table collapse (‖u‖ 0.80 → 0.038)
under a weight decay that is harmless in the native architecture because a
BatchNorm follows the latent stream there. The remedy changes the regulariser on those
two tables and nothing else — same architecture, schedule, seed, folds and B stage.
Both runs are reported.

## Re-run the experiments (≈20 CPU-hours, credentialed MIMIC-IV access)

See [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

---

## The null, and a correction we report

Every reported quantity is a **net increment**: the observed gain minus the
gain obtained when provider identity is destroyed. The null therefore decides
the answer.

Our first implementation was wrong:

```python
nh = np.zeros(hadm.max() + 1, dtype=np.int64)
nh[hadm] = nur                  # fancy-index assignment keeps only the LAST write
shuffled = rng.permutation(nh)[hadm]
```

Every admission collapsed to a **single** provider, although the data has 3.02
distinct providers per admission (median 2, max 635) and 90.7% of rows sit in
multi-provider admissions.

```
                                      providers/admission      provider events
                                      mean  median   max      median      max
observed data                         3.02       2   635       2,654   46,637
null v1  (broken)                     1.00       1     1       2,505  182,967
null v2  (corrected, used here)       3.02       2   635       2,754   42,465
```

Correcting it moved the cyclic-dual-latent estimate by **20%** and the low-rank
estimate by **2%**. Both are reported. `results/fold_json_v1_broken_null/`
retains the superseded runs.

**A null adequate for one estimator can be inadequate for another** — which is
itself one of the paper's points.

---

## What is here

```
experiments.ipynb            ★ the study end to end, outputs included (training logs verbatim)
reproduce.ipynb              ★ short version: tables, figures, adjudication - all executed
logs/                        raw stdout of every run, unedited
verify.py                    ★ recompute every reported number; non-zero exit on mismatch
results/
  fold_level.csv             every out-of-sample fold result (R2, AUC, SSE, SStot)
  table1_cohort.csv          the cohort as analysed
  table_null_design.csv      observed vs broken null vs corrected null
  table2_two_estimators.csv  the ladder under both estimator families
  table2b_adjudication.csv   the five prespecified criteria and their verdicts
  table3_settings.csv        every setting side by side
  fold_json/                 raw per-unit outputs
  als/                       low-rank bilinear outputs
  v5/ v6/                    the decomposition-aligned experiment and the 2x2
  v7/                        the alternative null (arm-level and derived), the matched
                             2x2 table, the attribution table split by null count, the
                             latent-norm table Figure 4 is drawn from, and the null_B
                             permutation-strength diagnostic
figures/                     the paper's figures (PDF + PNG)
src/
  cdld_core.py               cyclic dual latent discovery (PyTorch)
  als_core.py                alternating least squares + Gauss-Seidel demeaning
  data.py                    cohort, covariates, folds, and the null (both designs)
  run_cdld.py run_sens.py    main / ward-controlled / sensitivity runners
  interp_fit.py direction.py profiles2.py    latent interpretation
  adj2.py figs_v2.py build_repo_v2.py        aggregation, figures, result files
  run_prereg.py run_ward.py  the low-rank bilinear arm (both nulls)
  diag_null_B.py             permutation strength of the alternative null
  paths.py                   every path in this repository resolves through here
  verify_v8.py figs_v8.py    current verifier and figure generator
protocol/                    the analysis protocol, fixed before the confirmatory runs
```

**No raw or derived MIMIC-IV data is redistributed.** `results/` holds only
aggregate out-of-sample statistics.

## The three model levels

```
A : shift + drug + diagnosis chapter + patient covariates + drug latent   (no provider)
B : A + provider identity (1-dim)
C : B + provider latent (64-dim), discovered cyclically against the drug latent
```

`A` and `B` carry the drug latent too. Without that alignment `C − B` mixes the
provider latent with extra drug-side capacity, and the shuffled-provider null
turns positive — the null catching a confounded model contrast.

## Scope of `verify.py`

`verify.py` checks that the **reported numbers follow from the committed
per-fold results**. It does not, and cannot, check that those results estimate
the intended quantity. Study design questions — whether the recording provider
is the administering clinician, whether the outcome is an omission or a
documented non-administration, whether admission-level covariates leak — are
discussed in the manuscript's Limitations, not settled here.

## License

Code: MIT. Result files: CC BY 4.0.
MIMIC-IV is governed by the PhysioNet Credentialed Health Data Use Agreement
and is **not** included.
