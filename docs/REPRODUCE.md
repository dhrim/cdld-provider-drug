# Re-running the experiments

`verify.py` reproduces every number in the paper **without any data access**.
This document is for re-running the experiments themselves, which requires
credentialed access to MIMIC-IV.

## 1. Data

Obtain **MIMIC-IV v3.1** from PhysioNet (credentialing + Data Use Agreement).
The analysis needs:

```
emar.csv.gz            hosp/     (subset the columns; see src/build_emar.py)
emar_detail.csv.gz     hosp/     (provider-field diagnostics only)
admissions.csv.gz      hosp/
patients.csv.gz        hosp/
diagnoses_icd.csv.gz   hosp/
icustays.csv.gz        icu/
transfers.csv.gz       hosp/     (ward assignment; src/map_careunit.py)
```

`src/build_emar.py` produces the slimmed eMAR file; `src/map_careunit.py` maps
each administration event to a care unit via `transfers` (98.8% assigned).

**Do not redistribute these files or any derived cache.**

## 2. Cohort

`src/data.py` applies the prespecified pruning to convergence: nurses with
≥200 events, drugs with ≥200 events, cells with ≥5 events. `Not Given per
Sliding Scale` is excluded from the primary outcome.

The build must land on exactly:

```
rows                4,362,548
nurses x drugs      936 x 522
omission rate       0.09285
```

If it does not, stop — something upstream differs.

## 2b. Paths — nothing is hard-coded any more

All scripts resolve their locations through `src/paths.py`. Override with environment
variables; the defaults keep everything inside the repository.

```bash
export CDLD_WORK=./work          # checkpoints and caches   (default: <repo>/work)
export CDLD_DATA=./work/cache    # MIMIC-derived inputs     (default: <CDLD_WORK>/cache)
export EMAR_SLIM=$CDLD_DATA/emar_slim.csv.gz
export MIMIC_HOSP=$CDLD_DATA/hosp
```

`src/build_emar.py <path to MIMIC-IV hosp/emar.csv.gz>` writes the slimmed eMAR file
(nine columns; see the script). MIMIC-IV requires PhysioNet credentialing and neither
the raw nor the derived files are redistributed here.

**Result verification needs none of this** — `verify.py`, `src/verify_v5.py`,
`src/verify_v6.py` and `src/verify_v8.py` read only committed result files and exit 0
without any data. `verify_v8.py` additionally has a *tier 2* block that recomputes the
null's frequency distribution from the MIMIC-derived cache; with no cache it prints
`SKIP` and the exit code is unaffected. It was verified that way — with the cache moved
out of the way — and that is the run whose transcript is shipped.

## 3. Folds and the null

Both are constructed identically for both estimators, from fixed seeds:

```python
# patient-disjoint 5-fold
pt, pl = pd.factorize(e.subject_id)
fold = np.random.default_rng(42).integers(0, 5, len(pl))[pt]
```

The null relabels providers **within each admission**, preserving exactly the rows per
admission, the providers per admission, and the within-admission assignment pattern
(multiplicity). Only the provider-to-patient/drug pairing is destroyed.

It **approximately** preserves the marginal provider-frequency distribution: the mean is
exact by construction and the rank order is nearly so (Spearman 0.99), but the tails
compress — and the minimum falls below the cohort's own inclusion threshold of 200 events
per provider. The numbers are committed in `results/v6/table8_null_frequency.csv`.

| | real | null 0 | null 1 | null 2 |
|---|---|---|---|---|
| mean events | 4,661 | 4,661 | 4,661 | 4,661 |
| median | 2,654 | 2,754 | 2,724 | 2,691 |
| max | 46,637 | 42,465 | 42,833 | 41,481 |
| **min** | **200** | **99** | **94** | **108** |
| Spearman vs real | — | 0.991 | 0.990 | 0.991 |

See `data.shuffle_nurse`:

```python
w = np.bincount(nur, minlength=n_nurse).astype(float); w /= w.sum()
for s0, s1 in zip(starts, ends):              # one admission
    idx = order[s0:s1]
    uq, inv = np.unique(nur[idx], return_inverse=True)
    out[idx] = rng.choice(pool, size=len(uq), replace=False, p=w)[inv]
```

> **Correction, recorded.** An earlier implementation wrote
> `nh = np.zeros(hadm.max()+1, int); nh[hadm] = nur` and permuted that. NumPy
> fancy-index assignment keeps only the last write per admission, so all 158,422
> admissions collapsed to a single provider — while the real data has 3.02
> providers per admission on average (median 2, max 635), with 90.7% of rows in
> multi-provider admissions. **That was a different null.** Everything was rerun
> after the fix; the impact was asymmetric (−20% for the cyclic dual-latent
> estimator, −2% for the low-rank one). The superseded per-fold results are kept
> under `results/fold_json_v1_broken_null/`, and the broken implementation is
> reproduced in `experiments.ipynb` §2 for the record.
>
> **Null construction is part of the estimand, not a technical implementation
> detail.**

## 4. Run

Each unit of work is one `(arm, fold, model level)` and writes a JSON
checkpoint, so runs are interruptible and resumable. Pass a per-invocation time
budget in seconds.

```bash
python src/run_cdld.py 100000            # main grid, 35 units   (~4 h)
python src/run_cdld.py 100000 ward       # ward-controlled, 35 units (~4 h)
python src/run_sens.py temporal 100000   # sensitivity arms
python src/collect.py                    # both ladders
python src/final.py                      # adjudication
python src/final_ward.py                 # ward gate
```

Alternating-least-squares arm:

```bash
python src/run_prereg.py none            # main ladder (real + shuffle0 + shuffle1)
python src/run_prereg.py none shuffle2   # third null replicate
python src/run_prereg.py nohold          # sensitivity: 'Hold Dose' excluded from the outcome
python src/run_prereg.py slide           # sensitivity: 'Not Given per Sliding Scale' included
python src/run_ward.py                   # ward gate
```

**Background jobs die when the host session suspends.** Because every unit
checkpoints, simply re-issuing the same command resumes from where it stopped.

## 5. Interpretation

The latent *directions* are analysed from a **separate long fit on the full
data**, never from the fold models used for adjudication:

```bash
python src/interp_fit.py 100000   # 8 cycles x 2 epochs per turn
python src/direction.py           # clusters + permutation test
python src/profiles2.py           # the two-nurse table
```

One detail matters: `interp_fit.py` **excludes the latent tables from weight
decay**. With decay applied to them, the embedding norms collapse from ~0.80 to
~0.125 — predictions are unaffected, but the directions become too weakly
trained to interpret.

## 6. Expected runtime

On 2 CPU cores, ~350 s per A/B unit and ~370 s per C unit; roughly 12 hours for
the full grid plus the ward gate. A GPU is not required and was not used.

## 7. What is checked

`verify.py` asserts the reported values against the committed per-fold outputs
and exits non-zero on any mismatch, including the **deliberately failed**
criterion Q2b.


---

## 8. v5 — the decomposition-aligned experiment

The analysis plan for this diagnostic was **frozen before it was run**; see
`protocol/2026-09-06_analysis_amendment_frozen.md`, which carries its own
completion timestamp and the sha256 of its body.

```bash
# unit tests: B additive in the provider; C's second difference == the bilinear form
python src/cdld_a.py

# native B/C refit with checkpoints (the original runs saved metrics, not weights)
for f in 0 1 2 3 4; do python src/retrain_b.py $f; python src/retrain_c.py $f; done

# function-level diagnostics
python src/diag_d.py        # non-additivity of native B on a supported grid
python src/diag_fn.py       # two-way variance decomposition, native C vs CDLD-A

# the aligned ladder — A is SHARED with the native estimator, so it costs nothing
python src/run_cdld_a.py         # run 1: nominal hyperparameters   -> fails the fit gate
python src/run_cdld_a2.py        # run 2: latent tables excluded from L2
python src/run_cdld_a2_ward.py   # ward and ward-by-drug controlled

# adjudication and gates
python src/adj_a2.py        # gamma*, kappa, the identity, the attribution
python src/boot_gate2.py    # patient-level paired bootstrap, 2000 replicates
python src/collect_v5.py    # all v5 tables
python src/verify_v5.py     # recomputes all 26 reported numbers; non-zero exit on mismatch
```

`verify_v5.py` reads only committed result files under `results/v5/`, so it runs
in seconds without MIMIC access. `figs_v5.py` regenerates the four v5 figures.

**Two runs of the aligned model are committed, not one.** Run 1 failed the
prespecified fit gate because the bilinear latent tables collapsed (‖u‖ 0.80 →
0.038) under a weight decay that is harmless in the native architecture, where a
BatchNorm follows the latent stream and restores the scale. Run 2 changes the
regulariser applied to those two tables and nothing else — same architecture, same
schedule, same seed, same folds. Both are reported.


---

## 9. v6 — separating the readout from the regularisation, and the function-level diagnostics

```bash
python src/run_nat_nodecay.py     # unconstrained readout, latent tables excluded from L2
python src/run_a2_mid.py          # aligned readout, latent L2 = 1e-5
python src/run_a2_null3.py        # third null replicate for the aligned arm
python src/run_seed.py 1 ; python src/run_seed.py 2   # seed repeats of the B stage
python src/diag_fn2.py            # function decomposition with the PRESPECIFIED metrics
python src/collect_v6.py
python src/verify_v6.py           # 28 checks
python src/figs_v6.py
```

`diag_fn2.py` reports `frac_inter` and `sd(D)/sd(M)` as the primary metrics (that is what
the first amendment fixed; v5 had promoted a secondary one), weights the grid by the
**product** of the provider and drug marginal frequencies within the selected grid, and
computes `corr(c_native, c_aligned)` on the interaction **component** — the raw η
correlation is 0.96 only because the drug main effect is 90% of the grid variance, and is
not evidence that two models learned similar functions.

---

## 10. v7 — the alternative null, run to completion

The second amendment promised the alternative null's *performance*, not just its design.
It is a property of the null, not of the estimator, so it was run on the low-rank arm.

```bash
python src/run_prereg.py none blockswap0    # size-stratified admission block swap
python src/run_prereg.py none blockswap1
python src/diag_null_B.py                   # permutation strength (no model fitting)
python src/verify_v8.py                     # tier 1 only, no data needed
python src/figs_v8.py                       # regenerates the four figures
```

`null_B(seed)` in `run_prereg.py` swaps whole admissions between providers within size
strata. It preserves more structure than the primary null — and that is the problem.

**Three numbers, and they count different things.** Up to v7 the manuscript conflated the
first two:

    1.6%   admissions in a stratum with NO eligible exchange partner (structural floor)
    3.5%   admissions whose provider vector is ACTUALLY unchanged after the swap
   31.7%   rows that keep their original provider identity

`1.6%` comes from a counter that only asks whether a stratum *had* a partner; a random
permutation of a size-2 stratum is the identity half the time, so the measured figure is
more than double it. `src/diag_null_B.py` computes all three and writes
`results/v7/table12_null_B_diag.csv`; `verify_v8.py` asserts that the second exceeds the
first, so the two can never be conflated again.

The consequence is measured, not asserted: `S` turns positive (+0.0014) under this null,
so the null arm still predicts. `R` and `Q2b` hold under both designs — **in the low-rank
estimator, which is the only one run under the alternative null.**

Committed under `results/v7/`:

| file | what |
|---|---|
| `table10_alt_null_arms.csv` | the five arm-level A/B/C values at full precision |
| `table10_alt_null.csv` | the ladders derived from them — primary at 2 nulls (matched to the alternative), the alternative at 2, and the 3-null primary for reference |
| `arm_none_blockswap{0,1}.json` | the raw runner checkpoints |
| `table11_two_by_two_v7.csv` | the 2×2, all five cells at **2** null repetitions |
| `table2_attribution_v7.json` | `gamma*` / `kappa` / the identity, given separately for the 3-null headline and the 2-null matched cells |
| `table7_latent_norms_v7.json` | latent-table norms at all three regularisation levels — Figure 4 is drawn from this, so it needs no model weights |
| `table12_null_B_diag.csv` | the three permutation-strength numbers above |

`verify_v7.py` rebuilds `table10_alt_null.csv` from the arm-level file, rebuilds all five
2×2 cells from the per-fold results, re-derives both identities (residual 0.00e+00), and
asserts the direction of every claim the manuscript makes about the alternative null.


---

## 11. Every command above is smoke-tested from a clean extraction

The reproduction path is part of the deliverable, so it is checked the same way the numbers
are. `logs/v8_smoke_test.txt` is the transcript of running every command in this document
and in the README against a fresh `tar xzf` of this repository, with no MIMIC-IV data
present. The contract is:

* commands that need no data **exit 0** — the four verifiers, both figure generators, the
  `cdld_a.py` unit tests;
* commands that need MIMIC-IV **stop cleanly** with a `[cdld] …` line naming the missing
  input, via `paths.need_data()`. They do not raise `AttributeError` on a `None` frame, do
  not raise `FileNotFoundError` on a hard-coded path, and do not write outside the repo.

Up to v7 this was not true. `run_prereg.py` and `run_ward.py` imported a module name that
does not exist in this repository (`nd_core`, renamed to `als_core`); `figs_v7.py` wrote its
output to an absolute path from the development machine, so it *appeared* to succeed there
and failed everywhere else; and thirteen scripts crashed with `AttributeError` instead of
saying what was missing, because `data.prep(None)` returned `None` when its cache was absent.
All of that is fixed, and the smoke test exists so it stays fixed.
