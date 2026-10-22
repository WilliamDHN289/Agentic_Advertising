# Pre-registration: E5 validating the advertising unit (field experiment)

Draft for OSF Registries ("OSF Preregistration" template), SL-01. **Status: draft, not filed.**
It is filed before any field data exists, and it cites the SL-22 protocol version.

**Protocol version:** to be set by SL-22. The partner and the sample size (A10) are to be
determined. The machine-readable form is `features/storelab/docs/claims_register.yaml`, which
governs where the two differ.

## Registered files

| File | sha256 |
|---|---|
| `features/storelab/docs/claims_register.yaml` | `83c41b23565aaf68a40b58a9a9b38a4f1fb0b91b2eb8615418e215a51da85592` |
| `requirements/A1_formal_statements.md` | `be45dbfa81b0694693435f5cd2a32eedbf9370ccd24418c11250102b5dd278b5` |
| `requirements/A12_thresholds.md` | `82217db6d087b78799db59aa680f0be70bfb26040d7acc06bbd533e9c3813f9d` |
| `requirements/paper.tex` | `e4bcf05266b5c835d8dc827e829a86d1856f23bb3f295766aae899c154d63295` |

## 1. Study information

`ass:value` posits that the platform's policy affects advertiser value only through effective
choices. Sandbox experiments cannot test this, because purchases there are not real.

E5 is a field experiment with a partner that observes agent-initiated selections, for example
through an agent checkout interface, together with purchases and returns.

## 2. Design plan

Randomisation is at the request level, for a subset of products, between candidate sets with
strong and with weak competitors. The experiment records:
- the candidate qualifying events: a confirmed selection, or an add-to-cart instruction;
- clicks;
- purchases and returns.

The qualifying event is fixed in the SL-22 protocol. No PII enters the data.

## 3. Sampling plan

The sample size comes from the SL-22 power design at the registered margin and is written here
before filing.

## 5. Analysis plan

**Decision labels.** Each claim is labelled *supported*, *not supported* or *inconclusive* by
`features/storelab/analysis/decide.py`, applied to the results file the analysis code writes
(schema `contracts/storelab/analysis_results.schema.json`). No label is set by hand. The rules
(A12 §2) are:
- **Three-way rule.** A claim is *supported* if its interval lies entirely on the claim's side of
  the threshold, or inside the equivalence margin. It is *not supported* if the interval lies
  entirely on the other side, or entirely outside the margin. Otherwise it is *inconclusive*.
- **Below a floor or excluded for missingness.** A cell below its minimum N, or excluded for
  missingness, is *inconclusive*.
- **Rejection tests.** A test with no registered margin (H1a, H1c heterogeneity) is never *not
  supported*: a non-rejection is *inconclusive*.
- **Multiplicity.** Holm–Bonferroni applies within each family: the cell-level tests of one
  hypothesis and test. There is no correction across hypotheses.
- **One-sided p-values.** `decide.py` applies the interval rules through their dual one-sided
  p-values: a bound at level 1 − a lies beyond the threshold exactly when the one-sided p-value
  against it is at most a. This makes Holm's step-down adjustment exact. For counts it computes
  them itself: exact binomial for Clopper–Pearson rules, the score test for Wilson rules.
- **Significance.** α = 0.05. Intervals are 95%, and the bounds of a two-sided interval are tested
  at 0.025 each. The sign part of H1c and the slope signs are one-sided at 0.05. Equivalence uses
  TOST with the 90% interval, and "entirely outside" uses the 95% interval.
- **Several parts.** A claim with several parts is *supported* if every primary part is supported
  and *not supported* if any is not.
- **Exact results.** A failed implementation check of an exact (I-type) result is never a finding.
  The claim is *inconclusive*, flagged, and its reporting is blocked until the failure is
  root-caused.

**Operating characteristics.** Every confirmatory test must have empirical size in [0.03, 0.07] at
α = 0.05 over 1,000 Monte Carlo null replications (SL-15, SL-22). A test that fails is replaced by
its permutation version before this registration is filed.

**Deviations.** Deviations are logged as dated amendments with reasons. Thresholds, margins and
decision rules change only through a dated owner amendment made before registration, and every
amendment is listed here. Any analysis not in this document is labelled exploratory.

**H5a** uses TOST with Fisher randomisation intervals (10,000 draws) on the log ratio of purchases
net of returns per effective choice, with margin ±0.10.

**H5b** compares the out-of-sample log-loss of the qualifying event with that of clicks, by arm
and, where identifiable, by traffic type. AUC is reported.

**Interpretation.** Following the surrogate literature, a failure of H5a means that effective
choices are not a valid unit across policies. `ass:value` would then need context- or
set-dependent values, and `prop:ic` would require multidimensional bids.

## 6. Hypotheses, metrics and decision rules

### H5a

Purchases net of returns per effective choice are equal (within +-0.10 on the log ratio) between requests randomised to strong-competitor and weak-competitor candidate sets.

| | |
|---|---|
| Kind | P |
| Sources | `ass:value` |
| Experiment and mode | E5, field |
| Unit of analysis | request, randomised at the request level for a subset of products |
| Metric | log ratio of purchases net of returns per effective choice, strong vs weak arm |
| Estimator | difference in means of the ratio's logs with Fisher randomisation intervals (10,000 draws) |
| Test | TOST via Fisher randomisation intervals |
| alpha / CI level | 0.05 / 0.95 |
| Multiplicity family | none (one primary comparison) |
| Tolerance or margin | equivalence margin +-0.10 on the log ratio (about +-10%) |
| Minimum N | set by the SL-22 power design (A10: partner and sample size to be determined) |
| Decision rule | Supported if the 90% randomisation interval lies within (-0.10, 0.10); not supported if the 95% interval lies entirely outside [-0.10, 0.10]; inconclusive otherwise. |
| If supported | Effective choices are a valid advertising unit across policies (ass:value). |
| If not supported | Effective choices are not a valid unit across policies: ass:value would need context- or set-dependent values and prop:ic would require multidimensional bids (paper E5). |
| If inconclusive | The field data cannot establish invariance within +-10%. |

### H5b

The qualifying event predicts purchases better than clicks, consistently across arms and, where identifiable, across traffic types.

| | |
|---|---|
| Kind | P |
| Sources | `ass:value` |
| Experiment and mode | E5, field |
| Unit of analysis | request, by arm and, where identifiable, traffic type |
| Metric | out-of-sample log-loss difference (event - click) for predicting purchase; AUC reported |
| Estimator | cross-fitted predictions with a randomisation-respecting bootstrap |
| Test | 95% interval of the log-loss difference against 0, per arm and traffic type |
| alpha / CI level | 0.05 / 0.95 |
| Multiplicity family | H5b: the arm (x traffic type) cells (Holm) |
| Tolerance or margin | threshold 0 (superiority) |
| Minimum N | set by the SL-22 power design |
| Decision rule | Per cell: supported if the Holm-adjusted 95% upper bound is < 0; not supported if the lower bound is > 0. Claim: supported if every cell is supported; not supported if any is not supported. |
| If supported | The qualifying event is the better surrogate for purchases (prentice1989surrogate, athey2026surrogate). |
| If not supported | Clicks predict purchases at least as well in some arm or traffic type: the qualifying event is not the better unit there. |
| If inconclusive | The partner data cannot rank the two predictors. |
