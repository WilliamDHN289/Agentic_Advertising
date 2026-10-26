# Pre-registration: E5 validating the advertising unit (field experiment)

Draft for OSF Registries ("OSF Preregistration" template), SL-01. **Status: draft, not filed.**
It is filed before any field data exists, and it cites the SL-22 protocol version.

**Protocol version:** `e5_protocol@1`, the SL-22 protocol `features/storelab/docs/e5_protocol.md`
(sha256 below). The partner and the sample size (A10) are to be determined. The machine-readable
form is `features/storelab/docs/claims_register.yaml`, which governs where the two differ.

## Registered files

| File | sha256 |
|---|---|
| `features/storelab/docs/claims_register.yaml` | `4ae71145c6a07da7a0ef3cb52e00cc9bd2713f4e84a92baae635190e24b9e0bc` |
| `requirements/A1_formal_statements.md` | `be45dbfa81b0694693435f5cd2a32eedbf9370ccd24418c11250102b5dd278b5` |
| `requirements/A12_thresholds.md` | `801640ec726d0abc2efd81400fdf3562c0fdeb96c718ad9134e97c9c679600f8` |
| `requirements/A17_stipulated_parameters.md` | `c8b863e4866dd75679828def51b8aef628d0398c0c364791e0247f8d2c559e3c` |
| `requirements/paper.tex` | `e4bcf05266b5c835d8dc827e829a86d1856f23bb3f295766aae899c154d63295` |
| `features/storelab/docs/e5_protocol.md` (`e5_protocol@1`) | `5ae6bd185998f7f39a43dd448908e20534b1182a17a6ca441792bd425408b4f5` |

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

The qualifying event is fixed in the SL-22 protocol (§5: a confirmed selection or an add-to-cart
instruction within 24 hours). No PII enters the data (protocol §7, schema
`contracts/storelab/e5_event.schema.json`).

## 3. Sampling plan

The sample size comes from the SL-22 power design at the registered margin (protocol §9:
`analysis/e5.py` `h5a_required_n`, 80% TOST power at τ = 0 from the partner's baseline rates) and
is written here before filing.

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
- **Rejection tests.** A test with no registered margin is never *not supported*: a non-rejection
  is *inconclusive*. Since amendment 1 (A12 §6.1) no claim label rests on one. H1a is decided by
  the size of the violation, η, and the H1c heterogeneity part by the range of δ/σ across models,
  each against a smallest effect of interest of 0.25 logit. The triple odds test, Hausman–McFadden
  and the Wald χ² test of equal δ/σ are reported beside the labels.
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
- **Tolerances net of Monte Carlo error (A12 §6.2).** A simulation check whose criterion is a
  tolerance on a quantity estimated from Monte Carlo replications (bias, coverage, size, power,
  or a slope from simulated data) fails only if the estimate exceeds the tolerance significantly
  at familywise 5%. With m checks in one criterion and z = z_{1−0.05/m} (Bonferroni), a tolerance τ
  on |x| passes if |x̂| ≤ τ + z·SE_MC, and a band [a, b] around a rate passes if
  a − z·SE_MC ≤ ĉ ≤ b + z·SE_MC. Exact computations (solver outputs, closed forms) keep their
  numeric tolerances. Reports give the raw tolerance and the adjusted bound.

**Operating characteristics.** Every confirmatory test must have empirical size in [0.03, 0.07] at
α = 0.05 over 1,000 Monte Carlo null replications (SL-15, SL-22), judged net of Monte Carlo error
(A12 §6.2). A test that fails is replaced before this registration is filed (A12 §2). The SL-15
replacements are owner decisions recorded in A12 §6.4:
- the triple odds test uses a Monte Carlo p-value under the fitted conditional logit (B = 999),
  because the set-level permutation version is anti-conservative (size 0.105 against 0.048);
- Hausman–McFadden is reported only, on δ/σ with 1 df (the joint δ and β version had size
  0.21–0.25);
- separation is detected exactly by a linear program, and every cell is estimated with Firth's
  penalised likelihood and profile penalised-likelihood intervals.

Evidence is in `features/storelab/docs/analysis_validation.md` at the SL-15 approval (b9f8449).

**Deviations.** Deviations are logged as dated amendments with reasons. Thresholds, margins and
decision rules change only through a dated owner amendment made before registration, and every
amendment is listed here. Any analysis not in this document is labelled exploratory.

**Amendment 1 (owner, 2026-10-25; A12 §6).** Made before this registration is filed and before any
E1 data exists, on the evidence of the SL-15 validation study and the pre-E1 dry runs SL-19S to
SL-21S. It changes no threshold that E1 data could have informed:
- §6.1: equivalence margins of 0.25 logit for H1a (η rule) and for H1c heterogeneity (range rule),
  and the SL-17 power requirement for H1a;
- §6.2: tolerances net of Monte Carlo error;
- §6.3: the H4b slope rule;
- §6.4: the SL-15 test substitutions, recorded as owner decisions.

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
