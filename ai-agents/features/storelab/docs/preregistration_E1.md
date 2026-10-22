# Pre-registration: E1 agent response audit, with the E2–E4 decision rules

Draft for OSF Registries ("OSF Preregistration" template; public, or embargoed for at most 4 years),
SL-01. **Status: draft, not filed.** It is filed after SL-17 freezes the sample sizes and before
the first SL-18 trial. The URL and timestamp go in `docs/registrations.yaml`, and
`python -m features.storelab.analysis.registration check-e1 <SL-18 runs>` checks the order from the
run manifests (SL-18 acceptance).

It covers H1a–H4d:
- the tests of E1 (H1a–H1e);
- the decision rules of the E2–E4 simulations calibrated on E1 (H2a–H4d).

The machine-readable form of every claim is `features/storelab/docs/claims_register.yaml`. Where
this text and the register differ, the register governs. The register follows
`requirements/A1_formal_statements.md` (statements; corrections §E take precedence over tickets.md
§1.1) and `requirements/A12_thresholds.md` (venue, thresholds, rules, floors).

## Registered files

| File | sha256 |
|---|---|
| `features/storelab/docs/claims_register.yaml` | `83c41b23565aaf68a40b58a9a9b38a4f1fb0b91b2eb8615418e215a51da85592` |
| `requirements/A1_formal_statements.md` | `be45dbfa81b0694693435f5cd2a32eedbf9370ccd24418c11250102b5dd278b5` |
| `requirements/A12_thresholds.md` | `82217db6d087b78799db59aa680f0be70bfb26040d7acc06bbd533e9c3813f9d` |
| `requirements/paper.tex` | `e4bcf05266b5c835d8dc827e829a86d1856f23bb3f295766aae899c154d63295` |

At filing, the following are added: the analysis code commit, the E1 main config with its
`config_hash`, and the master seed. Pilot seed for reference: 2026102101.

## 1. Study information

The paper (`requirements/paper.tex`) claims three things:
- behavioural premises about LLM shopping agents: near-deterministic choice, set dependence,
  label penalties, and states shared across agents built on the same model (E1);
- mechanism predictions that must survive finite noise and calibrated heterogeneity (E2);
- predictions about operations (E3) and learning (E4).

Each hypothesis is stated and labelled in §6.

## 2. Design plan

**E1** is a sandbox audit of real agents:
- **Products:** Amazon Reviews 2023 item metadata, snapshot `d7750e73…`.
- **Tasks:** shopping tasks with explicit requirements, with eligible sets of v products.
- **Candidate sets:** K ∈ {2, 4, 8}, from a balanced incomplete block design or pair covering in
  which every pair co-occurs.
- **Labels and positions:** positions randomly permuted, sponsored labels with probability 0.3.
  Presentation is randomised per set (A15): positions and labels are drawn once per set and
  repeated identically r = 10 times.
- **Interfaces:** three, namely a text listing, structured data, and `browser_tool`. The last is
  a tool-driven storefront, not a GUI agent, so H1e conclusions apply to `browser_tool` only
  unless SL-10b runs.
- **Models:** at least three providers with at least two pinned versions each, at default and zero
  temperature.

**E2–E4** are offline simulations calibrated on the E1 estimates:
- E2: mechanisms M1, M2, set-blind M3, M3, M4 and the profit rule;
- E3: correlated demand, Part A two-leader closed forms and Part B calibrated categories;
- E4: audit designs, h, n₀ and state variance.

Common random numbers are used across mechanisms.

## 3. Sampling plan

The registered floors are minima: SL-17 may raise them, never lower them. Below a floor, a label is
inconclusive by rule.

| Claim | Floor |
|---|---|
| H1a | ≥ 20 informative triples per primary cell |
| H1b | ≥ 30 eligible sets per primary cell (a set needs ≥ 8 valid repeats of 10) |
| H1c | 80% power at δ/σ = −0.25 (SL-17 sizes for 80% power at max(0.25, \|pilot estimate\|)) |
| H1d | ≥ 20 tasks per provider version pair; alignment needs ≥ 10 credible changes, each (provider version pair, task) counted once |
| H1e | ≥ 200 non-chosen displayed products per browser cell |
| E2 | ≥ 200 valid markets per configuration |
| E4 | R = 200 replications per cell; the H4b implementation check uses 10,000 |

**Main design:** to be set by SL-17 and written here before filing. With about 10 tasks, H1d is
below its floor and is inconclusive by rule; the set-level change share is then the main
descriptive evidence for E3.

## 4. Variables and definitions

- **Ground-truth utility** (`shrunk_rating@1`): u(x, j) = (n_j r_j + m r̄_x) / (n_j + m). Here r_j
  is the product's average rating, n_j its rating count, r̄_x the mean rating over E(x), and
  m = 20.
- **Effective choice:**
  - text and structured: a reply that parses to a candidate id; a null choice is the outside
    option;
  - `browser_tool`: a `select(product_id)` tool call; `leave()` is the outside option; page
    requests are not choices.
- **Non-choices** (`invalid`, `refusal`, `error`, `not_run`): missing in the primary analysis,
  coded as the outside option in the sensitivity analysis.
- **Missingness per cell** (model version × interface × temperature):
  - up to 10%: complete-case;
  - 10–25%: the cell keeps its label only if the sensitivity analysis agrees;
  - above 25%: excluded and inconclusive.
- **Differential missingness:** if missingness depends on label or position, the cell's H1c label
  stands only if both codings agree.
- **E1 primary cells:** (model version × interface) at default temperature. Temperature-0 cells
  are secondary. Claim-level labels need at least 2/3 of primary cells supported; more than 1/3
  not supported gives not supported.
- **H1d alignment units:** each (provider version pair, task) is counted at most once, through
  the first interface in the fixed order text → structured → browser_tool that shows a credible
  change. A tie in u at that interface excludes the unit. The units are then pooled across
  providers. Counting a task once per interface would break the independence of the binomial
  test.
- **Symbols:** as defined in A1 §E7 (N_x = \|E(x)\|; n₀ per cell per period; χ_m per displayed
  product).
- **Profit-rule regularity (A16):** ironing of non-monotone virtual values; the value grid is not
  restricted.

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

**Resampling.** The bootstrap clusters over candidate sets (E1) and over markets or replications
(E2–E4), with B = 2,000 and percentile intervals. Seeds come from the master seed.

**Interpretation.**
- If H1a and H1b fail for a model or interface, separable approximations suffice for that traffic,
  and E2 reports results by interface.
- If version shifts align with u (H1d alignment), `prop:threshold` rather than `prop:separation`
  governs set design.
- The estimated parameters are released and calibrate E2–E4.

## 6. Hypotheses, metrics and decision rules

### H1a

The odds of choosing i over j change when a product preferred to both is added or removed (an IIA violation).

| | |
|---|---|
| Kind | P |
| Sources | `premise` |
| Experiment and mode | E1, sandbox |
| Unit of analysis | informative triple (i, j, k) within an E1 primary cell: sets containing i and j with and without k, where k is chosen over both |
| Metric | log odds ratio of i over j with versus without k (Haldane-corrected) |
| Estimator | pooled triple odds statistic over the cell's informative triples; Hausman-McFadden statistic from the cell's conditional logit with and without k |
| Test | triple odds permutation test (primary, decides the label); Hausman-McFadden test (reported) |
| alpha / CI level | 0.05 / 0.95 |
| Multiplicity family | H1a: the triple odds tests of all E1 primary cells (Holm) |
| Tolerance or margin | none: a rejection test (no equivalence margin is registered) |
| Minimum N | 20 informative triples per primary cell |
| Decision rule | Cell: supported if the Holm-adjusted p-value of the triple odds permutation test is <= 0.05 (IIA rejected); otherwise inconclusive, since a non-rejection without a registered margin is not evidence for IIA. Claim: two-thirds aggregation over primary cells. |
| If supported | Agents are set-dependent as the model assumes: separable approximations do not suffice for this traffic, and the gaps of Section 4 apply. |
| If not supported | Not reachable by this rule (no equivalence margin is registered); see inconclusive. |
| If inconclusive | If H1a and H1b fail for a model or interface, separable approximations suffice for that traffic and the Section 4 gaps should shrink; E2 then reports results by interface (paper E1 Interpretation). |

### H1b

Repeated identical presentations give the same choice in most trials (mean modal share >= 0.80), and score gaps are large relative to sigma (median top-two gap >= ln 4).

| | |
|---|---|
| Kind | P |
| Sources | `lem:small-noise`, `ass:deterministic` |
| Experiment and mode | E1, sandbox |
| Unit of analysis | candidate set presented r = 10 times identically (A15 per_set) within an E1 primary cell; a set needs >= 8 valid repeats |
| Metric | (i) mean over eligible sets of the modal share m_s = max_j n_sj / r_valid (j over displayed products and the outside option); (ii) median over sets of the gap G_s in sigma units between the highest and second-highest label-adjusted scores in S u {0} |
| Estimator | (i) sample mean with a cluster bootstrap over sets; (ii) gaps from the cell's conditional-logit fit (penalised under separation, a lower bound), bootstrap median |
| Test | one-sided comparisons of the 95% bootstrap interval bounds with 0.80 and ln 4 |
| alpha / CI level | 0.05 / 0.95 |
| Multiplicity family | H1b part (i) across primary cells; H1b part (ii) across primary cells (Holm, separately) |
| Tolerance or margin | thresholds 0.80 (modal share) and ln 4 = 1.386 (gap) |
| Minimum N | 30 eligible sets per primary cell |
| Decision rule | Cell: supported if the Holm-adjusted lower bounds of both parts are at or above their thresholds; not supported if the Holm-adjusted upper bound of either part is below its threshold; inconclusive otherwise. Claim: two-thirds aggregation. |
| If supported | Agents are near-deterministic in the sense the deterministic-limit results use; the mechanism results of Section 4 are a reasonable first-order description for this traffic. |
| If not supported | Choice noise is large: with H1a failing too, separable approximations suffice for that traffic, the Section 4 gaps should shrink, and E2 reports results by interface. |
| If inconclusive | The data cannot place consistency or gaps relative to the thresholds; E2 calibrates on the estimated sigma as is and reports by interface. |

### H1c

The label effect delta_m / sigma_m is negative, and it differs across models.

| | |
|---|---|
| Kind | P |
| Sources | `premise` |
| Experiment and mode | E1, sandbox |
| Unit of analysis | E1 primary cell (sign); model (heterogeneity) |
| Metric | delta_m / sigma_m from the conditional logit with product fixed effects, label and position effects |
| Estimator | conditional-logit MLE (penalised under separation), cluster bootstrap over sets |
| Test | sign: one-sided Wald test of delta/sigma < 0 per primary cell; heterogeneity: Wald chi-square test of equal delta/sigma across models |
| alpha / CI level | 0.05 / 0.95 |
| Multiplicity family | H1c sign: all primary cells (Holm); heterogeneity: one claim-level test |
| Tolerance or margin | threshold 0 (sign); minimum effect of interest for power delta/sigma = -0.25 (SL-17 sizes for 80% power at max(0.25, \|pilot estimate\|)) |
| Minimum N | 80% power at the minimum effect of interest, per primary cell |
| Decision rule | Sign cell: supported if the Holm-adjusted one-sided 95% upper bound is < 0; not supported if the lower bound is > 0; two-thirds aggregation. Heterogeneity: supported if equality is rejected at 0.05, else inconclusive. Claim: supported if both parts are supported; not supported if either is. |
| If supported | Labels penalise sponsored products and the penalty is model-specific, so E2 uses model-specific delta_m/sigma_m and the labelled-favourite screen of A1 §E5 matters. |
| If not supported | Labels do not lower agents' choice odds (or raise them): position-rent arguments that rely on a label penalty do not apply to that traffic. |
| If inconclusive | E2 uses the estimated delta_m/sigma_m with its uncertainty and reports both the estimate and a zero-penalty arm. |

### H1d

Version changes within a provider reorder favourites for a non-negligible share of tasks (>= 0.10); alignment of the reorderings with the ground-truth utility u is tested.

| | |
|---|---|
| Kind | P |
| Sources | `premise` |
| Experiment and mode | E1, sandbox |
| Unit of analysis | task x, for one provider's version pair (A, B), within one interface at default temperature; alignment: (provider version pair, task), counted at most once across interfaces |
| Metric | share of tasks whose favourite f_v(x) = argmax over E(x) u {0} of unlabeled v_j/sigma differs between versions: raw changes, and credible changes (different in >= 90% of bootstrap replicates resampling sets within the task) |
| Estimator | per-version conditional-logit scores; task-level bootstrap for credible changes |
| Test | exact Clopper-Pearson bounds, one-sided 95%, tasks as units; alignment: exact binomial test against 0.5 |
| alpha / CI level | 0.05 / 0.95 |
| Multiplicity family | H1d: all (provider version pair x interface) cells (Holm) |
| Tolerance or margin | threshold 0.10 (share of tasks); alignment thresholds 0.5 (aligned) and 0.65 (not aligned) |
| Minimum N | 20 tasks per provider version pair; alignment needs 10 credible changes, each (provider version pair, task) counted once |
| Decision rule | Cell: supported if the Clopper-Pearson one-sided 95% lower bound of the credible-change share is > 0.10; not supported if the upper bound of the raw-change share is < 0.10; inconclusive otherwise; two-thirds aggregation. Alignment (secondary): each (provider version pair, task) is counted at most once, through the first interface in the fixed order text, structured, browser_tool that shows a credible change (a tie in u at that interface excludes the unit); the units are then pooled across providers. Aligned if the Clopper-Pearson lower bound of the share whose new favourite has higher u is > 0.5; not aligned if the upper bound is < 0.65; inconclusive otherwise or with fewer than 10 such credible changes. Counting a task once per interface would break the independence of the binomial test. |
| If supported | Version changes are a common shock across agents of one provider: states matter for set design (E3) at any realistic volume (A12 §3.2). If the changes are aligned with u, the state carries quality information and prop:threshold, not prop:separation, governs set design. |
| If not supported | Version changes rarely move favourites; common-state exposure from versions is small and E3 Part B is reported as a sensitivity analysis. |
| If inconclusive | The set-level change share becomes the main descriptive evidence for E3 (A12 §5 note: with fewer than 20 tasks H1d is inconclusive by rule). |

### H1e

Browser-based agents request pages of products they do not choose (rate >= 0.10).

| | |
|---|---|
| Kind | P |
| Sources | `ass:retrieval` |
| Experiment and mode | E1, sandbox (browser_tool; GUI agents only if SL-10b runs) |
| Unit of analysis | non-chosen displayed product in a browser_tool trial, within an E1 primary browser cell |
| Metric | page-request rate for non-chosen displayed products (estimates chi_m for E2) |
| Estimator | pooled proportion over trials |
| Test | Wilson 95% interval |
| alpha / CI level | 0.05 / 0.95 |
| Multiplicity family | H1e: all primary browser_tool cells (Holm) |
| Tolerance or margin | threshold 0.10 |
| Minimum N | 200 non-chosen displayed products per browser cell |
| Decision rule | Cell: supported if the Holm-adjusted Wilson 95% lower bound is >= 0.10; not supported if the upper bound is < 0.10; inconclusive otherwise. Claim: two-thirds aggregation over browser cells. |
| Amendment | A2 / A1 §E7: chi_m is the probability of requesting each displayed product's page, independent of the choice; H1e estimates it from non-chosen products |
| If supported | Browser agents generate page requests that are not choices, so click-based charging (M1) bills agent traffic for inclusion, not choice (lem:click); E2 uses the estimated chi_m. |
| If not supported | Page requests are rare among non-chosen products; M1's agent charges track choices, and E2 sets chi_m near 0 for that agent. |
| If inconclusive | E2 varies chi_m over {0, 0.5, 1} without an E1 anchor for that model. Conclusions apply to browser_tool only unless SL-10b runs. |

### H2a

Under agent-aware organic policies, agent-traffic payments under M3 are 0 (equivalent within the margin) and total payments go to 0 as alpha -> 1; under human-optimised organic policies agent-traffic payments rise with the excess e(x) = |F(x)| - (K - K^s).

| | |
|---|---|
| Kind | Pr |
| Sources | `prop:rents`, `cor:aware` |
| Experiment and mode | E2, offline, calibrated on E1 |
| Unit of analysis | simulated market (>= 200 per configuration) |
| Metric | psi_A = sum_j P^A_j / (n E_mu[alpha] thetabar); psi_tot = sum_j P_j / (n thetabar) |
| Estimator | mean over markets with a bootstrap over markets; OLS slopes with bootstrap intervals |
| Test | part 1: TOST of mean psi_A within (-0.01, 0.01); part 2: one-sided slope of psi_tot on alpha < 0 and TOST of psi_tot(alpha=1) within 0.01; part 3: one-sided slope of psi_A on e(x) > 0 |
| alpha / CI level | 0.05 / 0.95 |
| Multiplicity family | H2a part 1: the (alpha in {0.25, 0.5, 0.75, 1} x organic variation) configurations (Holm) |
| Tolerance or margin | equivalence margin 0.01 in psi units |
| Minimum N | 200 markets passing the A1 §E5 screen per configuration |
| Decision rule | Part 1 per configuration: supported if the 90% interval of mean psi_A lies in (-0.01, 0.01); not supported if the 95% interval lies entirely outside [-0.01, 0.01]; all configurations must be supported. Part 2: supported if the slope's 95% upper bound is < 0 and psi_tot(alpha=1) is equivalent to 0 within 0.01. Part 3: supported if the slope's 95% lower bound is > 0. H2a: supported if all three parts are supported, not supported if any is not supported, inconclusive otherwise. |
| Amendment | A1 §E5: payments split by traffic type, P^A_j = b_j X^A_j - int_0^{b_j} X^A_j dz; markets must pass the labelled-favourite screen, failing markets reported separately |
| If supported | Inclusion rents, not position rents, drive revenue from agent traffic once noise is finite and agent types are heterogeneous (paper E2 Interpretation). |
| If not supported | Agent traffic pays position rents under agent-aware policies, or payments do not track excess favourites: the deterministic-limit mechanism argument does not survive calibrated noise. |
| If inconclusive | The simulation cannot separate the payments from the margin; reported with the screened-out boundary markets. |

### H2b

In a monoculture, gating raises profit by n xi E[min{theta_f, theta_r}] and lowers match quality by the amount in prop:gating(b) (exact); with human traffic added, the profit gain from gating is nondecreasing in alpha.

| | |
|---|---|
| Kind | I/Pr |
| Sources | `prop:gating`, `conj:gating` |
| Experiment and mode | E2, offline |
| Unit of analysis | simulated market (I-part); configuration alpha in {0, 0.25, 0.5, 0.75, 1} (Pr-part) |
| Metric | Pi_b - Pi_a against n xi min{theta_f, theta_r}; adjacent increments of DeltaPi(alpha) |
| Estimator | per-market solver output; mean increments with bootstrap intervals over markets |
| Test | I-part: tolerance checks; Pr-part: Holm-adjusted 95% bounds of the four adjacent increments against -tau |
| alpha / CI level | 0.05 / 0.95 |
| Multiplicity family | H2b Pr-part: the four adjacent increments (Holm) |
| Tolerance or margin | I-part: relative error <= 1e-6 per market; \|mean - n xi E min\| <= 3 Monte Carlo SE. Pr-part: tau = 1% of DeltaPi(1) |
| Minimum N | 200 markets per configuration |
| Decision rule | I-part: pass if every market's relative error is <= 1e-6 and the mean is within 3 MC SE of the closed form; a failure is an implementation error, not a finding. Pr-part: supported if every increment's Holm-adjusted 95% lower bound is >= -tau; not supported if some increment's upper bound is < -tau. |
| Closed form | Pi_b - Pi_a = n xi min{theta_f, theta_r}; M_a - M_b = n xi (u(f) - u(r)) 1{theta_f < theta_r} |
| Closed form | E[Pi_b - Pi_a] = n xi int_0^inf (1 - F_f(t))(1 - F_r(t)) dt; uniform U[0, thetabar]: E min = thetabar / 3; lognormal: E min = 2 exp(mu + s^2/2) Phi(-s / sqrt 2) |
| If supported | Gating sells inclusion to agent traffic at the predicted price, and adding human traffic does not erode the gain (conj:gating holds numerically). |
| If not supported | With human traffic the gain from gating falls with alpha at some step: the conjecture fails in the calibrated environment. |
| If inconclusive | The increments cannot be signed within tau; or the closed-form check failed (implementation error) and reporting is blocked. |

### H2c

M2 revenue is proportional to E_mu[1 - alpha(x)]; M1 payment per effective choice rises with alpha chibar; the set-blind gap V^aw - V^bl is 0 with fixed organic lists (exact) and grows with organic variation.

| | |
|---|---|
| Kind | I/Pr |
| Sources | `lem:click`, `prop:setblind` |
| Experiment and mode | E2, offline |
| Unit of analysis | simulated market; regression across configurations |
| Metric | M2 revenue on E[1 - alpha]; M1 payment per effective choice on alpha chibar; V^aw - V^bl |
| Estimator | OLS slopes and intercepts with bootstrap intervals over markets; solver values |
| Test | one-sided slope signs; TOST of the M2 intercept within 0.01 psi; I-check of the gap with deterministic organic lists; one-sided slope of the gap on organic variation |
| alpha / CI level | 0.05 / 0.95 |
| Multiplicity family | none within H2c (each part is one test) |
| Tolerance or margin | intercept margin 0.01 (psi units); gap I-check \|V^aw - V^bl\| <= 1e-6 relative |
| Minimum N | 200 markets per configuration |
| Decision rule | M2: supported if the slope on E[1 - alpha] is positive (one-sided 95% lower bound > 0) and the intercept is equivalent to 0 within 0.01; M1: supported if the slope on alpha chibar is positive; set-blind: the deterministic-list gap must pass the I-check (a positive gap blocks reporting, SL-19) and the slope on organic variation must have a 95% lower bound > 0. All parts combine. |
| Closed form | V^aw(b) = n sum_x mu(x) E_R[max_S U_b] >= V^bl(b) = n sum_x mu(x) max_S E_R[U_b], with equality when Q^o(. \| x) is deterministic for every x |
| Closed form | hand-solved instance (prop:setblind(b)): (V^bl - V^0) / (V^aw - V^0) = 1 / H_{N_a} |
| Amendment | A1 index: prop:setblind(a) is registered as I-type (the E2 implementation gate), lem:click as Pr; tickets.md §1.1 lists H2c as Pr |
| If supported | Click benchmarks misprice agent traffic as lem:click predicts, and set awareness is worth something only when organic lists vary. |
| If not supported | A failure of the M1 prediction under browser agents points to page requests that track choices, which ass:retrieval rules out (paper E2 Interpretation). |
| If inconclusive | A positive set-awareness gap with deterministic organic lists would contradict prop:setblind and indicates an implementation or calibration error: reporting is blocked until root-caused. |

### H3a

The separated policy matches expected loads, contributions and match quality with no across-state variation (exact); composition-blind randomisation keeps exposure near the lem:contested bound.

| | |
|---|---|
| Kind | I/Pr |
| Sources | `lem:contested`, `prop:separation` |
| Experiment and mode | E3, offline, calibrated on E1 |
| Unit of analysis | calibrated category (Part B); leader product |
| Metric | I: relative differences of loads, contributions and match quality; across-state variance. Pr: r = Var_Lambda(p_j) / (Theta_j^2 / 4) for leaders under composition-blind randomisation |
| Estimator | solver output; bootstrap over replications for r |
| Test | I: tolerance checks; Pr: Holm-adjusted 95% lower bound of r against 0.5 per category |
| alpha / CI level | 0.05 / 0.95 |
| Multiplicity family | H3a Pr-part: calibrated categories (Holm) |
| Tolerance or margin | I: equal within 1e-9 relative, across-state variance <= 1e-12; Pr: threshold r = 0.5 |
| Minimum N | R = 200 replications per category |
| Decision rule | I-part passes or is an implementation error. Pr-part per category: supported if the lower bound of r is >= 0.5, not supported if the upper bound is < 0.5; supported in at least 2/3 of calibrated categories (two-thirds aggregation). |
| Closed form | Var_Lambda(p_j^Q) <= Theta_j(Q)^2 / 4; two states (w, 1 - w) flipping together: Var = w (1 - w) Theta_j^2, ratio 4 w (1 - w) |
| Closed form | Example diversify, n = 1000: Var(D_A) = 250,000; 250; 62,625 |
| Amendment | A1 §E4: the bound Theta_j^2/4 is attained only with two equally likely state outcomes; E3 Part B states are equally likely model versions, so the attainable ratio is 1 |
| If supported | If separated policies lower cost without lowering match quality in calibrated environments, prop:separation applies in practice; composition-blind randomisation does not remove common-shock exposure. |
| If not supported | Composition-blind randomisation removes most exposure in calibrated categories: the bound is loose there. |
| If inconclusive | If separated policies lower match quality, the calibrated states carry quality information and the relevant trade-off is prop:threshold (paper E3 Interpretation). |

### H3b

When states carry quality information, the optimal head-to-head share follows the closed forms (E3 Part A).

| | |
|---|---|
| Kind | I |
| Sources | `prop:threshold` |
| Experiment and mode | E3, offline |
| Unit of analysis | grid point of (Delta, lambda, omega, n) in the two-leader environment |
| Metric | \|y - y*\|, objective relative error, finite-n excess f(y*) - f(y_n) - 2 lambda / sqrt n |
| Estimator | convex solver |
| Test | tolerance checks at every grid point |
| alpha / CI level | 0.05 / 0.95 |
| Multiplicity family | none (implementation check) |
| Tolerance or margin | quadratic: \|y - y*\| <= 1e-4 and objective relative error <= 1e-6; overflow limit: objective relative error <= 1e-6; finite n: f(y*) - f(y_n) <= 2 lambda / sqrt n + 1e-6 |
| Minimum N | 100% of grid points |
| Decision rule | Supported if every grid point passes; otherwise an implementation error that blocks E3 Part B (SL-20). |
| Closed form | quadratic: y* = min{1, Delta B / (lambda (n - 1))} |
| Closed form | overflow: y* = 1 if Delta > lambda or omega >= 1; y* = \|2 omega - 1\| if Delta < lambda and omega < 1 |
| Amendment | A1 §E6: quadratic case compares y and the objective; overflow compares objective values, y only away from knife-edges; the grid excludes Delta in {0, lambda} |
| If supported | The numerical solver reproduces prop:threshold; E3 Part B may run. |
| If not supported | Not reachable: an exact result failing is an implementation error. |
| If inconclusive | Implementation error: Part B is blocked until root-caused. |

### H3c

With sponsorship, leaders buy back sponsored inclusion under M3, and under M4 only when their bids exceed the congestion term.

| | |
|---|---|
| Kind | Pr |
| Sources | `cor:congestion` |
| Experiment and mode | E3, offline |
| Unit of analysis | (market, leader) pair |
| Metric | M3: share of leaders sponsored back into lists the organic policy removed them from; M4: share of pairs where buy-back coincides with b_j DeltaX_j exceeding the remaining terms of cor:congestion |
| Estimator | solver output, term by term |
| Test | M3: 95% lower bound of the buy-back share > 0; M4: coincidence share >= 0.95 (pairs within 1e-6 of indifference excluded) |
| alpha / CI level | 0.05 / 0.95 |
| Multiplicity family | none (one test per mechanism) |
| Tolerance or margin | M4 coincidence >= 0.95; indifference band 1e-6 |
| Minimum N | 200 markets per configuration |
| Decision rule | Supported if the M3 buy-back share's Clopper-Pearson 95% lower bound is > 0 and the M4 coincidence share is >= 0.95; not supported if the M4 share is < 0.95. The M3 part can only be supported or inconclusive: no interval of a share lies below 0 and no other threshold is registered. |
| If supported | Under M3 leaders buy back the concentration that organic diversification removed; M4 charges them for it (paper l.674). |
| If not supported | Buy-back is absent under M3 or does not follow the congestion term under M4. |
| If inconclusive | The buy-back share cannot be separated from zero. |

### H4a

The identified share of target sets is determined by reachability in G_A, not by inclusion frequency.

| | |
|---|---|
| Kind | I/Pr |
| Sources | `prop:identify`, `cor:design` |
| Experiment and mode | E4, offline |
| Unit of analysis | audit design x target set, within replications |
| Metric | identified share; competitor-varying minus leader-inclusive at matched inclusion frequency |
| Estimator | reachability prediction (exact); paired difference over replications with a bootstrap |
| Test | I: exact equality; Pr: paired difference 95% lower bound > 0 |
| alpha / CI level | 0.05 / 0.95 |
| Multiplicity family | none (one comparison) |
| Tolerance or margin | I: difference 0 exactly |
| Minimum N | R = 200 replications per cell |
| Decision rule | I-part: the identified share equals the reachability prediction exactly, else implementation error. Pr-part: supported if the paired difference's 95% lower bound is > 0; not supported if the upper bound is < 0. |
| Closed form | identified iff some alternative in S' u {0} reaches every other in G_A (prop:identify) |
| Closed form | audited sets >= log A(N_x) / log(K + 1), A(N_x) = floor(e N_x!) |
| Amendment | A1 §E1: cor:design(b) lower bound uses A(N_x) = floor(e N_x!) choice functions, not (N_x + 1)! |
| If supported | Exploration must vary competitors; including the leader often teaches nothing about choices against it. |
| If not supported | Inclusion frequency identifies as much as comparison structure in the calibrated designs. |
| If inconclusive | The designs cannot be separated at R = 200. |

### H4b

Estimation error falls like h^(-1/2) (log-log RMSE slope in [-0.55, -0.45]); in n_0 it follows the closed form RMSE(n_0)/RMSE(inf) = sqrt(1 + E[q(1-q)] / (n_0 Var_Lambda q)), falling to a floor when Var_Lambda(q) > 0.

| | |
|---|---|
| Kind | I/Pr |
| Sources | `prop:learning(a)` |
| Experiment and mode | E4, offline |
| Unit of analysis | cell (h, n_0, state variance, N_x), replications |
| Metric | bias and variance of q-hat against the closed form; log-log slope of RMSE on h |
| Estimator | Monte Carlo over replications (10,000 per checked cell for the I-check) |
| Test | I: \|bias\| <= 3 MC SE and variance ratio within +-5%; Pr: 95% interval of the slope within [-0.55, -0.45] |
| alpha / CI level | 0.05 / 0.95 |
| Multiplicity family | none (one slope) |
| Tolerance or margin | I: 3 MC SE, +-5% variance ratio; slope margin 0.05 around -0.5 |
| Minimum N | 10,000 replications per checked cell (I-check); R = 200 replications per cell (slope) |
| Decision rule | I-part passes or is an implementation error. Slope: supported if the 95% interval lies within [-0.55, -0.45]; not supported if it lies entirely outside; inconclusive otherwise. |
| Closed form | Var q-hat_j(c) = (1/h) {Var_Lambda q_j(c \| Lambda) + (1/n_0) E_Lambda[q_j (1 - q_j)]}; RMSE = sqrt(Var) since q-hat is unbiased |
| Amendment | A1 §E3: RMSE in n_0 is not flat; 'flat beyond small values' means within 5% of the floor, which holds once n_0 >= 10 E[q(1-q)] / Var_Lambda q; H4b is checked against the closed form, not against 'flat' |
| If supported | Learning is limited by periods, not requests: more audit requests per period do not help beyond the floor when states vary. |
| If not supported | A failure in the direction of errors that keep falling with n_0 when Var_Lambda(q) > 0 would indicate that the calibrated states are less persistent than assumed (paper E4 Interpretation). |
| If inconclusive | The slope cannot be placed within the margin; or the closed-form check failed (implementation error). |

### H4c

The largest gain from misreporting, by grid search, is below 2 theta_j n epsilon_1 (at rho = 0.05) and declines with h.

| | |
|---|---|
| Kind | Pr |
| Sources | `prop:learning(d)` |
| Experiment and mode | E4, offline |
| Unit of analysis | (replication, advertiser j) grid search |
| Metric | share of searches whose best misreport gain exceeds 2 theta_j n epsilon_1; slope of the maximum gain on log h |
| Estimator | proportion over searches; OLS slope with a bootstrap |
| Test | Clopper-Pearson 95% upper bound of the violation share against 3 rho = 0.15; one-sided slope sign |
| alpha / CI level | 0.05 / 0.95 |
| Multiplicity family | none (one test per part) |
| Tolerance or margin | violation share 0.15 = 3 rho |
| Minimum N | R = 200 replications per cell |
| Decision rule | Supported if the Clopper-Pearson 95% upper bound of the violation share is <= 0.15 and the slope on log h has a 95% upper bound < 0; not supported if the lower bound of the violation share is > 0.15 or the slope's lower bound is > 0. |
| If supported | The plug-in mechanism is approximately incentive compatible at the bound's rate. |
| If not supported | Misreporting gains exceed the bound more often than the failure probability allows, or do not decline with audit length. |
| If inconclusive | The grid search cannot place the violation share relative to 0.15. |

### H4d

The growth exponent of cumulative loss in h does not exceed the bound's (log-log slope <= 0.6); bounded loss (slope near 0) is consistent with the theory.

| | |
|---|---|
| Kind | Pr |
| Sources | `cor:cumulative` |
| Experiment and mode | E4, offline |
| Unit of analysis | replication, h in {5, 10, 20, 50, 100} |
| Metric | log-log slope of cumulative loss on h |
| Estimator | OLS on log means with a bootstrap over replications |
| Test | 95% interval bounds against 0.6 |
| alpha / CI level | 0.05 / 0.95 |
| Multiplicity family | none (one slope) |
| Tolerance or margin | threshold 0.6 |
| Minimum N | R = 200 replications per cell |
| Decision rule | Supported if the 95% upper bound is <= 0.6; not supported if the 95% lower bound is > 0.6; inconclusive otherwise. |
| Amendment | A1 §E2: cor:cumulative is an upper bound of order sqrt(h log h), not a growth rate; tickets.md §1.1 'cumulative loss proportional to sqrt h' is replaced |
| If supported | Cumulative loss grows no faster than the bound allows. |
| If not supported | Loss grows faster than the bound's exponent: the calibrated environment violates the corollary's conditions or the implementation is wrong. |
| If inconclusive | The exponent cannot be placed relative to 0.6. |
