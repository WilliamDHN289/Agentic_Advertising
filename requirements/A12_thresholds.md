# A12 · Pre-registration venue and decision thresholds

Owner input to SL-01 (tickets.md §5, item A12). It also fixes the other numbers SL-01 must register: tolerances for exact results, decision rules and minimum sample sizes.

| | |
|---|---|
| Decided by | Project owner, 2026-10-22 |
| Status | Fixed before any E1 data exists. The SL-17 pilot may change **sample sizes only** (it writes them into SL-01 before SL-18). Thresholds, margins and decision rules here change only through a dated owner amendment made before the E1 registration, and every amendment is listed in the registration. |
| Companion | `A1_formal_statements.md` (statements, closed forms, corrections §E). |

---

## 1. Venue and timing

| Item | Decision |
|---|---|
| Venue | OSF Registries, "OSF Preregistration" template. Public, or embargoed for at most 4 years. |
| E1 registration | One registration covering H1a–H4d: E1's tests, and the decision rules of the E2–E4 simulations calibrated on E1. Filed after SL-17 freezes the sample sizes and before the first SL-18 trial. SL-18 checks the timestamp against the manifests. |
| E5 registration | Separate, covering H5a–H5b. Filed before any field data exists. Cites the SL-22 protocol version. |
| Contents | The claims register (YAML and schema), the pre-registration document, the analysis code commit, the master seed, and this file and A1 by sha256. |
| Deviations | Logged as dated amendments with reasons. Any analysis not in the registration is labelled exploratory. |
| Receipts | OSF URL and registration timestamp stored with the SL-01 docs. |

---

## 2. Conventions that apply to every claim

**Significance and intervals.**

- α = 0.05. Tests are two-sided unless the claim is directional: the sign part of H1c and the slope signs named below are one-sided.
- Confidence intervals are 95%.
- Equivalence (TOST) uses the 90% interval, which equals two one-sided 5% tests.

**Three-way decision rule.** Each claim, or each claim within a cell, gets one label by comparing its interval with its registered threshold:

| Label | Condition |
|---|---|
| **supported** | The interval lies entirely on the claim's side of the threshold, or inside the equivalence margin. |
| **not supported** | The interval lies entirely on the other side, or entirely outside the equivalence margin. |
| **inconclusive** | Otherwise. Also used whenever the minimum sample size (§5) is not met or a cell is excluded for missingness (§3.4). |

**Multiplicity.**

- Holm–Bonferroni within each family. A family is the set of confirmatory cell-level tests of one hypothesis ID; for example, H1c's sign test across all primary cells is one family.
- There is no correction across hypothesis IDs, because each claim is read on its own.
- Intervals used for decisions are Holm-adjusted to match: within a family of size $k$, the $i$-th ordered test uses level $\alpha/(k-i+1)$.

**Resampling.** Cluster bootstrap over candidate sets for E1, and over markets or replications for E2–E4. B = 2,000, percentile intervals. Seeds are derived from the registered master seed.

**E1 primary cells.**

- Primary cells are (model version × interface) at default temperature. Temperature 0 cells are secondary and reported, but they do not enter the claim-level label.
- Cell-level labels govern how E2–E4 are calibrated by interface, as in the paper's E1 Interpretation.

**Claim-level aggregation for E1 claims** (applied over primary cells):

| Claim-level label | Condition |
|---|---|
| supported | supported in at least 2/3 of primary cells |
| not supported | not supported in more than 1/3 of primary cells |
| inconclusive | otherwise |

**Operating characteristics.** Every confirmatory test must have empirical size in [0.03, 0.07] at α = 0.05 over 1,000 Monte Carlo null replications (SL-15, SL-22). A test that fails is replaced by its permutation version before registration.

**Non-choices.** `invalid`, `refusal`, `error` and `not_run` are missing in the primary analysis. In the sensitivity analysis they are coded as the outside option. See §3.4.

---

## 3. The four A12 thresholds

### 3.1 H1b, "most trials": modal-choice share ≥ 0.80

**Data.**

- Unit: a candidate set $s$, presented r = 10 times identically within a cell (A15 `per_set`).
- A set is eligible if it has at least 8 valid repeats; ineligible sets are counted and reported.
- Modal share $m_s=\max_j n_{s,j}/r^{\mathrm{valid}}_s$, where $j$ runs over the displayed products and the outside option. An explicit "none" is a choice; missing repeats are excluded.

**Part (i), consistency.**

- Statistic: $\bar M_c$, the mean of $m_s$ over the eligible sets in cell $c$.
- Threshold: **0.80**.

**Part (ii), score gaps.**

- Statistic: for each set, the gap $\hat G_s$ in σ units between the highest and second-highest label-adjusted scores in $S\cup\{0\}$, taken from the cell's conditional-logit fit. The cell statistic is the median of $\hat G_s$ over sets.
- Threshold: **ln 4 ≈ 1.386**. This is the gap at which a binary logit picks the better option with probability 0.80, so parts (i) and (ii) use the same scale.
- Under separation (SL-15 penalised estimates), the penalised gap counts as a lower bound.

**Cell decision.**

| Label | Condition |
|---|---|
| supported | The Holm-adjusted lower bounds of both parts are at or above their thresholds. |
| not supported | The Holm-adjusted upper bound of either part is below its threshold. |
| inconclusive | Otherwise. |

**Why 0.80.** The paper's mechanism results hold in the deterministic limit. 0.80 is the lowest consistency at which treating a model as near-deterministic is still a reasonable first-order description: 8 of 10 identical presentations give the same choice. A majority threshold such as 0.5 would call a model "near-deterministic" when it switches choices in 4 of 10 repeats, which is too weak for the claim.

### 3.2 H1d, "non-negligible": favourite-change share ≥ 0.10

**Data.**

- Unit: a task (context) $x$, for one provider with versions A and B, within one interface at default temperature.
- Favourite $f_v(x)=\arg\max_{j\in E(x)\cup\{0\}}\hat v_j/\sigma$, using unlabeled scores from version $v$'s fit.
- A task has a **raw change** if $f_A(x)\ne f_B(x)$. It has a **credible change** if $f_A(x)\ne f_B(x)$ in at least 90% of the bootstrap replicates (resampling candidate sets within the task).

**Test.**

- Threshold: share **0.10**.
- Exact Clopper–Pearson intervals with tasks as units, one-sided 95%.

| Label | Condition |
|---|---|
| supported | Lower bound of the credible-change share > 0.10 |
| not supported | Upper bound of the raw-change share < 0.10 |
| inconclusive | Otherwise |

Using credible changes for "supported" and raw changes for "not supported" keeps both errors conservative. A near-tie that flips by estimation noise cannot create support, and the conservative count cannot create a rejection.

**Alignment, which decides whether `prop:separation` or `prop:threshold` governs E3.**

- Among credible changes, count tasks where the new favourite has higher ground-truth $u$; ties in $u$ are excluded. Use an exact binomial test against 0.5.
- At least 10 credible changes are needed; otherwise the alignment label is inconclusive.

| Alignment label | Condition | E3 consequence |
|---|---|---|
| aligned | Clopper–Pearson lower bound > 0.5 | `prop:threshold` governs |
| not aligned | Upper bound < 0.65 | `prop:separation` governs |
| inconclusive | Otherwise | E3 Part B reports both state structures (it does so anyway) |

**Reported with no threshold:** Spearman ρ and Kendall τ of $\hat v/\sigma$ between versions, and the share of candidate sets whose modal choice changes. The latter is the empirical contested mass that feeds $\Theta_j$ in E3.

**Why 0.10.** By `lem:contested`, a contested mass $\Theta$ adds up to $n^2\Theta^2/4$ to the load variance, against at most $n/4$ within-state variance. Once $n\ge1/\Theta^2$, which is 100 requests at $\Theta=0.1$, the common shock can dominate. A 10% share of tasks therefore matters at any realistic platform volume, while smaller shares can plausibly come from noise in favourite estimation.

### 3.3 H2a equivalence margin: 0.01 of mean value per agent request

**Attribution.** Payments are split by traffic type as in A1 §E5: $P^A_j=b_jX^A_j-\int_0^{b_j}X^A_j\,dz$.

**Statistic.** The normalised agent-payment rate:
$$\psi_A=\frac{\sum_jP^A_j}{n\,\mathbb E_\mu[\alpha(x)]\,\bar\theta},$$
where $\bar\theta$ is the analytic mean of the market's value distribution. It is the agent-traffic payment per agent request, in units of the mean advertiser value of one effective choice.

**Margin:** $\varepsilon_A=0.01$.

**Part 1: zero agent payments under agent-aware policies.**

- Agent-aware regime, for each $\alpha\in\{0.25,0.5,0.75,1\}$ and each organic-variation level, over at least 200 markets.
- Holm across these tests.
- Markets must pass the A1 §E5 screen. Markets that fail it are reported separately.

| Label | Condition |
|---|---|
| supported | 90% interval of mean $\psi_A$ ⊂ (−0.01, 0.01) |
| not supported | 95% interval lies entirely outside [−0.01, 0.01] |
| inconclusive | Otherwise |

**Part 2: total payments fall toward zero as α → 1.**

- Same regime. $\psi_{\mathrm{tot}}=\sum_jP_j/(n\bar\theta)$.
- Supported if both:
  - the slope of $\psi_{\mathrm{tot}}$ on $\alpha$ is negative (95% upper bound < 0);
  - $\psi_{\mathrm{tot}}(\alpha=1)$ is equivalent to 0 within ±0.01.

**Part 3: agent payments rise with excess favourites under human-optimised policies.**

- Regime: human-optimised. Supported if the slope of $\psi_A$ on $e(x)=|F(x)|-(K-K^s)$ is positive (95% lower bound > 0), across the levels for the number of models and versions.

**H2a label.** Supported if all three parts are supported; not supported if any part is not supported; inconclusive otherwise.

**Why 0.01.** In the gating benchmark, agent revenue per request is $\xi\,\mathbb E\min\{\theta_f,\theta_r\}$, which is about $0.67\,\xi\,\bar\theta$ for uniform values. A margin of $0.01\bar\theta$ is therefore at most 1.5% of the inclusion rent at $\xi=1$, and 15% at $\xi=0.1$. That is small enough that "equivalent to zero" rules out inclusion rents of the size the theory cares about, yet large enough to absorb the residual payments that finite E1-calibrated noise produces.

### 3.4 Per-cell missingness tolerance: 10% and 25%

**Definition.**

- Cell: (model version × interface × temperature).
- Missing rate: $m_c$ = trials coded `invalid`, `refusal`, `error` or `not_run`, divided by planned trials.

| $m_c$ | Treatment |
|---|---|
| ≤ 0.10 | Primary analysis is complete-case. The sensitivity analysis (missing coded as the outside option) is reported alongside. |
| 0.10 < $m_c$ ≤ 0.25 | Analysed and flagged. The cell keeps its label only if the sensitivity analysis gives the same label; otherwise it is labelled inconclusive. |
| > 0.25 | Excluded from confirmatory tests, reported as incomplete with reasons, and labelled inconclusive in the aggregation. |

**Additional rules.**

- **Within a set:** H1b needs at least 8 valid repeats out of 10 (§3.1).
- **Differential missingness:** within each cell, a χ² test checks whether missingness depends on the sponsored label or on position, with Holm across cells. If it does, the cell's H1c and position estimates keep their label only if both missing-data codings agree.
- **Refusals** are also reported by label arm. A refusal caused by the label is itself a label effect.
- **Engineering completeness is separate.** SL-18's requirement that at least 95% of trials have a recorded outcome is engineering; an `invalid` outcome is recorded but missing for analysis.

**Why 10% and 25%.** Complete-case analysis is usually safe below 10% missing. Above about 25%, it depends on missingness being unrelated to outcomes, which the registration cannot justify for agents whose refusals may respond to the manipulation itself.

---

## 4. Other registered numbers SL-01 needs

| Claim | Quantity | Registered value or rule |
|---|---|---|
| H1a | Triple odds test (Haldane-corrected, permutation) and Hausman–McFadden | α = 0.05, Holm over cells. A cell needs ≥ 20 informative triples; otherwise inconclusive. Supported per cell if the IIA null is rejected. |
| H1c | Minimum effect of interest for power (SL-17) | $\delta/\sigma=-0.25$, about 22% lower odds. SL-17 sizes for 80% power at max(0.25, \|pilot estimate\|). |
| H1c | Sign | One-sided Wald test, $\delta/\sigma<0$, per primary cell. |
| H1c | Heterogeneity | Wald χ² test of equal $\delta/\sigma$ across models, at the claim level. Supported if rejected. |
| H1e | Page-request rate for non-chosen displayed products (browser cells) | Threshold 0.10. Supported if the Wilson 95% lower bound ≥ 0.10; not supported if the upper bound < 0.10. Needs ≥ 200 non-chosen displayed products per cell. Also yields $\hat\chi_m$ for E2. |
| H2b, I-part | Gating gain against A1 §1.4 | Per market: relative error ≤ 1e-6. Across markets: \|mean − $n\xi\mathbb E\min$\| ≤ 3 Monte Carlo SE. Failure is an implementation error, not a finding. |
| H2b, Pr-part (`conj:gating`) | Monotonicity of ΔΠ(α) on α ∈ {0, .25, .5, .75, 1} | Tolerance τ = 1% of ΔΠ(1). Supported if every adjacent increment has a Holm-adjusted 95% lower bound ≥ −τ. Not supported if some increment has an upper bound < −τ. |
| H2c | M2 revenue ∝ E[1−α]; M1 payment per effective choice rises with αχ̄ | Regression slopes with 95% intervals: supported if the sign holds (one-sided). For proportionality, the intercept is equivalent to 0 within ±0.01 in ψ units. |
| H2c | Set-blind gap with deterministic organic lists (`prop:setblind`(a)) | I-check: \|V^aw − V^bl\| ≤ 1e-6 relative. A positive gap blocks reporting (SL-19). Growth with organic variation: slope 95% lower bound > 0. |
| H3a, I-part | Separated policy against the original (`prop:separation`) | Expected loads, contributions and match quality equal within 1e-9 relative. Across-state variance ≤ 1e-12. |
| H3a, Pr-part | Composition-blind exposure | Ratio $r=\operatorname{Var}_\Lambda(p_j)/(\Theta_j^2/4)$ for leaders. Supported if $r$'s 95% lower bound ≥ 0.5 in at least 2/3 of calibrated categories. States are equally likely model versions (A1 §E4). |
| H3b | `prop:threshold` closed forms (E3 Part A) | Quadratic: \|y − y*\| ≤ 1e-4 and objective relative error ≤ 1e-6. Overflow, limit objective: objective relative error ≤ 1e-6. Overflow, finite n: $f(y^\ast)-f(y_n)\le2\lambda/\sqrt n$ + 1e-6. The grid excludes Δ ∈ {0, λ}. 100% of grid points must pass (SL-20). |
| H3c | Buy-back under M3 and M4 (`cor:congestion`) | Under M3, the leader buy-back share > 0 (95% lower bound). Under M4, buy-back must coincide with "bid term exceeds congestion term" in ≥ 95% of (market, leader) pairs; pairs within 1e-6 of indifference are excluded. |
| H4a | Identification share | I-check: the identified share equals the reachability prediction exactly. Comparison: competitor-varying minus leader-inclusive at matched inclusion frequency, paired difference 95% lower bound > 0. Lower bounds on audit counts use A1 §3.2(b) (corrected). |
| H4b | `prop:learning`(a) | I-check with 10,000 replications per checked cell: bias ≤ 3 MC SE, and the variance ratio to the closed form within ±5%. Log–log RMSE slope on h: 95% interval ⊂ [−0.55, −0.45]. Dependence on n₀ is compared with the closed form (A1 §E3), not with "flat". |
| H4c | Misreport gain against $2\theta_jn\epsilon_1$ at ϱ = 0.05 | Supported if the Clopper–Pearson 95% upper bound of the violation share is ≤ 3ϱ = 0.15, and the max gain falls with h (slope on log h, 95% upper bound < 0). |
| H4d | Cumulative-loss exponent (corrected, A1 §E2) | Log–log slope of cumulative loss on h. Supported if the 95% upper bound ≤ 0.6; not supported if the 95% lower bound > 0.6; inconclusive otherwise. A slope near 0 (bounded loss) is consistent with the theory. |
| H5a | Purchases net of returns per effective choice, strong-competitor vs weak-competitor arms | Equivalence margin ±0.10 on the log ratio (about ±10%). TOST via Fisher randomisation intervals (10,000 draws). |
| H5b | Qualifying event vs click as purchase predictor | Superiority: out-of-sample log-loss difference (event − click), 95% upper bound < 0. AUC reported. Assessed by arm and, where identifiable, by traffic type. |

---

## 5. Minimum sample sizes (floors)

SL-17 sizes the main design within budget. It may raise these floors, never lower them. Below a floor, the label is inconclusive by rule.

| Claim | Floor |
|---|---|
| H1a | ≥ 20 informative triples per primary cell |
| H1b | ≥ 30 eligible sets per primary cell |
| H1c | 80% power at the minimum effect (§4) |
| H1d | ≥ 20 tasks per provider and version pair; alignment needs ≥ 10 credible changes |
| H1e | ≥ 200 non-chosen displayed products per browser cell |
| E2 | ≥ 200 valid markets per configuration (tickets SL-19) |
| E4 | R = 200 replications per cell (tickets SL-21); the H4b I-check uses 10,000 |

Note on H1d: the illustrative E1 design in tickets.md has about 10 tasks, which is below the floor. Unless SL-17 raises the task count to at least 20, H1d will be labelled inconclusive by rule, and the set-level change share in §3.2 becomes the main descriptive evidence for E3.
