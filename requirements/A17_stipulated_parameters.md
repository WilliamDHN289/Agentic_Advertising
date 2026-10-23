# A17 · Stipulated parameters for the pre-E1 simulations

Owner input to SL-19S, SL-20S and SL-21S (tickets.md §5, item A17).

| | |
|---|---|
| Decided by | Project owner, 2026-10-23 |
| Status | Frozen input. Changes are made by the owner between tickets (commit, then `review.py adopt-workflow`). |
| Companions | `A1_formal_statements.md` (statements, §E corrections) and `A12_thresholds.md` (statistics). |

## 0. What these parameters are for, and what they are not

- **Pre-E1 stand-ins.** These parameters replace the E1 estimates (`agent_params_e1.json`) only in the pre-E1 dry runs SL-19S, SL-20S and SL-21S. They are not estimates of any real agent. SL-19 to SL-21 rerun the same code on E1 parameters, and only those runs are confirmatory.
- **Choice principle.** Each baseline sits in the regime the paper's results address: small agent noise, a label penalty, and an outside option rarely chosen. Each sensitivity level moves one assumption toward where a result should weaken or fail, so the dry runs show both the prediction and its boundary.
- **Exploratory labels.** Every figure, table and decision label from these runs is marked *"Stipulated parameters (A17), pre-E1, exploratory"*. Registered statistics are computed as written in the claims register, but no confirmatory label is recorded.
- **No forking paths.** The A12 thresholds and the claims register stay frozen while these runs are inspected. A change to a registered rule after seeing dry-run output is a dated owner amendment that states the reason, and it is listed in the E1 registration.

**Units.** Each product has a latent quality $z_j\sim N(0,1)$. Agent scores are on the same scale (standard deviation 1), and $\sigma_m$ is the logit noise scale in those units. Reports give $v/\sigma$, $\delta/\sigma$ and $\beta/\sigma$, as E1 will.

---

## 1. Shared primitives (all three tickets unless a section overrides them)

| Symbol | Meaning | Baseline | Sensitivity levels | Reason |
|---|---|---|---|---|
| $\lvert\mathcal X\rvert$ | contexts per market | 1 | — | The mechanism results hold context by context. One context keeps the runtime within the SL-16 profile (about 3.2 s per baseline market for all six mechanisms). |
| $N_x=\lvert E(x)\rvert$ | eligible products | 10 | E4: {10, 30} | E1 uses eligible sets of 10–30, and the paper's E4 grid is {10, 30}. |
| $z_j$ | latent quality | $N(0,1)$, i.i.d. | — | A neutral common scale. |
| $u(x,j)$ | match quality | $\Phi(z_j)\in(0,1)$; $u(x,0)=0$ | — | The guardrail is multiplicative and needs $M^{\mathrm{ref}}>0$ and a fixed cardinal scale (`eq:mq-constraint`). The normal CDF gives positive values on [0, 1]. |
| $v^m_j$ | agent baseline score | $\rho_v z_j+\sqrt{1-\rho_v^2}\,e_{mj}$, with $e_{mj}\sim N(0,1)$ i.i.d. per model and product | $\rho_v\in\{0.5,0.95\}$ | Agents track quality but disagree across models. $\rho_v=0.8$ makes models agree on most favourites, but not all. $\rho_v$ is the main control on $\lvert F(x)\rvert$ besides the number of models. |
| $\sigma_m$ | agent noise scale | 0.3 | {1.0, 0.1} | With 10 standard-normal scores, the mean gap between the top two is about 0.54, i.e. 1.8σ at σ = 0.3. That exceeds H1b's ln 4 = 1.39 (A12 §3.1), so a typical set has a modal choice share near 0.85: near-deterministic, but not the limit. σ = 1.0 is clearly noisy (gap 0.54σ). σ = 0.1 approaches the limit (gap 5.4σ). The levels match the SL-15 recovery grid. |
| $v^m_0$ | outside-option score | −1.0 | {0.0} | At −1.0, a 4-product set loses to the outside option with probability about $0.16^4\approx0.0006$, as Props gating and separation assume ("outside option below every product"). At 0.0 the outside option beats half the products, which tests that assumption. |
| $\delta_m$ | label effect, score units | $L=2$: (−0.3, −0.7), mean −0.5 | all models 0; all models −1.5 | The audit evidence (Allouah et al. 2026) shows sponsored-label penalties that differ across models (H1c). −0.5 means a sponsored product needs about half a standard deviation more quality to be chosen. δ = 0 is the setting of Prop setblind(b). δ = −1.5 often breaks the labelled-favourite condition of `prop:rents`(b), so it tests the H2a screen (A1 §E5). With $L=1$: δ = −0.5. With $L=4$: (−0.3, −0.5, −0.5, −0.7). |
| $\beta_{m,k}$ | position effect | 0 | $\beta_k=-0.1(k-1)$ | The main analysis sets β = 0 (Ass. deterministic, l.311). The sensitivity level is a small residual effect, about 0.1 SD per position; it was the earlier `e2_core.yaml` placeholder. |
| $\chi_m$ | page-request probability (Ass. RET) | 0.5 | {0, 1} | The paper's E2 factor: 0 for structured-data agents, 1 for browser agents. |
| $L$, $\pi_m$ | number of models, model shares | 2, equal shares | $L\in\{1,4\}$, equal shares | A monoculture against a diverse population. With $\rho_v$, this sets $\lvert F(x)\rvert$, the H2a regressor. |
| $\gamma^H_k$ | human position weight | $0.30\,k^{-0.6}$: 0.300, 0.198, 0.155, 0.131, 0.114, 0.102, 0.093, 0.086 | exponent {0.3, 1.0} | Holds until A11 is extracted from Ursu (2018). It is a power-law decay matching the earlier placeholder (0.30, 0.20, 0.15, 0.12); the levels bracket flat and steep decay. |
| $\rho_j(x,\ell)$ | human page-request probability per examination | organic 0.6; sponsored 0.54 | — | A 10% discount for the label. Moving a product to position 1 still raises human choices whenever $\gamma_1\cdot0.9>\gamma_k$ (true for every k ≥ 2), so position rents exist, as `prop:rents`(d) needs. |
| $\zeta_j(x,\ell)$ | human conversion after a request | $0.2+0.6\,u_j$ | — | Humans buy better matches more often. $w=\rho\zeta\in[0.11,0.48]$, and $\sum_k\gamma_kw\le0.57<1$ (Ass. HUM). |
| advertisers | | 6 of the 10 eligible products, drawn uniformly per market | — | Enough advertisers for competition, while leaving some non-advertised products. Non-advertised products matter for `prop:gating`(c) and the `prop:setblind`(b) instance. |
| $g(x,j)$ | platform contribution | 0.1 for every product | $g_j\sim U[0.05,0.15]$ | The common-contribution case of Props gating and setblind. Small next to $\bar\theta=0.5$, as in retail media, where ad values exceed commissions. |
| $\theta_j$ | value per effective choice | $U[0,1]$ | lognormal with $\mu=-0.5$ and $s\in\{0.5,2\}$ | Uniform is regular. Lognormal with s = 2 needs Myerson ironing (A16, registered). These match the SL-16 validation markets. |
| $\kappa$ | guardrail slack | 0.05 | {0, 0.2} | Allows a 5% loss of attainable match quality. 0 rules out gating revenue unless $u(r)=u(f)$; 0.2 is permissive. |
| $n$ | requests per period | 1000 | — | Large enough for the $n^2$ common-shock term to dominate (A12 §3.2), and small enough for exact binomial cost terms. |
| $C_j$ | operating cost (M4, E3) | quadratic, $\lambda=0.5$, $B=\omega n$, $\omega=0.5$ | E2: $\lambda\in\{0.1,2\}$; E3: §3 grid | Per request, a product with choice share p costs about $\lambda p^2/(2\omega)$. A leader at p = 0.5 then costs 0.125 per request, against about 0.25 of advertiser value. Cost is material, but not dominant. |

**Determinism.** Every draw comes from `stream(master_seed, "<ticket>-market", config_id, index)`, so the market inputs are identical across mechanisms (common random numbers). The master seed is registered in each ticket's config.

---

## 2. E2 (SL-19S)

| Item | Value |
|---|---|
| K, $K^s$ | 4, 1 (baseline); 8, 2 (sensitivity) |
| α (agent share) | {0, 0.25, 0.5, 0.75, 1} |
| States | 1 (baseline). Sensitivity: 2 equally likely "versions" per model, where version 2 adds $\tau\eta_{mj}$ to $v^m_j$ with $\eta_{mj}\sim N(0,1)$ and $\tau=0.5$. |
| Mechanisms | M1, M2, set-blind M3, M3, M4, profit rule (SL-16) |
| Bidding in M1, M2 | fixed-point value per click (baseline); human conversion rate (sensitivity) |
| Markets | 200 per configuration |
| Prop gating instance (H2b, I-part) | One context, $\alpha=1$, $L=1$, one state, β = 0, δ = −0.5, $K=4$, $K^s=1$, $C\equiv0$, $g=0.1$. $f$ and $r$ are the agent's top two products. Scores are redrawn until $v_f+\delta>v_r$ and $v_r+\delta>v_i$ for all other $i$. $r$ is an advertiser. κ is in {0.05, 0.2}, and the limit σ → 0 is evaluated exactly. |
| Conj gating | The same instance with humans added at shares $1-\alpha$ for α ∈ {0, .25, .5, .75, 1}, using the lists $R^a$ and $R^b$ of A1 §1.5. |

**Organic regimes.**

| Regime | Definition |
|---|---|
| human-optimised | $E(x)$ ranked by $u$, descending. This maximises human match quality and contribution under a common $g$, since human conversion rises with $u$. |
| agent-aware | The favourites $F(x)$ first, ordered by $u$, then the rest by $u$. If $\lvert F(x)\rvert>K-K^s$, the market is *aware-infeasible* (`cor:aware`) and is reported separately. |
| gated | As defined in A1 §E7. |

**Organic variation.**

- *none:* one ranking.
- *randomised:* two equally likely rankings, the regime's ranking $R$ and $R'$. To build $R'$, swap the unconstrained products in positions $1..K-K^s$ with those in positions $K-K^s+1..2(K-K^s)$. Favourites in agent-aware lists, and second-ranked rivals in gated lists, stay where they are. The displayed organic products then differ between the two lists, which is the variation `prop:setblind` prices.

---

## 3. E3 (SL-20S)

### 3.1 Part A: grid for the `prop:threshold` check (no stipulated agents)

The net value of the unfavoured leader is $\nu_-=0$.

| Case | Grid |
|---|---|
| Quadratic | $n\in\{10,100,1000\}$; $\omega\in\{0.3,0.5,0.6,0.8,1\}$ with $B=\omega n$; $\lambda\in\{0.25,0.5,1,2,4\}$; $\Delta\in\{0.05,0.1,0.2,0.4,0.8,1.6\}$. 450 points. |
| Overflow, limit objective | the same ω, λ and Δ values. 150 points. No Δ equals a λ, so there are no knife-edges (A1 §E6). |
| Overflow, finite n | the same 150 points at $n\in\{100,1000,10000\}$, checked against the A1 §2.3 bound. |

### 3.2 Part B: stipulated states

| Item | Value |
|---|---|
| Leaders | Two leaders with equal mean score. Every other product has $z$ drawn from $N(0,1)$ truncated to at most (leader mean − 0.5), so the two leaders lead. |
| States | 2, equally likely (A1 §E4). The state adds ±0.5 to the two leaders' scores in opposite directions. Their gap of 1.0 is 3.3σ at σ = 0.3, so the favoured leader wins a head-to-head set with probability about 0.96. |
| State and quality | *not aligned:* $u$ does not depend on the state (`prop:separation` applies). *aligned:* $u(x,j,\Lambda)=\Phi(z_j+\text{state shift})$ (`prop:threshold` applies). |
| Costs | quadratic and overflow; $\omega\in\{0.3,0.5,0.6,0.8,1\}$; $\lambda\in\{0.1,0.5,2\}$; $n=1000$ |
| Policies | match-quality-maximising deterministic organic policy; its separated version; composition-blind randomisation; a quota implementation of the separated version |
| Sensitivity | human share $1-\alpha\in\{0,0.25,0.5\}$; state probabilities (0.7, 0.3); 3 states (the third has zero shift); two models whose state shifts are identical, independent, or opposite |
| H3c | sponsorship on, M3 against M4, with the E2 baselines for advertisers and values |

---

## 4. E4 (SL-21S)

| Item | Value |
|---|---|
| Setting | One context and one model (`prop:identify` setting). σ = 0.3, $v_j\sim N(0,1)$, outside score −1.0, $K=4$. |
| Across-state variance | 2 equally likely states. State 2 adds $\tau\eta_j$, with $\eta_j\sim N(0,1)$, $\tau=0.5\sqrt f$ and factor $f\in\{0,0.5,1,2\}$. |
| Audit designs | Each audits $\lvert\mathcal S\rvert=2N_x$ sets of size K. *Leader-inclusive:* every set contains the product with the highest mean score. *Competitor-varying:* a seeded design whose comparison graph $\mathcal G_{\mathcal A}$ connects every pair of the top 5 products. *Random:* sets matched to the competitor-varying design on per-product inclusion frequency. |
| Grids | $h\in\{5,10,20,50,100\}$, $n_0\in\{2,10,100,1000\}$, $N_x\in\{10,30\}$, R = 200 replications per cell |
| Mechanism (H4c, H4d) | $n=1000$ mechanism requests; 4 advertisers; $\theta\sim U[0,1]$; $g=0.1$; quadratic cost with λ = 0.5 and ω = 0.5; a menu $\mathcal Q_{\mathrm{fin}}$ of 16 seeded policies, each a uniform mixture over 4 audited cells; ϱ = 0.05 (A12 §4) |
| Recovery after a version change | Exploratory (SL-21). The state distribution switches after period h/2. |
