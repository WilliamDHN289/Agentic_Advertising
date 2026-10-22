# A1 · Formal statements for paper Sections 4–6

Owner input to SL-01, SL-16 and SL-19 to SL-21 (tickets.md §5, item A1).

| | |
|---|---|
| Source | `requirements/paper.tex`, a snapshot of the working draft taken 2026-10-22, sha256 `e4bcf05266b5c835d8dc827e829a86d1856f23bb3f295766aae899c154d63295`. Line numbers below refer to it. |
| Decided by | Project owner, 2026-10-22 |
| Status | Frozen input. Changes are made by the owner between tickets (commit, then `review.py adopt-workflow`). |
| Precedence | The paper is authoritative for the mathematics. This file restates each result so it can be read alone, gives the closed forms the experiments compute, and records corrections (§E). Where §E differs from the paper or from tickets.md §1.1, the registration follows §E. Decision thresholds are in `A12_thresholds.md`. |

## Index

Kind follows tickets.md §1.1: **I** is an exact consequence of the model (simulation checks only the implementation); **Pr** is a prediction tested under finite noise or calibrated heterogeneity.

| A1 item | Label | Line | Kind in the registration | Hypotheses | Tickets |
|---|---|---|---|---|---|
| Prop rents | `prop:rents` | 554 | Pr (exact only in the deterministic limit) | H2a | SL-16, SL-19 |
| Cor aware (supporting) | `cor:aware` | 570 | I | H2a regressor | SL-16, SL-19 |
| Prop gating | `prop:gating` | 580 | I | H2b, first part | SL-16, SL-19 |
| Conj gating | `conj:gating` | 593 | Pr (numerical) | H2b, second part | SL-19 |
| Prop setblind | `prop:setblind` | 607 | I for (a), I for the instance in (b) | H2c, third part | SL-16, SL-19 |
| Lemma click | `lem:click` | 615 | Pr (under finite noise and the retrieval model) | H2c, first and second parts | SL-16, SL-19 |
| Lemma contested | `lem:contested` | 626 | I (bound); Pr ("close to the bound") | H3a | SL-20 |
| Prop separation | `prop:separation` | 636 | I | H3a | SL-20 |
| Prop threshold | `prop:threshold` | 652 | I | H3b (E3 Part A) | SL-20 |
| Cor congestion | `cor:congestion` | 664 | I (identity); Pr (buy-back behaviour) | H3c | SL-16, SL-20 |
| Prop identify | `prop:identify` | 685 | I | H4a | SL-21 |
| Cor design | `cor:design` | 689 | I; (b) **corrected**, §E1 | H4a | SL-21 |
| Prop learning | `prop:learning` | 710 | I for (a); Pr for (b)–(d) | H4b, H4c | SL-21 |
| Cor cumulative | `cor:cumulative` | 730 | Pr (upper bound only, §E2) | H4d | SL-21 |

---

## 0. Standing notation and assumptions

**Primitives.** Products $\mathcal N=\{1,\dots,N\}$, advertisers $\mathcal N^{\mathrm{ad}}\subseteq\mathcal N$. A period has $n$ requests. Context $x_t\sim\mu$ on a finite set $\mathcal X$; eligible set $E(x)$; at most $K$ displayed products, at most $K^s\le K$ of them sponsored. An allocation $a=(R,S^s)$ pairs an organic ranking $R$ of $E(x)$ with an ordered sponsored list $S^s\subseteq E(x)\cap\mathcal N^{\mathrm{ad}}$. The sponsored products occupy the first $|S^s|$ positions with labels. The rest of the list is $O(R,S^s)$: the first $K-|S^s|$ products of $R$ not in $S^s$, unlabeled. $S(a)$ is the displayed set, $\ell_j(a)$ the label and $k_j(a)$ the position.

**Policies.** The organic policy $Q^o(\cdot\mid x)$ is fixed before bids (organic integrity, Def. `def:organic-integrity`). The sponsored policy is $Q^s(\cdot\mid x,R)$, and $Q(a\mid x)=Q^o(R\mid x)Q^s(S^s\mid x,R)$. Allocations are i.i.d. across requests given contexts. The policy is fixed within a period.

**Buyers.** The type is $\tau\in\{H,A_1,\dots,A_L\}$, agent with probability $\alpha(x)$, model $m$ with probability $\pi_m(x)$. The common state $\Lambda\sim F_\Lambda$ is drawn once per period. Choice probabilities:

- humans: $q^H_j=\gamma^H_{k_j(a)}w_j(x,\ell_j(a))$ (Ass. `ass:human`);
- agents: logit with scores $s^m_j=v^m_j(x,\Lambda)+\delta_m\ell_j(a)+\beta_{m,k_j(a)}$, noise scale $\sigma_m$ and outside score $v^m_0(x)$ (eq. `eq:agent-logit`);
- pooled: $q_j(x,a\mid\Lambda)$ and $\bar q_j=\mathbb E_\Lambda q_j$ (eq. `eq:mixed-q`).

**Outcomes.**

- $p^Q_j(\Lambda)=\sum_x\mu(x)\sum_aQ(a\mid x)q_j(x,a\mid\Lambda)$.
- $X_j(Q)=n\,\mathbb E_\Lambda p^Q_j(\Lambda)$ is the expected number of effective choices.
- $D_j$ is the load. Given $\Lambda$, $D_j\sim\mathrm{Bin}(n,p^Q_j(\Lambda))$ (Lemma `lem:load`).
- $G(Q)$ is expected contribution minus expected operating cost $\sum_j\mathbb E\,C_j(D_j)$.
- $M(Q)$ is expected match quality. $M^{\mathrm{ref}}=\max_{Q^o}M(Q^o,\varnothing)$.
- Guardrail: $M(Q^o,Q^s)\ge(1-\kappa)M^{\mathrm{ref}}$ with $0\le\kappa<1$ (eq. `eq:mq-constraint`).

**Mechanism.**

- Allocation: $Q^{s\ast}(b)\in\arg\max_{Q^s\in\mathcal Q}\{\sum_jb_jX_j+G\}$ (eq. `eq:alloc`), with ties broken by a fixed bid-independent rule.
- Payment: $P_j(b)=b_jX_j(b)-\int_0^{b_j}X_j(z,b_{-j})\,dz$ (eq. `eq:payment`).

**Assumptions used below** (short names):

| Short | Label | Content |
|---|---|---|
| CI | `ass:cond-indep` | Given $\Lambda$ and $Q$, the tuples $(x_t,\tau_t,a_t,J_t)$ are i.i.d. The law of $J_t$ depends only on $(x_t,\tau_t,a_t,\Lambda)$. |
| HUM | `ass:human` | Separable human response, $\gamma^H_1\ge\dots\ge\gamma^H_K>0$, total probability at most 1, and no dependence on $\Lambda$. |
| DET | `ass:deterministic` | For every $m$: $\beta_{m,k}=0$, $\delta_m\le0$, and $\sigma_m\to0$ limits apply. No ties among $\{v^m_0\}\cup\{v^m_j+d: d\in\{0,\delta_m\}\}$. |
| RET | `ass:retrieval` | A human requests a page with probability $\gamma^H_k\rho_j$ and then chooses with probability $\zeta_j$, so $w_j=\rho_j\zeta_j$. A model-$m$ agent requests each displayed page independently with probability $\chi_m$, whatever it chooses. |
| VAL | `ass:value` | Value $\theta_j\ge0$ per effective choice, private, drawn independently from $F_j$. Risk neutral, quasilinear. |
| COST | `ass:cost` | $C_j$ is nondecreasing with nonnegative second differences. |
| KNOWN | `ass:known` | The platform knows the policy outcomes. Bids do not change buyer response. |

**Objects defined under DET.**

- *Type.* A type is $(m,\Lambda)$. It chooses the unique $w^m(x,a,\Lambda)\in S(a)\cup\{0\}$.
- *Favourite.* $f_{m,\Lambda}(x)$ is the alternative in $E(x)\cup\{0\}$ with the highest *unlabeled* score.
- *Type with positive probability at $x$.* $\pi_m(x)>0$ and $\Lambda\in\operatorname{supp}F_\Lambda$.

---

## 1. Section 4: what advertising sells to agent traffic

### 1.1 Agent-aware organic policy (Def. `def:aware`, l.550)

Let $K^s<K$. $Q^o$ is *aware* of type $(m,\Lambda)$ at $x$ if $f_{m,\Lambda}(x)\in E(x)$ and every list in $\operatorname{supp}Q^o(\cdot\mid x)$ places $f_{m,\Lambda}(x)$ among its first $K-K^s$ products.

### 1.2 Proposition rents: agent traffic pays no position rents (`prop:rents`, l.554)

**Hypotheses.** DET and $K^s<K$. Fix $x$ and a type $(m,\Lambda)$ with favourite $f\in E(x)$. $Q^o$ is aware of $(m,\Lambda)$ at $x$.

**Statement.**

- **(a)** For every $R\in\operatorname{supp}Q^o(\cdot\mid x)$ and every feasible $S^s$, the type chooses $f$ under $(R,\varnothing)$. It chooses an alternative other than $f$ under $(R,S^s)$ only if $f\in S^s$.
- **(b)** Suppose in addition that $v^m_f+\delta_m>v^m_0$ and $v^m_f+\delta_m>v^m_i$ for every $i\in E(x)\setminus\{f\}$; call this the *labelled-favourite condition*. Then the type chooses $f$ under every $(R,S^s)$.
- **(c)** Suppose that at every $x$, $Q^o$ is aware of every positive-probability type whose favourite lies in $E(x)$, and each such type satisfies the labelled-favourite condition. Then the agent component $X^A_j$ of $X_j$ does not depend on $Q^s$. Under eq. `eq:alloc`, for every $j$ and $b$:
  $$P_j(b)=b_jX^H_j(b)-\int_0^{b_j}X^H_j(z,b_{-j})\,dz\ \le\ b_j\,n\,\mathbb E_\mu[1-\alpha(x)],$$
  where $X^H_j=X_j-X^A_j$ is the human component. Total payments therefore tend to 0 as $\mathbb E_\mu[\alpha]\to1$.
- **(d)** Under HUM, sponsorship can raise human choices under any organic policy. Suppose $j$ is displayed organically at position $k$, or not displayed (set $\gamma^H_k=0$), and $\gamma^H_1w_j(x,1)>\gamma^H_kw_j(x,0)$. Then $S^s=(j)$ raises $j$'s human choice probability.

**Closed form used by E2.** Under (c):
$$X^A_j=n\sum_x\mu(x)\alpha(x)\sum_m\pi_m(x)\Pr_\Lambda\big(f_{m,\Lambda}(x)=j\big).$$

**Scope.** The result is exact only in the deterministic limit and under the labelled-favourite condition. E2 calibrates $\delta_m/\sigma_m$ and $\beta_{m,k}/\sigma_m$ on E1, so both may fail. See §E5 for how H2a handles that.

### 1.3 Corollary aware: when agent-aware lists exist (`cor:aware`, l.570)

Let $F(x)=\{f_{m,\Lambda}(x)\in E(x):\pi_m(x)>0,\ \Lambda\in\operatorname{supp}F_\Lambda\}$. The outside option is never in $F(x)$.

**Statement.** An organic policy aware of all positive-probability types at $x$ exists if and only if $|F(x)|\le K-K^s$. The excess $e(x)=|F(x)|-(K-K^s)$ is the regressor in the third part of H2a.

### 1.4 Proposition gating: organic gating (`prop:gating`, l.580)

**Hypotheses.** All of the following hold:

- Traffic and market: one context, $\alpha\equiv1$, one agent model, a degenerate common state, DET.
- Mechanism and economics: $K^s=1<K$, $C_j\equiv0$, contribution $g(x,j)=g_0$ for all $j$.
- Outside option: scored below every product.
- Rankings: $f$ is the agent's favourite and $r$ its second-ranked product, with $v_f+\delta>v_r$ and $v_r+\delta>v_i$ for all $i\notin\{f,r\}$.
- Match quality: $u(f)=\max_ju(j)>0$, and
  $$\xi=\begin{cases}1 & u(r)=u(f),\\ \min\{1,\ \kappa u(f)/(u(f)-u(r))\} & \text{otherwise.}\end{cases}$$
- Bidding: advertisers bid truthfully, and $\theta_r:=0$ if $r\notin\mathcal N^{\mathrm{ad}}$.

**Statement.**

- **(a) Aware design.** $f$ is among the first $K-1$ products of every organic list. Every payment is 0, every request ends with $f$, and $M=M^{\mathrm{ref}}=nu(f)$.
- **(b) Gated design.** $f$ is never displayed organically, and $r$ is among the first $K-1$ products of every list. Eq. `eq:alloc` sponsors $f$ on every request if $\theta_f>\theta_r$, and on a fraction $1-\xi$ of requests if $\theta_f<\theta_r$. Revenue and match quality are:

  | | Revenue | Match quality |
  |---|---|---|
  | $\theta_f>\theta_r$ | $n\xi\min\{\theta_f,\theta_r\}$ | $nu(f)$ |
  | $\theta_f<\theta_r$ | $n\xi\min\{\theta_f,\theta_r\}$ | $n[u(f)-\xi(u(f)-u(r))]$ |
- **(c)** Both designs satisfy organic integrity and the guardrail. If $r\in\mathcal N^{\mathrm{ad}}$, $\kappa>0$ and $\Pr(\min\{\theta_f,\theta_r\}>0)>0$, then (b) yields strictly higher expected profit than (a).

**Closed forms checked by E2 (H2b, I-type).** Per market with realised $(\theta_f,\theta_r)$:
$$\Pi_{(b)}-\Pi_{(a)}=n\xi\min\{\theta_f,\theta_r\},\qquad M_{(a)}-M_{(b)}=n\xi\,(u(f)-u(r))\,\mathbf 1\{\theta_f<\theta_r\}.$$
In expectation:
$$\mathbb E[\Pi_{(b)}-\Pi_{(a)}]=n\xi\int_0^\infty\big(1-F_f(t)\big)\big(1-F_r(t)\big)\,dt.$$

- i.i.d. $U[0,\bar\theta]$ values: $\mathbb E\min=\bar\theta/3$.
- i.i.d. $\log\theta\sim N(\mu_\theta,s^2)$: $\mathbb E\min=2e^{\mu_\theta+s^2/2}\,\Phi(-s/\sqrt2)$.
- Expected match-quality loss: $n\xi(u(f)-u(r))\Pr(\theta_f<\theta_r)$.

Ties $\theta_f=\theta_r$ have probability zero under the registered distributions.

### 1.5 Conjecture gating: gating and the agent share (`conj:gating`, l.593)

**Setting.** Take the environment of Prop gating and add a share $1-\alpha$ of human requests satisfying HUM. The platform does not observe types, so both traffic types see the same organic policy. The designs are made precise as follows (registration operationalisation):

- **Aware list.** $R^{a}$ is a fixed organic list with $f$ first, $r$ second, then the other eligible products in a fixed order.
- **Gated list.** $R^{b}$ is $R^{a}$ with $f$ removed and the remaining products moved up.

**Conjecture.** Let $\Pi_{(a)}(\alpha)$ and $\Pi_{(b)}(\alpha)$ be expected profits under eq. `eq:alloc` with truthful bids. The gain
$$\Delta\Pi(\alpha)=\Pi_{(b)}(\alpha)-\Pi_{(a)}(\alpha)$$
is nondecreasing on $[0,1]$. At $\alpha=1$ it equals the expected gain from Prop gating.

### 1.6 Set-blind policy and Proposition setblind (`def:setblind` l.603, `prop:setblind` l.607)

**Definition.** $Q^s$ is *set-blind* if $Q^s(\cdot\mid x,R)$ does not depend on $R$. "Set-blind M3" is eq. `eq:alloc` with $C_j\equiv0$, maximised over set-blind sponsored policies only.

**Hypotheses.** $C_j\equiv0$, and the guardrail does not bind. Bids are $b$, with $b_j=0$ for $j\notin\mathcal N^{\mathrm{ad}}$. Define
$$U_b(x,R,S^s)=\sum_j\big(b_j+g(x,j)\big)\,\bar q_j\big(x,(R,S^s)\big).$$

**Statement.**

- **(a)** With $R\sim Q^o(\cdot\mid x)$:
  $$V^{\mathrm{aw}}(b)=n\sum_x\mu(x)\,\mathbb E_R\Big[\max_{S^s}U_b\Big]\ \ge\ V^{\mathrm{bl}}(b)=n\sum_x\mu(x)\max_{S^s}\mathbb E_R\big[U_b\big].$$
  Equality holds if and only if, for every $x$ with $\mu(x)>0$, one sponsored list maximises $U_b(x,R,\cdot)$ for every $R\in\operatorname{supp}Q^o(\cdot\mid x)$. In particular, $V^{\mathrm{aw}}=V^{\mathrm{bl}}$ whenever $Q^o(\cdot\mid x)$ is deterministic for every $x$; this is the E2 implementation gate.
- **(b)** For every $N_a\ge1$, some instance has
  $$\frac{V^{\mathrm{bl}}-V^{\varnothing}}{V^{\mathrm{aw}}-V^{\varnothing}}=\frac1{\mathsf H_{N_a}},\qquad \mathsf H_{N_a}=\sum_{i=1}^{N_a}\frac1i.$$
  The instance:
  - Traffic: one context, $\alpha\equiv1$, one model, DET with $\delta=0$.
  - Lists: $K\ge2$, $K^s=1$.
  - Products: advertisers $1..N_a$, non-advertised $o_1..o_{N_a}$, and $K-1$ fillers.
  - Scores: $v_1>v_{o_1}>v_2>v_{o_2}>\dots>v_{N_a}>v_{o_{N_a}}>v_{\text{filler}}>v_0$.
  - Organic policy: $i\sim U\{1..N_a\}$, with list $R_i = (o_i,\text{fillers},\dots)$. Advertisers are never shown organically.
  - Bids: $b_j=1/(N_a-j+1)$; $g$ is constant.

  Then $(V^{\mathrm{aw}}-V^{\varnothing})/n=\mathsf H_{N_a}/N_a$ and $(V^{\mathrm{bl}}-V^{\varnothing})/n=1/N_a$. SL-16 uses this instance as a hand-solved market.

### 1.7 Lemma click: click benchmarks under agent traffic (`lem:click`, l.615)

**Hypotheses.** RET. M1 ranks by $b^{\mathrm c}_j\big[(1-\alpha(x))\gamma^H_k\rho_j(x,1)+\alpha(x)\bar\chi(x)\big]$ and charges every page request. M2 ranks by $b^{\mathrm c}_j\gamma^H_k\rho_j(x,1)$ and charges human requests only. Here $\bar\chi(x)=\sum_m\pi_m(x)\chi_m$.

**Statement.**

- M1 and M2 are set-blind.
- Under M2, agent requests generate no charges, and the ranking does not depend on agent choices.
- Under M1, the expected number of charged agent page requests for $j$ at $x$ equals $\bar\chi(x)$ times $j$'s expected number of sponsored inclusions in agent requests, regardless of how often agents choose $j$.
- As $\alpha(x)\to1$, the predicted rates of all products in a position converge to $\bar\chi(x)$, and M1 ranks by click bid alone.

---

## 2. Section 5: correlated agent demand

### 2.1 Lemma contested: exposure comes from state-contested sets (`lem:contested`, l.626)

**Hypotheses.** DET. An action $a$ is *$j$-contested* at $x$ for model $m$ if $\mathbf 1\{w^m(x,a,\Lambda)=j\}$ is not constant over $\Lambda\in\operatorname{supp}F_\Lambda$. $\mathcal C^m_j(x)$ is the set of such actions. Define
$$\Theta_j(Q)=\sum_x\mu(x)\alpha(x)\sum_m\pi_m(x)\,Q\big(\mathcal C^m_j(x)\mid x\big).$$

**Statement.**

- $\operatorname{Var}_\Lambda\big(p^Q_j(\Lambda)\big)\le\Theta_j(Q)^2/4$.
- Equivalently, the across-state component of the load variance satisfies $n^2\operatorname{Var}_\Lambda(p^Q_j)\le n^2\Theta_j(Q)^2/4$.
- A policy with no mass on $j$-contested actions has zero common-shock exposure for $j$.

**When the bound is attained.** Write $p^Q_j(\Lambda)=c+Y$ with $Y\in[0,\Theta_j]$. Equality holds if and only if $Y$ takes only the values $0$ and $\Theta_j$, each with probability $1/2$. With two states of probabilities $(\varpi,1-\varpi)$ and every contested action flipping together, $\operatorname{Var}_\Lambda(p^Q_j)=\varpi(1-\varpi)\Theta_j^2$. The ratio to the bound is $4\varpi(1-\varpi)$; see §E4.

**Worked check (Example `ex:diversify`, l.395).** $n=1000$, two equally likely states. Variances of $D_A$:

| Policy | $\operatorname{Var}(D_A)$ |
|---|---|
| $\{A,B\}$ always | 250,000 |
| $\{A,C\}$ or $\{B,C\}$, equally likely | 250 |
| $\{A,B\}$ or $\{A,C\}$, equally likely | 62,625 (= 125 within states + 62,500 across states) |

In the last row $\Theta_A=1/2$, and the across-state part $n^2/16$ attains the bound.

### 2.2 Proposition separation: separating state-sensitive rivals (`prop:separation`, l.636)

**Hypotheses.** DET, $\alpha\equiv1$, no sponsorship, and an outside option scored below every eligible product. Assume *anchors exist*: for every $x$ and every product $j$ chosen by some type under some feasible action, there is a feasible organic list $R^\dagger_j(x)$ under which every positive-probability type chooses $j$.

**Construction.** Let $\varpi_j(x,R)=\sum_m\pi_m(x)\Pr_\Lambda\big(w^m(x,(R,\varnothing),\Lambda)=j\big)$. Then
$$\tilde Q^o(\cdot\mid x)=\sum_RQ^o(R\mid x)\sum_j\varpi_j(x,R)\,\delta_{R^\dagger_j(x)}.$$

**Statement.** For every organic policy $Q^o$, the policy $\tilde Q^o$ satisfies:

- **(a)** $p^{\tilde Q}_j(\Lambda)=\mathbb E_\Lambda[p^Q_j(\Lambda)]$ for all $j,\Lambda$, so there is no common-shock exposure.
- **(b)** $\tilde Q^o$ gives the same expected loads, expected contributions and match quality as $Q^o$.
- **(c)** $\mathbb E\,C_j(D_j)$ is weakly lower for every $j$. It is strictly lower for every $j$ whose $C_j$ has a positive second difference and whose $p^Q_j(\Lambda)$ varies with $\Lambda$.

Consequently, if every $C_j$ has a positive second difference, every optimal organic policy has no common-shock exposure.

**Scope.** The proof uses that $g$ and $u$ do not depend on $\Lambda$. When version shifts align with $u$ (H1d), Prop threshold governs instead.

### 2.3 Proposition threshold: optimal head-to-head share (`prop:threshold`, l.652)

**Two-leader environment.**

- Traffic and lists: one context, $\alpha\equiv1$, one agent model, $K=2$. Lists show two of the products $\{1,2,3\}$.
- States: two, equally likely, with $v_1>v_2>v_3>v_0$ in $\Lambda_1$ and $v_2>v_1>v_3>v_0$ in $\Lambda_2$.
- Costs: products 1 and 2 share a cost function; product 3 has zero cost.
- Net value: $\nu=g+\eta u$, where $\eta\ge0$ is the guardrail multiplier. In each state, the net value of the leader the state favours exceeds that of the other leader by $\Delta\ge0$; $\nu_-$ is the net value of the unfavoured leader.
- Decision variable: $y$ is the probability of showing $\{1,2\}$.

**Statement.** Some optimal policy shows $\{1,3\}$ and $\{2,3\}$ with probability $(1-y)/2$ each. Every such policy gives each leader expected load $n/2$. With $\lambda>0$ and quadratic cost, the optimum is unique and symmetric (§E6).

- **(a) Quadratic cost** $C(D)=\lambda D^2/(2B)$, $n\ge2$:
  $$J(y)=n\nu_-+\frac{n\Delta(1+y)}2-\frac{\lambda n}{4B}\big[n+1+(n-1)y^2\big],\qquad y^\ast=\min\Big\{1,\ \frac{\Delta B}{\lambda(n-1)}\Big\}.$$
  With $B=\omega n$, $y^\ast\to\min\{1,\Delta\omega/\lambda\}$. With $\Delta=0$, $y^\ast=0$.
- **(b) Overflow cost** $C(D)=\lambda(D-\omega n)^+$. The per-request objective converges uniformly in $y$ as $n\to\infty$ to
  $$f(y)=\nu_-+\frac{\Delta(1+y)}2-\lambda\Big[\big(\tfrac{1+y}2-\omega\big)^++\big(\tfrac{1-y}2-\omega\big)^+\Big],$$
  with $|f_n(y)-f(y)|\le\lambda/\sqrt n$. For $\Delta>0$:
  - if $\Delta>\lambda$ or $\omega\ge1$, then $y^\ast=1$;
  - if $\Delta<\lambda$ and $\omega<1$, then $y^\ast=|2\omega-1|$, and in each state one leader's load equals its capacity.

  Knife-edges are set-valued:
  - $\Delta=\lambda$ with $\omega<1$: every $y\in[|2\omega-1|,1]$ is optimal;
  - $\Delta=0$ with $\omega<1$: every $y\in[0,|2\omega-1|]$ is optimal.

**Derived quantities for E3 Part A.**

- Under symmetric policies: $p_1(\Lambda_1)=(1+y)/2$, $p_1(\Lambda_2)=(1-y)/2$, $\operatorname{Var}_\Lambda(p)=y^2/4$, and $\mathbb E_\Lambda[p(1-p)]=(1-y^2)/4$.
- Finite-$n$ accuracy in (b): $f(y^\ast)-f(y_n)\le2\lambda/\sqrt n$.
- Away from the knife-edges, $|y_n-y^\ast|\le4\lambda/\big(\sqrt n\,\min\{\Delta,|\Delta-\lambda|\}\big)$. When $\omega\ge1$, replace the minimum by $\Delta$.

### 2.4 Corollary congestion: payments include congestion (`cor:congestion`, l.664)

**Hypotheses.** As in Prop `prop:ic`. Write $G=\Gamma-\sum_i\mathbb E\,C_i(D_i)$, where $\Gamma$ is expected contribution. Let $Q^{-j}=(Q^o,Q^{s\ast}(0,b_{-j}))$ and $Q^\ast=(Q^o,Q^{s\ast}(b))$.

**Statement.**
$$P_j(b)=\sum_{i\ne j}b_i\big[X_i(Q^{-j})-X_i(Q^\ast)\big]+\big[\Gamma(Q^{-j})-\Gamma(Q^\ast)\big]+\sum_{i\in\mathcal N}\Big\{\mathbb E_{Q^\ast}C_i(D_i)-\mathbb E_{Q^{-j}}C_i(D_i)\Big\}.$$
The last sum includes $j$'s own congestion. Under M3, $C\equiv0$ removes it.

**Registered reading for H3c.** Under M4, leader $j$ is sponsored back into a list from which the organic policy removed it only if $b_j\,\Delta X_j$ exceeds the change in the remaining terms. SL-20 computes the right-hand side term by term from the solver output.

---

## 3. Section 6: learning with hidden types and common states

### 3.1 Proposition identify: identification from comparisons (`prop:identify`, l.685)

**Setting.**

- One context, one model and a fixed state under DET, so choices follow a strict ranking of $E(x)\cup\{0\}$.
- The audited family $\mathcal A$ is a set of unlabeled sets $S$, and the choice $w(S)$ is observed for each.
- The graph $\mathcal G_{\mathcal A}$ on $E(x)\cup\{0\}$ has an edge $w(S)\to i$ for every $S\in\mathcal A$ and every $i\in(S\cup\{0\})\setminus\{w(S)\}$.

**Statement.** The choice from $S'$ is identified, meaning it is the same under every strict ranking consistent with the observations, if and only if some alternative in $S'\cup\{0\}$ has a directed path in $\mathcal G_{\mathcal A}$ to every other alternative in $S'\cup\{0\}$.

### 3.2 Corollary design: inclusion frequency is not information (`cor:design`, l.689)

- **(a)** If every audited set containing $j$ also contains an alternative chosen over $j$, then for every $S'$ from which $j$ is chosen, the choice from $S'$ is not identified, however often $j$ was included.
- **(b) Corrected; see §E1.** Let $K\ge2$ and $N_x=|E(x)|$. Any adaptive audit design that identifies, for every strict ranking, the choice from every subset of $E(x)$ with at most $K$ products uses at least
  $$\frac{\log A(N_x)}{\log(K+1)},\qquad A(N_x)=\sum_{k=0}^{N_x}\frac{N_x!}{(N_x-k)!}=\lfloor e\cdot N_x!\rfloor,$$
  audited sets. Here $A(N_x)$ is the number of distinct choice functions, that is, ordered lists of the products ranked above the outside option.

### 3.3 Proposition learning: audit-then-commit (`prop:learning`, l.710)

**Audit design.**

- Cells: a finite set $\mathcal S$ of context–action cells $c=(x,a)$, and a finite menu $\mathcal Q_{\mathrm{fin}}$ of policies supported on $\mathcal S$.
- Audit traffic: in each period, exactly $n_0\ge2$ requests per cell receive action $a$. They are selected without regard to bids or outcomes, and are additional to the $n$ mechanism requests.
- Observations: $Y_{e,c,j}$ is the fraction of cell-$c$ audit requests in period $e$ that end with $j$; $Y^g_{e,c}$ is their mean contribution.

**Estimators** after $h$ periods, with $\iota_c(Q)=\mu(x)Q(a\mid x)$:

| Quantity | Estimator |
|---|---|
| Choice probability | $\hat q_j(c)=h^{-1}\sum_eY_{e,c,j}$ |
| Contribution per cell | $\hat g(c)=h^{-1}\sum_eY^g_{e,c}$ |
| Effective choices | $\hat X_j(Q)=n\sum_c\iota_c\hat q_j(c)$ |
| Contribution | $\hat\Gamma(Q)=n\sum_c\iota_c\hat g(c)$ |
| Quadratic cost | $\hat C_j(Q)=\frac{\lambda_j}{2B_j}\big\{\hat X_j+n(n-1)h^{-1}\sum_e\sum_{c,c'}\iota_c\iota_{c'}V_{e,c,c',j}\big\}$ |

In the cost estimator, $V_{e,c,c',j}=Y_{e,c,j}Y_{e,c',j}$ for $c\ne c'$ and $V_{e,c,c,j}=Y_{e,c,j}(n_0Y_{e,c,j}-1)/(n_0-1)$.

The plug-in mechanism maximises $\hat W_b=\sum_jb_j\hat X_j+\hat\Gamma-\sum_j\hat C_j$ over $\mathcal Q_{\mathrm{fin}}$, with a fixed tie-breaking rule. It charges eq. `eq:payment` with $\hat X_j$ in place of $X_j$.

**Hypotheses.**

- CI, VAL and COST, with quadratic costs.
- $\mu$ is known.
- The states of different periods are i.i.d. draws from $F_\Lambda$.
- $\bar g=\max_{x,j}g(x,j)$ and $\varrho\in(0,1)$.

**Statement.**

- **(a)** $\hat q_j(c)$ is unbiased for $\bar q_j(c)$, with
  $$\operatorname{Var}\hat q_j(c)=\frac1h\Big\{\operatorname{Var}_\Lambda q_j(c\mid\Lambda)+\frac1{n_0}\mathbb E_\Lambda\big[q_j(1-q_j)\big]\Big\}.$$
  $\hat\Gamma(Q)$ and $\hat C_j(Q)$ are unbiased for every $Q$ supported on $\mathcal S$.
- **(b)** With probability at least $1-3\varrho$, simultaneously for all $b$, $\sup_{Q\in\mathcal Q_{\mathrm{fin}}}|\hat W_b(Q)-W_b(Q)|\le\varepsilon_h(b)$, where
  $$\varepsilon_h(b)=n\Big(\sum_jb_j+\sum_j\frac{\lambda_j}{2B_j}\Big)\epsilon_1+n\bar g\,\epsilon_g+n(n-1)\sum_j\frac{\lambda_j}{2B_j}\epsilon_2$$
  and
  $$\epsilon_1=\sqrt{\frac{\log(2N|\mathcal S|/\varrho)}{2h}},\quad \epsilon_g=\sqrt{\frac{\log(2|\mathcal S|/\varrho)}{2h}},\quad \epsilon_2=\sqrt{\frac{2\log(2N|\mathcal Q_{\mathrm{fin}}|/\varrho)}{h}}.$$
- **(c)** On that event, $\max_{\mathcal Q_{\mathrm{fin}}}W_b-W_b(\hat Q(b))\le2\varepsilon_h(b)$.
- **(d)** On that event, for every $j$ and $b_{-j}$, no report raises $j$'s expected utility by more than $2\theta_jn\epsilon_1$ above truthful reporting, and truthful reporting yields expected utility at least $-\theta_jn\epsilon_1$.

**Closed forms for E4.** Since $\hat q$ is unbiased, $\mathrm{RMSE}(\hat q_j(c))=\sqrt{\operatorname{Var}\hat q_j(c)}$ exactly. So:

- the log–log slope of RMSE on $h$ is exactly $-1/2$;
- in $n_0$, $\mathrm{RMSE}(n_0)/\mathrm{RMSE}(\infty)=\sqrt{1+\mathbb E_\Lambda[q(1-q)]/(n_0\operatorname{Var}_\Lambda q)}$, which falls to a floor when $\operatorname{Var}_\Lambda q>0$ and is proportional to $n_0^{-1/2}$ when $\operatorname{Var}_\Lambda q=0$ (§E3);
- H4c's bound is $2\theta_jn\epsilon_1$, evaluated at the registered $\varrho$.

### 3.4 Corollary cumulative: cumulative loss (`cor:cumulative`, l.730)

**Statement.** Suppose that in each period $e=2,\dots,h$ the platform runs the plug-in mechanism with estimates from periods $1..e-1$ and failure probability $\varrho_e=\varrho/(e(e-1))$, and that bids are fixed at $b$. With probability at least $1-3\varrho$:
$$\sum_{e=2}^h\Big[\max_{\mathcal Q_{\mathrm{fin}}}W_b-W_b(\hat Q_e(b))\Big]\le c_0\,n\sqrt{h\log\big(hN|\mathcal S||\mathcal Q_{\mathrm{fin}}|/\varrho\big)},$$
with $c_0\propto\sum_jb_j+\bar g+\sum_j\lambda_jn/B_j$. The loss excludes the value and cost of audit requests.

**Scope.** This is a worst-case upper bound, not a growth rate; see §E2.

---

## E. Corrections and registration notes

These are owner decisions. Where they differ from the paper or from tickets.md §1.1, the registration and the code follow this section. Paper fixes are listed for the next draft.

**E1 · `cor:design`(b) uses the wrong count (paper error).** The proof says choices from all subsets of size at most $K$ "determine the full ranking of $E(x)\cup\{0\}$". They do not: two rankings that differ only in the order of products ranked *below* the outside option give the same choice from every set. A design therefore needs to separate only $A(N_x)=\lfloor e\cdot N_x!\rfloor$ classes, not $(N_x+1)!$, so the paper's bound is not established. Checked by enumeration for $N_x\le7$. Example: at $N_x=5$, $K=4$, the paper's bound gives at least 5 audited sets, the corrected bound gives at least 4. **Fix:** replace $(N_x+1)!$ with $\lfloor e\cdot N_x!\rfloor$ in the statement, and replace "the full ranking" with "the ordered set of products ranked above the outside option" in the proof. H4a uses the corrected bound.

**E2 · H4d wording.** `cor:cumulative` bounds cumulative loss from above by order $\sqrt{h\log h}$. It does not say the loss grows at that rate. With a finite menu and a positive gap between the best and second-best policy, the plug-in picks the best policy with probability tending to 1, and cumulative loss stays bounded. "Cumulative loss ∝ √h" (tickets.md §1.1) is therefore not a consequence of the theory. H4d is registered as "the growth exponent does not exceed the bound's"; the rule is in A12 §4.

**E3 · H4b wording.** By `prop:learning`(a), RMSE in $n_0$ is not flat: it falls to the floor $\sqrt{\operatorname{Var}_\Lambda q/h}$. H4b is registered against the closed form in §3.3. "Flat beyond small values" means within 5% of the floor, which holds once $n_0\ge10\,\mathbb E[q(1-q)]/\operatorname{Var}_\Lambda q$.

**E4 · H3a "close to the bound".** The bound $\Theta_j^2/4$ is attained only with two equally likely state outcomes for the contested mass (§2.1). E3 Part B models states as model versions with equal probability, so the attainable ratio is 1. The H3a threshold is in A12 §4.

**E5 · H2a needs two definitions the paper leaves implicit.**

1. **Payments by traffic type.** Because eq. `eq:payment` is linear in $X_j=X^H_j+X^A_j$:
   $$P_j=P^H_j+P^A_j,\qquad P^\tau_j(b)=b_jX^\tau_j(b)-\int_0^{b_j}X^\tau_j(z,b_{-j})\,dz.$$
   Each component can be negative; only their sum is guaranteed to be nonnegative.
2. **Screen.** `prop:rents`(c) requires the labelled-favourite condition. With E1-calibrated label penalties, it can fail even under an agent-aware organic policy: a sponsored favourite then loses to its rival, so bids change agent outcomes. H2a's zero-payment prediction is tested only in markets that pass the screen. Markets that fail it are reported separately as boundary cases.

**E6 · `prop:threshold` in E3 Part A.**

- Quadratic case with $\lambda>0$: the optimum is unique and symmetric. Moving mass between $\{1,3\}$ and $\{2,3\}$ leaves expected net value unchanged and strictly raises the convex cost. Compare $y$ and the objective.
- Overflow case: compare objective values. Compare $y$ only away from the knife-edges in §2.3, and exclude $\Delta\in\{0,\lambda\}$ from the grid.

**E7 · A2 symbol definitions (tickets.md §5) corrected to the paper.**

| Symbol | Paper definition (governs) | tickets.md A2 said |
|---|---|---|
| $N_x$ | $\lvert E(x)\rvert$, the number of eligible products in context $x$ | number of contexts (wrong) |
| $n_0$ | audit requests per **cell** per period | audit requests per period (incomplete) |
| $\chi_m$ | probability that a model-$m$ agent requests the page of **each displayed** product, independently of its choice (Ass. RET). H1e estimates it from non-chosen products. | probability for products it does not choose (narrower; consistent with RET) |
| $F(x)$ | favourites in $E(x)$ of all positive-probability types $(m,\Lambda)$; the outside option is excluded | consistent |
| $\xi$ | guardrail slack in `prop:gating` | consistent |
| $\varepsilon_1$ | written $\epsilon_1$ in `prop:learning`(b) | consistent |
| $h$ | number of audit periods | consistent |
| agent-aware | Def. `def:aware` (§1.1) | consistent |
| set-blind M3 | §1.6 | consistent |
| gated | Operational definition for E2 markets with several types: no favourite in $F(x)$ appears organically, and, where the list allows, each favourite's second-ranked product for that type is among the first $K-K^s$ organic products. The remaining positions follow the human-optimised order. With one type, this is design (b) of `prop:gating`. | undefined |
