# Simulated backend: model and validation note

SL-06. Code: `backends/simulated.py`; parameters `core/agent_params.py`
(`contracts/storelab/agent_params.schema.json`, shared with SL-15); tests
`tests/features/storelab/test_simulated.py` and the reusable protocol checks in
`tests/features/storelab/protocol_conformance.py`.

## Model

**Agent model m** (`v1.tex` l.289). For a presentation with displayed set S(a):

s_j = v_j^m(x) + shift_j(Λ) + δ_m ℓ_j + β_{m,k_j},
q_j = exp(s_j/σ_m) / D,  q_0 = exp(v₀^m/σ_m) / D,  D = exp(v₀^m/σ_m) + Σ_{i∈S(a)} exp(s_i/σ_m).

Λ is one of the model's discrete states (probability, per-product shifts); it is
drawn once per period (`draw_state`, keyed stream `("state", model_key, …)`) and
passed to every trial of that period. `choice_probs` returns q(x, a | Λ) or, with
`average=True`, q̄(x, a) = E_Λ q(x, a | Λ), exactly (numerically stable softmax).

**Page requests** (A2): under `browser_tool`, each displayed product that is not
chosen is requested with probability χ_m, in position order, at most `max_turns` − 1
times. Other interfaces have no pages, so no requests.

**Human model** (Ass. human, l.236): q_j^H = γ_{k_j} w_j(x, ℓ_j), q_0^H = 1 − Σ q_j^H,
with nonincreasing γ (A11) and per-label product weights; parameters that make
Σ q_j^H > 1 are rejected. The legacy cascade human stays in `core/legacy_env.py`
(SL-04), used by the legacy runner.

**Sampling**: Gumbel-max. A trial draws from `stream(seed, "agent", trial_key)`
(`"human"` for the human model): one Gumbel variate per alternative (outside first,
then candidates in position order), argmax(logit + Gumbel) is the choice, then one
uniform per candidate for page requests. Draws depend only on (seed, trial_key), so
execution order and resumption cannot change any trial.

**Actions and outcome**: `act` returns `RequestPage`… then `Select` or `Leave`, the
types the LLM backends emit; `decide` returns a `TrialOutcome` with status `choice` or
`outside`, the page requests, `turns` = number of actions, `attempts` = 1, backend
`simulated` (or `human_model`) and model `simulated:<model_key>`. A simulated
decision-maker cannot produce `invalid`, `refusal` or `error`.

**Parameter sources** (`params_from_config`): `stipulated` gives every configured
model key the config's σ, δ, β, v₀ and χ, a single state, and baseline scores
v_j^m(x) = u(x, j) from SL-05 (the stipulated agent ranks by the ground-truth
utility); humans under `humans.model: position_weights` get those γ and product
weights 1. `estimated` loads the config's `params_file`, refusing a file whose
sha256 differs from the pinned one. SL-15 estimates are identified only up to scale,
so they come with σ = 1 (scores, δ and β are v/σ, δ/σ, β/σ).

**Load model**: `load_moments` implements Lemma load,
E[D] = n E_Λ p, Var(D) = n E_Λ[p(1 − p)] + n² Var_Λ(p); `policy_choice_prob` gives
p_j^Q(Λ) for a policy of (observation, probability) pairs; `simulate_loads` simulates
periods (draw Λ, a set per request, a Gumbel-max choice per request).

## Validation (logs in `agent-workflow/runs/SL-06/logs/validation-evidence.log`)

**Analytic checks.** Probabilities sum to 1 including the outside option (200
random sets, σ from 1e-3 to 20, to 1e-12). They equal the closed-form logit to 1e-14.
Lemma small-noise: with σ = 1e-4 the argmax (a product, or the outside option) gets
probability ≥ 1 − 1e-9 (1.0 in floating point here); a gap of 0.01 is enough.

**Example diversify** (`v1.tex` l.388; σ = 1e-3, two equally likely states with
v_A > v_B > v_C and v_B > v_A > v_C, outside below C, n = 1,000). Lemma load gives
the stated variances exactly; 20,000 simulated periods agree within 3 Monte Carlo
standard errors (SE of the sample variance from its 2nd and 4th moments):

| Policy | p_A(Λ₁), p_A(Λ₂) | Var(D_A), Lemma load | Simulated Var | SE | \|z\| |
|---|---|---|---|---|---|
| every request {A,B} | 1, 0 | 250,000 | 250,012.3 | 17.9 | 0.69 |
| {A,C} or {B,C} | ½, ½ | 250 | 251.5 | 2.5 | 0.59 |
| {A,B} or {A,C} | 1, ½ | 62,625 | 62,674.7 | 40.9 | 1.22 |

**Sampling vs `choice_probs`** (χ² goodness of fit, 20,000 keyed trials of one
8-product presentation with labels, df = 8, fixed seeds; pass if p > 0.001):
agent σ = 1: p = 0.129; σ = 0.3: p = 0.837; two-state model, state a: p = 0.674,
state b: p = 0.322; human model: p = 0.963. Page-request rate within 4 SE of χ = 0.3.

**Seeds and order.** On the SL-05 fixture E1 design (4 tasks, 252 sets, 3
repetitions, 3 interfaces, 2 models: 4,536 trials), a second run with the same seed
gives identical outcomes; seed + 1 changes over 10% of the choices; three random
execution orders and a run interrupted at a random trial and resumed in another order
give the same record-set hash.

**Protocol conformance.** The simulated agent (all three interfaces), the human
model and the local LLM backend (stubbed replies) pass the same checks: protocol
`Action` types only, page requests before one final action and only under
`browser_tool`, ids among the candidates, an outcome consistent with the actions that
fits a `TrialRecord`.
