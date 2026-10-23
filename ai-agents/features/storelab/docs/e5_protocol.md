# E5 field protocol: validating the advertising unit

**Protocol version: `e5_protocol@1`** (SL-22). This version is cited by the E5
pre-registration (`docs/preregistration_E5.md`) with this file's sha256. Any change after
filing is a dated deviation (§11). The registered hypotheses, metrics and decision rules
are H5a and H5b in `docs/claims_register.yaml`, which governs where it and this text
differ. The analysis code is `features/storelab/analysis/e5.py`, and its synthetic
validation is `docs/e5_validation.md`. The partner and the sample size (A10) are to be
determined; partner integration is out of scope.

## 1. Purpose

`ass:value` (paper l.344) values an advertiser's outcome only through the number of
effective choices, at a value per choice that does not depend on the candidate set. The
sandbox cannot test this, because its purchases are not real. E5 tests it with a partner
that records agent-initiated selections, purchases and returns:

- **H5a (invariance).** Purchases net of returns per effective choice are equal, within
  ±0.10 on the log ratio, between requests randomised to strong-competitor and to
  weak-competitor candidate sets.
- **H5b (surrogacy).** The qualifying event predicts purchases better than clicks. This
  should hold in both arms and, where traffic type is identified, for human and for
  agent traffic.

## 2. Population and focal products

- **Requests.** Shopping requests served by the partner's product listing or agent
  interface in which at least one focal product is eligible. A request is the unit of
  randomisation and of analysis.
- **Focal products.** A subset of products is fixed before launch. A product is
  eligible as focal if, in the partner's catalog, it has at least two strong and two
  weak competitors (§3) for its query class. The list is frozen and its sha256 is
  recorded with the registration. A request concerns one focal product: the
  highest-ranked eligible one in the partner's organic order.

## 3. Arms

Both arms show a candidate set of the partner's usual size K. The focal product is in
the same position, with the same label (organic, unsponsored), in both arms. The other
K − 1 positions are filled from competitors in the same query class, as follows:

- **Strong-competitor arm (`strong`).** The top K − 1 competitors by the partner's
  relevance or quality score.
- **Weak-competitor arm (`weak`).** K − 1 competitors drawn from below the median
  score. They must remain eligible (in stock, meeting the request's hard
  requirements).

Arms change only which competitors the focal product faces. Prices, labels and the
focal product's own presentation are unchanged.

## 4. Randomisation

Each eligible request is assigned to `strong` with probability π = 0.5, independently
across requests. The rule is HMAC-SHA256(secret salt, partner request id) < π · 2²⁵⁶.
The salt is generated before launch and escrowed with the partner, and the data
carry only the hashed request id. The assignment, with its probability, the candidate
set and the traffic type, is logged *before* the set is rendered (`assignment` event,
§7). The assignment mechanism used by the analysis is exactly this independent
Bernoulli(π). No other randomisation (e.g. of positions) runs on focal requests during
the study.

## 5. Events, outcomes and windows

Windows run from the assignment time.

| Quantity | Definition |
|---|---|
| Qualifying event (effective choice) | A confirmed selection of the focal product, or a valid add-to-cart instruction for it, within 24 hours (the paper's effective choice, l.325; claims register `effective_choice.per_interface.field_E5`). |
| Click | A page request of the focal product, by a human or an agent, at any time in the request's session. |
| Purchase | An order of the focal product linked to the request, within 7 days. |
| Return | A return of that order within 30 days of the purchase. |
| Net purchase | A purchase with some quantity not returned. |
| Traffic type | `agent` for requests through the partner's verifiable agent interface (agent checkout or agent API); `human` for web and app sessions; `unknown` otherwise. |

## 6. Horizon and exclusions

- **Horizon.** Enrolment runs until the registered sample size of eligible requests is
  reached, or for at most 12 weeks. Outcome data are collected until 37 days after the
  last enrolled request (7 days for purchases, then 30 for returns).
- **Exclusions,** decided from logged data only, and the same in both arms:
  - requests flagged by the partner's invalid-traffic filter (`partner_traffic_filter`);
  - internal and test traffic (`internal_test`);
  - requests whose focal product became unavailable before rendering
    (`focal_unavailable`);
  - requests with no assignment record, or with more than one;
  - events timestamped before their request's assignment;
  - duplicate events (by event id).

  The analysis counts each exclusion by reason (`request_table`), and both arms are
  reported.

## 7. Data contract

- **Schema.** `contracts/storelab/e5_event.schema.json` (generated from
  `analysis/e5.py`), with one JSON object per event:
  - `assignment`: arm, π, candidate set, traffic type, exclusion;
  - `click`;
  - `qualifying_event`: confirmed selection or add-to-cart;
  - `purchase` and `return`: order id, quantity.
- **Identifiers.** Request, event and order ids are sha256 digests of the partner's ids
  with the escrowed salt; product and set ids are catalog ids.
- **No personal data.** No field holds personal data. The schema forbids additional
  properties, and `pii_fields` scans the schema and every delivery for field names
  denoting personal data (email, names, addresses, phone, IP, birth date,
  card or bank numbers) and for values that look like an email, phone number or IP
  address.
- **Delivery.** Deliveries are rejected if any validation or scan fails. The
  synthetic fixture is `tests/features/storelab/fixtures/e5_events.jsonl`.

## 8. Analysis

As registered, implemented in `analysis/e5.py`. Results go to the registered decision
rules through `analysis/decide.py`.

**H5a.** The estimate is τ̂ = log R̂_strong − log R̂_weak, where R̂_a = Σ net purchases /
Σ qualifying events in arm a.

- **Randomisation intervals.** Inverted Fisher randomisation tests of H0(τ₀). Under
  H0(τ₀), dividing the strong arm's net outcomes by e^τ₀ removes the effect.
- **Statistic.** The studentised log-ratio difference, with a delta-method standard
  error, which keeps the tests valid when only the ratios are equal.
- **Null distribution.** 10,000 draws of the Bernoulli(π) assignment.
- **Labels.** TOST at the margin ±0.10. *Supported* if the 90% interval lies inside
  (−0.10, 0.10), i.e. both one-sided p-values are ≤ 0.05 (`p_support`). *Not
  supported* if the 95% interval lies entirely outside [−0.10, 0.10] (`p_against`).
  *Inconclusive* otherwise.

**H5b.** The cells are arm × traffic type for identified types, and arm alone if no
type is identified.

- **Predictions.** Purchase is predicted from the qualifying event and, separately,
  from the click. Each predictor is the purchase rate given the indicator, with 0.5
  added to each count, cross-fitted over 5 folds keyed by request id.
- **Metric.** The out-of-sample log-loss difference (event − click), with AUC reported.
- **Interval.** A percentile bootstrap (2,000 draws) that resamples requests within
  each cell, and hence within an arm.
- **Labels.** One-sided p-values against 0, Holm across cells, as registered.

## 9. Power and sample size

**H5a.** The power at true τ is Φ((m − τ)/se − z₀.₉₅) + Φ((m + τ)/se − z₀.₉₅) − 1, with
m = 0.10 and

se² = (1 − r_s) / (n π c_s r_s) + (1 − r_w) / (n (1 − π) c_w r_w).

Here c_a is the arm's qualifying-event rate per request, and r_a is net purchases per
qualifying event. The functions are `h5a_power` and `h5a_required_n`. The registered
sample size is the n for 80% power at τ = 0 with the partner's baselines (c_w, c_s,
r), written into the pre-registration before filing. With the validation's stand-in
baselines (c_w = 0.30, c_s = 0.22, r = 0.495, π = 0.5), n = 13,768 requests.

**H5b.** `h5b_power` gives the normal-approximation power of a cell from the joint rates of
(event, click, purchase), Bonferroni over the cells (conservative for Holm).

## 10. Validation

`python -m features.storelab.analysis.e5_validation` (`docs/e5_validation.md`) checks the
analysis on 1,000 synthetic datasets per condition:
- the size of the Fisher test of τ = 0 under the sharp and the weak null;
- the size of the TOST at both margins;
- the power at the designed effect against §9;
- H5b's size and power;
- the contract and the PII scan.

Every size must lie in [0.03, 0.07], and the power within ±5 points of the target.

## 11. Deviations

Changes after filing are dated amendments, with reasons, in the E5 registration. An
analysis not listed here is exploratory.
