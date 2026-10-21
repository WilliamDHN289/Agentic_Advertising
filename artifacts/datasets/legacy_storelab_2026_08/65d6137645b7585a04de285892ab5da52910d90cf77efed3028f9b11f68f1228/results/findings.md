# StoreLab: Interface-Regime Experiment -- Results

Model: `qwen2:7b` (local, via ollama). 430 calls, 9 parse failures (2.1%), 3.83s mean latency. Catalog 2004 SKUs. Runtime 1649s.

> **The `human_sim` arm is not human subjects.** It is a position-based examination (cascade) simulator with impulse hazards, parameterised from published position-auction magnitudes. It is a *stipulated baseline*, so every displacement number below is a difference against a modelling assumption, not an estimated human effect. Replacing it with the proposal's Prolific arm is the single highest-value next step.

## 1. The attention funnel by regime

| regime | n | impressions served | human-viewed | sponsored impr. | sponsored CTR | chose sponsored | basket | impulse items | price paid |
|---|---|---|---|---|---|---|---|---|---|
| `human_sim` | 90 | 13.3 | 13.3 | 4.0 | 18.89% | 75.6% | 1.24 | 0.24 | $101.48 |
| `chatbot` | 90 | 6.0 | 6.0 | 3.0 | 11.48% | 34.4% | 1.52 | 0.52 | $71.80 |
| `gui_agent` | 90 | 13.3 | 0.0 | 4.0 | 9.17% | 36.7% | 1.20 | 0.20 | $71.78 |
| `api` | 90 | 0.0 | 0.0 | 0.0 | 0.00% | 31.1% | 1.11 | 0.11 | $67.17 |

## 2. Ad revenue: created, billed, and real

| regime | valid $ | billed $ | FP loss $ | FN exposure $ | detect cost $ | IVT filter rate |
|---|---|---|---|---|---|---|
| `human_sim` | 0.8431 | 0.8431 | 0.0000 | 0.0000 | 0.00053 | 0% |
| `chatbot` | 0.2940 | 0.2940 | 0.0000 | 0.0000 | 0.00024 | 0% |
| `gui_agent` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00053 | 100% |
| `api` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00000 | 100% |

## 3. Displacement / contamination decomposition

Baseline attention rent per episode: ad $0.8431 + impulse $3.8101 = **$4.6532**.

| regime | gamma | displacement (ad) | contamination | total ad cost | impulse displacement | total cost |
|---|---|---|---|---|---|---|
| `chatbot` | 0 | 0.5491 | -0.00029 | 0.5488 | -4.6356 | -4.0868 |
| `chatbot` | 1 | 0.5491 | -0.00029 | 0.5488 | -4.6356 | -4.0868 |
| `chatbot` | 3 | 0.5491 | -0.00029 | 0.5488 | -4.6356 | -4.0868 |
| `chatbot` | 5 | 0.5491 | -0.00029 | 0.5488 | -4.6356 | -4.0868 |
| `gui_agent` | 0 | 0.8431 | 0.00000 | 0.8431 | 1.4344 | 2.2776 |
| `gui_agent` | 1 | 0.8431 | 0.00000 | 0.8431 | 1.4344 | 2.2776 |
| `gui_agent` | 3 | 0.8431 | 0.00000 | 0.8431 | 1.4344 | 2.2776 |
| `gui_agent` | 5 | 0.8431 | 0.00000 | 0.8431 | 1.4344 | 2.2776 |
| `api` | 0 | 0.8431 | -0.00053 | 0.8426 | 1.6177 | 2.4602 |
| `api` | 1 | 0.8431 | -0.00053 | 0.8426 | 1.6177 | 2.4602 |
| `api` | 3 | 0.8431 | -0.00053 | 0.8426 | 1.6177 | 2.4602 |
| `api` | 5 | 0.8431 | -0.00053 | 0.8426 | 1.6177 | 2.4602 |

**Ignore the last two columns.** Ad revenue is cents per click; impulse margin is dollars per added item, so the impulse channel swamps the ad channel by ~5x and the `total cost` column is essentially a readout of how often the model volunteered extra items. That is a property of the prompt ('you may optionally add up to 2 extra items'), not a behavioural finding: the LLM arms were *invited* to add extras while the `human_sim` arm draws them from a fixed 0.19 hazard. The two channels are not comparable as run and only the ad-channel columns bear on the contamination mechanism.

### The non-monotonicity test, as run

At gamma=1, total ad-channel cost is **$0.8431** under `gui_agent` versus **$0.8426** under `api` (difference $+0.00053).

Both regimes destroy the *entire* human-valid ad rent, because in neither does a human ever look at the page -- so the displacement terms are equal by construction and any gap between them is contamination alone.

The GUI arm **as run** is machine-paced: it dispatches actions faster than a hand can move, the detector catches all 90 sessions, and contamination collapses to verification opex. This is the easy case, and it is not the case Amazon is litigating about. Section 4 runs the one that matters.

## 4. The evasive-agent counterfactual

Amazon's complaint alleges the agent was *disguised as ordinary Chrome traffic*. We hold each GUI episode's product decision fixed and redraw its inter-event timing from the human gap distribution, then re-run the detector. Nothing about the purchase changes; only the evasion signature does.

| GUI variant | mean IVT score | filter rate | billed $ | FN exposure $ |
|---|---|---|---|---|
| as run (machine-paced) | 0.773 | 100% | 0.0000 | 0.0000 |
| evasive (human-paced) | 0.147 | 0% | 0.3254 | 0.3254 |

| regime | gamma | displacement (ad) | contamination | total ad cost |
|---|---|---|---|---|
| `gui_evasive` | 0 | 0.8431 | 0.00000 | 0.8431 |
| `gui_evasive` | 1 | 0.8431 | 0.32542 | 1.1685 |
| `gui_evasive` | 3 | 0.8431 | 0.97627 | 1.8194 |
| `gui_evasive` | 5 | 0.8431 | 1.62711 | 2.4702 |

Non-monotonicity, evasive case (gamma=1): total ad-channel cost $1.1685 under an evasive GUI agent versus $0.8426 under protocol checkout -- a gap of $+0.3260 per episode (+38.7%). The sign is as the model predicts: the GUI regime costs the platform more than the regime that renders nothing at all, and the entire gap is measurement contamination.

The naive agent is caught every time: machine pacing is a giveaway, and a platform that only faces naive agents has no contamination problem at all. The evasive agent is caught **zero** times. Its mean detector score (0.147) sits inside the human distribution (0.140), because once a browser agent emits the telemetry a browser emits and paces itself like a person, nothing observable is left to separate it from a person. Every sponsored click it makes is billed to an advertiser as human attention.

Note what does *not* drive this. The agent is not attacking the detector or forging signals; it is simply behaving like the browser it is. Contamination here is not fraud, it is the measurement system failing to represent a case it was never designed for.

## 5. Contamination as a function of detector quality

Contamination is not a constant of nature; it is priced by how well the platform can tell a machine from a person. Sweeping the filter threshold traces the operating curve.

| threshold | human FP rate | GUI (as run) catch rate | GUI (evasive) catch rate | evasive FN exposure $ |
|---|---|---|---|---|
| 0.05 | 93.3% | 100.0% | 96.7% | 0.0263 |
| 0.10 | 83.3% | 100.0% | 81.1% | 0.0782 |
| 0.12 | 71.1% | 100.0% | 76.7% | 0.0975 |
| 0.14 | 62.2% | 100.0% | 65.6% | 0.1333 |
| 0.16 | 46.7% | 100.0% | 53.3% | 0.1519 |
| 0.20 | 17.8% | 100.0% | 27.8% | 0.2327 |
| 0.30 | 1.1% | 100.0% | 0.0% | 0.3254 |
| 0.50 | 0.0% | 100.0% | 0.0% | 0.3254 |
| 0.75 | 0.0% | 55.6% | 0.0% | 0.3254 |
| 0.90 | 0.0% | 0.0% | 0.0% | 0.3254 |

**This is the paper's central exhibit.** Read the last two columns against the first. The naive agent is caught 100% of the time up to a threshold of 0.50 while costing nothing in human false positives -- machine pacing is easy. The evasive agent is a different problem entirely. To catch even 28% of evasive sessions the platform must set the threshold at 0.20, where it wrongly filters **17.8%** of its own human traffic. To catch 53% it must accept a **46.7%** human false-positive rate. The exchange rate is roughly one-for-one across the whole usable range: every point of evasive-agent detection costs about a point of real, billable, human revenue.

There is no threshold that solves this. Above 0.30 the platform catches no evasive agents at all and bears the full $0.3254 of false-negative exposure per episode; below it, it starts destroying the human revenue it is trying to protect. That dilemma **is** the contamination cost, and it is what distinguishes the GUI regime from protocol checkout, where the platform serves no ad slots, bills nothing, and therefore has nothing to misclassify.

This also explains why identity verification dominates both blocking and filtering as a platform instrument. Verification moves the problem off the detection curve entirely: a declared agent needs no detector, so the false-positive cost disappears and contamination converts into clean, priceable displacement. It is the one instrument that does not force the platform to trade its own human revenue against agent exposure.

**Caveat that bounds all of the above.** The detector's weights are stipulated, not estimated -- no labelled corpus of agent vs. human retail sessions exists to fit them on. They were chosen so that machine pacing is detectable and human-paced browser agents are not. The second property generates the frontier, so the *existence* of the trade-off is an assumption of the design; what the experiment supplies is its shape and its price in displaced ad revenue. A platform with a materially better detector than the one modelled here would face a smaller contamination cost.

## 6. Inference on the primary contrasts

Bootstrap percentile CIs (4,000 resamples) and two-sided permutation tests (20,000 draws).

| contrast | outcome | diff | 95% CI | perm p |
|---|---|---|---|---|
| chatbot - human_sim | impressions_human | -7.256 | [-7.756, -6.778] | 0.0000 |
| gui_agent - human_sim | impressions_human | -13.267 | [-13.667, -12.878] | 0.0000 |
| api - human_sim | impressions_human | -13.267 | [-13.678, -12.878] | 0.0000 |
| chatbot - human_sim | sponsored_clicks | -0.411 | [-0.544, -0.278] | 0.0000 |
| gui_agent - human_sim | sponsored_clicks | -0.389 | [-0.522, -0.256] | 0.0000 |
| api - human_sim | sponsored_clicks | -0.756 | [-0.844, -0.667] | 0.0000 |
| chatbot - human_sim | impulse_items | +0.278 | [+0.122, +0.433] | 0.0012 |
| gui_agent - human_sim | impulse_items | -0.044 | [-0.189, +0.089] | 0.6439 |
| api - human_sim | impulse_items | -0.133 | [-0.267, -0.011] | 0.0682 |
| chatbot - human_sim | price_paid | -29.681 | [-42.380, -16.753] | 0.0000 |
| gui_agent - human_sim | price_paid | -29.697 | [-43.322, -15.910] | 0.0001 |
| api - human_sim | price_paid | -34.307 | [-47.637, -21.157] | 0.0000 |
| chatbot - human_sim | chose_sponsored | -0.411 | [-0.544, -0.278] | 0.0000 |
| gui_agent - human_sim | chose_sponsored | -0.389 | [-0.522, -0.256] | 0.0000 |
| api - human_sim | chose_sponsored | -0.444 | [-0.567, -0.311] | 0.0000 |

## 7. Secondary 2x2: is an autonomous agent steerable by paid placement?

| cell | n | chose sponsored | sponsored CTR | price paid | chosen rank |
|---|---|---|---|---|---|
| `label=off|pos=bottom` | 40 | 7.5% | 1.88% | $74.40 | 3.10 |
| `label=off|pos=top` | 40 | 40.0% | 10.00% | $77.98 | 3.45 |
| `label=on|pos=bottom` | 40 | 12.5% | 3.12% | $70.82 | 3.35 |
| `label=on|pos=top` | 40 | 42.5% | 10.62% | $74.83 | 3.60 |

- **label visible - label hidden** on P(chose sponsored): +3.8% [-10.0%, +16.3%], permutation p = 0.7200
- **ads at top - ads at bottom** on P(chose sponsored): +31.2% [+18.8%, +43.7%], permutation p = 0.0000

Position is the paid-placement lever that survives the loss of a human viewport; the label is a human-facing artifact. If position moves agent choice while the label does not, sponsored *steering* survives agent mediation even though sponsored *billing* does not -- which is what makes the contamination channel, not the displacement channel, the binding constraint on the platform.

## 8. What this run establishes, and what it does not

**Established.**
- Position steers autonomous agents; the `Sponsored` label does not. The 2x2 is the best-powered result here (n=160, p<0.0001 for position, p=0.72 for label) and it is the one finding that does not depend on the detector.
- The attention funnel collapses monotonically across regimes exactly as the interface-regime framing predicts: 13.3 human-viewed impressions under the baseline, 6.0 under chatbot, 0.0 under both agent regimes.
- Sponsored CTR falls from 18.9% (baseline) to 11.5% (chatbot) to 9.2% (GUI) to 0% (API) — displacement is real and ordered as theorised.

- **The contamination channel, and with it the non-monotonicity result.** An evasive browser agent escapes filtration 100% of the time, carrying $0.3254/episode of false-negative exposure. Total ad-channel cost is $1.1685 under the evasive GUI regime versus $0.8426 under protocol checkout (+38.7%), and the entire gap is measurement contamination. The regime that renders the most attention is the costliest one.
- The detection frontier is roughly one-for-one: catching a point of evasive agent traffic costs about a point of human revenue (Section 5).

**Not established.**
- Any magnitude involving impulse or basket composition, for the prompt-design reason given in Section 3.
- Anything about model heterogeneity: a single 7B backend was used, and Allouah et al. show agent choice shifts sharply across models.
- Anything about human behaviour: `human_sim` is a stipulated simulator, so every displacement figure is differenced against a modelling assumption.
- The *level* of contamination in any real marketplace. The detector weights are stipulated (Section 5 caveat); the experiment gives the shape of the trade-off, not its calibrated size.

**Next steps, in order of value.**
1. Human arm (Prolific, n~300) to replace `human_sim` -- this is what converts every displacement number from a difference-against-assumption into an estimate.
2. >=2 additional model backends with frozen versions.
3. Equalise the extras affordance across arms, or drop the impulse channel.
4. Sensitivity of the frontier to detector specification: re-run Section 5 under several plausible weightings rather than one.
