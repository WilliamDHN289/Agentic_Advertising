"""Mechanism and allocation solvers (SL-16; ``requirements/paper.tex`` l.184–512, A1).

- ``market``: the market (contexts, organic policy, advertisers, buyers, states,
  contributions, match quality, costs, guardrail, value distributions) and its
  action table of exact choice probabilities.
- ``allocation``: eq. alloc and prop:profit as convex programs (M3, set-blind M3,
  M4, profit rule).
- ``payments``: the externality form of Prop ic(c) and eq. payment over the
  breakpoints of X_j(z).
- ``gsp``: the click benchmarks M1 and M2 with their two bidding rules.
- ``costs``, ``values``: expected operating costs and (ironed) virtual values.
- ``solver``: the pinned solvers and the strict status rule; ``evaluate``: one entry
  point for all six mechanisms, with solver exclusions counted.

The validation study is ``features.storelab.analysis.mechanism_validation``
(docs/mechanism_validation.md).
"""
