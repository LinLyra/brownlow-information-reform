# SSAC27 information-reform analysis protocol

Frozen 2026-09-29 before inspecting any information-reform coefficient or bootstrap comparison.

## Question and estimand

Did the mapping from match performance information to Brownlow votes change in 2026, when field umpires were first supplied an approved Champion Data statistical page? This is a structural-change analysis around a league-wide information intervention, not a causal natural experiment: there is one post-reform season and no untreated AFL control group.

## Information sets

The AFL's official announcement specifies: kicks, handballs, disposals, marks, contested marks, tackles, goals, behinds, goal assists, score involvements, clearances, contested possessions, hitouts, kick-ins, intercept marks, intercept possessions, and spoils.

Repository mapping (`VISIBLE_STATS`):

`kicks`, `handballs`, `disposals`, `marks`, `contested_marks`, `tackles`, `goals`, `behinds`, `goal_assists`, `score_involvements`, `clearances`, `contested_possessions`, `hitouts`, `intercept_marks`, `intercepts`, `spoils`.

`kick-ins` is not present and will not be proxied or invented. `intercepts` is the repository mapping for official "intercept possessions". The resulting primary visible set contains 16 of the 17 announced categories.

The non-visible placebo set is fixed as eight available, positively oriented performance counts that were not on the official list:

`effective_disposals`, `uncontested_possessions`, `inside_fifties`, `rebounds`, `one_percenters`, `pressure_acts`, `ground_ball_gets`, `metres_gained`.

These are negative/falsification comparators, not untreated variables: they are correlated with visible performance and could still influence umpires through live observation.

## Transformations and primary model

Every stat is converted to a within-match percentile rank and centered at 0.5. A visible-information index is the equal-weight mean of the 16 centered visible ranks; a placebo index is the equal-weight mean of the eight centered non-visible ranks. Both indices are standardized over the full 2022–2026 analysis panel for coefficient comparability. No weights are learned from Brownlow votes.

Primary stacked model:

`actual_votes ~ visible_index + placebo_index + post2026 + visible_index:post2026 + placebo_index:post2026 + team_won + C(position_group) + C(season)`

Because `post2026` is collinear with season indicators, it is absorbed by the season fixed effects. Match-clustered HC1 standard errors are used. Primary mechanism contrast:

`(visible_index × post2026) - (placebo_index × post2026)`.

The two interaction coefficients and their direct contrast are the predeclared primary family; Holm correction applies to the two interaction p-values. The contrast is reported with its direct clustered covariance-based confidence interval.

Secondary feature-level models use the identical transformation and controls but one prespecified feature at a time. They are descriptive mechanism decomposition; Holm correction is applied separately within the 16 visible and eight placebo interaction families.

## Coach–umpire alignment outcomes

Pre-reform reference is the equal-weight average of annual 2022–2025 statistics. Post is 2026. Match-cluster bootstrap uncertainty uses 2,000 seeded replicates.

1. Player-match Spearman between coaches votes and Brownlow votes.
2. Mean match-level overlap between the coach top three and Brownlow three vote recipients, divided by three.
3. Extreme-disagreement match rate: at least one `coaches_votes == 10 & actual_votes == 0` or `actual_votes == 3 & coaches_votes == 0` event.
4. Mean within-match Spearman rank agreement.

All definitions are locked before inspecting the inferred deltas. The AFL-reported 35/34/34/23 extreme-disagreement counts are an external descriptive benchmark; discrepancies caused by dataset definitions will be retained and investigated rather than forced.

## Predictability and falsification

The existing frozen/OOT RMSE series is reported as instrument validation only. Reform-specific interpretation is strengthened only if visible sensitivity increases more than placebo sensitivity and alignment changes in the predicted direction. Similar visible and placebo shifts weaken that interpretation. Null and contrary results will be retained.

## Outputs

Only four paper-facing artifacts are produced:

1. `table1_predictive_instrument.csv`
2. `figure1_recognition_before_after.png`
3. `figure2_visible_vs_placebo.png`
4. `table2_robustness_falsification.csv`

Machine-readable supporting estimates may be stored in one additional long-form results file. Existing forecasting assets, website files, and the earlier frozen residual analysis are read-only.
