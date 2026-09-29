# SSAC27 final identification protocol

Frozen on 2026-09-29 after the reproduction gate passed and before inspecting any pseudo-reform, alternative-composite, leave-one-out, or event-study result.

## Primary estimand

The main estimand remains the 2026 change in the coefficient on the pre-specified AFL-visible composite, with the contemporaneous change in a non-visible placebo composite included in the same regression. The direct mechanism contrast is `visible_change - placebo_change`. These are structural-change associations, not causal effects.

## Information sets

Visible statistics are the 16 reliably mapped categories from the AFL's 17-category announcement: `kicks`, `handballs`, `disposals`, `marks`, `contested_marks`, `tackles`, `goals`, `behinds`, `goal_assists`, `score_involvements`, `clearances`, `contested_possessions`, `hitouts`, `intercept_marks`, `intercepts` (mapping intercept possessions), and `spoils`. `kick-ins` is missing and will not be proxied.

The non-visible placebo set remains: `effective_disposals`, `uncontested_possessions`, `inside_fifties`, `rebounds`, `one_percenters`, `pressure_acts`, `ground_ball_gets`, and `metres_gained`. These are correlated falsification comparators, not untreated controls.

## Primary construction and estimator

Each feature is converted to a within-match percentile rank centered at 0.5. Each information-set index is the equal-weight mean of its constituent ranks and is standardized once over the full 2022–2026 panel. The model is:

`votes ~ visible + placebo + visible×target_year + placebo×target_year + team_won + position FE + season FE`

The actual intervention target is 2026. Standard errors are HC1 finite-sample corrected and clustered by match. All player-match rows in the frozen/OOT evaluation universe are included. Required current statistics have no missing values; no observation or feature may be silently dropped.

## Pseudo-reform procedure

For target years 2023, 2024, 2025, and 2026, use exactly the same construction, controls, estimator, and clustering. To avoid using a real future reform to evaluate an earlier fake reform, each target-year analysis uses the expanding sample from 2022 through the target year; `target_year` equals one only in the final year. Thus every estimate compares the nominated year with all available earlier seasons, with season fixed effects. The four estimates are descriptive; three pseudo years are too few for a formal randomization-inference p-value. Actual-2026 rank and distance from the pseudo distribution will be reported.

Coach-alignment pseudo changes use each target year's metric minus the equal-weight mean of available prior annual metrics.

## Identification diagnostics

Report full-panel and within-season composite correlation/covariance, two-predictor VIF, residualized visible-index variance share after placebo and controls, interaction-estimate variance/covariance, and the variance of their difference. A feature-level visible-by-placebo correlation matrix is descriptive. Diagnostics—not the desired narrative—determine whether imprecision is primarily covariance, index instability, limited independent variation/power, or similar changes.

## Predefined robustness

- R1: equal-weight mean of within-match z-scored raw features, then standardized globally.
- R2: first principal component of within-match z-scored features. Loadings are fitted on 2022–2025 only, applied unchanged to 2026, and oriented so the component correlates positively with its equal-weight index.
- R3: pre-2026 predictive weights from a fixed Ridge(alpha=10) model fitted separately to each information set using 2022–2025 outcomes only. Inputs are within-match z-scores; weights and scaling are applied unchanged to 2026.

The primary percentile-rank index remains unchanged. Robustness is judged on sign, magnitude, interval overlap, and the direct contrast—not on significance.

Leave-one-visible-stat-out removes each mapped visible feature once, rebuilds the primary equal-weight rank index, and reruns the 2026 specification. No preferred subset will be selected.

## Event study

For each official mapped feature, use its globally standardized within-match percentile rank and estimate the same controlled model separately by season. The publication figure is restricted a priori to: possession (`disposals`, `kicks`, `handballs`), contest (`contested_possessions`, `clearances`), and scoring (`goals`, `score_involvements`). These families are based on the AFL's official stat labels, not observed significance.

## Alignment robustness

Annual metrics are player-match Spearman, mean within-match rank correlation, top-three overlap, three-vote winner agreement, and extreme-disagreement match rate. Coaches are an independent assessment, not ground truth. Match-cluster bootstrap uses 2,000 seeded replicates. Report annual levels and 2026 changes against both the equal-weight 2022–2025 annual reference and the pooled pre-2026 reference where defined.

## Other locked analyses

- Instrument validation reports 2022–2025 row-weighted pooled, 2026 frozen, and 2022–2026 row-weighted overall results for five fixed benchmarks; no retraining.
- Reputation non-replication is copied from the already frozen development, confirmation, and robustness outputs.
- Pre-period sensitivity compares 2022–2025 with 2023–2025 for major descriptive changes.
- Official disagreement events are reconstructed mechanically. No event is deleted to match the AFL article.
- Salience/cognitive availability remains exploratory because no external classification has been established. Only a descriptive table using the protocol's conceptual families may be produced.

## Multiple testing and decision rules

Primary interaction p-values use Holm correction as already specified; the direct contrast uses its joint covariance. Feature event-study/feature-level families use Holm within their declared families where p-values are shown. Pseudo-year, leave-one-out, salience, and composite-robustness analyses are specification diagnostics, not new confirmatory tests.

Evidence that 2026 is unusual requires its visible-change estimate to exceed all three pseudo-year point estimates and not merely attain a smaller p-value. Reform-specific evidence is called strong only if the direct visible-minus-placebo contrast excludes zero and is directionally stable across R1–R3 and leave-one-out. Otherwise it is labelled suggestive or mixed. No causal wording is permitted because there is one post year and no untreated league control.
