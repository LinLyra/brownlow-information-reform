# Final identification memo

## 1. Research question

Did Brownlow voting change unusually in 2026, and was that change concentrated in the statistical information newly supplied to field umpires?

## 2. Institutional setting

Beginning in 2026, AFL field umpires were supplied an approved Champion Data page after each match before finalising Brownlow votes. Sixteen of the 17 announced categories map reliably to the research dataset; kick-ins are unavailable. The league-wide timing creates a useful structural-change setting but not a controlled natural experiment.

## 3. Predictive measurement instrument

The full Brownlow Intelligence model uses rolling OOT predictions in 2022–2025 and a prospectively frozen 2026 forecast. Overall player-match RMSE is 0.3465, versus 0.3820 for ridge and 0.3930 for the coach-vote benchmark. Frozen-2026 RMSE is 0.3005, with 69.6% three-vote winner accuracy and 75.7% top-three recall. These results validate expected recognition as a measurement instrument; forecasting is not the paper's endpoint.

## 4. Primary structural-change evidence

Voting sensitivity to the equal-weight, within-match ranked visible-stat index increased by 0.0301 in 2026 (95% CI [0.0122, 0.0480]). The aggregate non-visible placebo change was 0.0045 (95% CI [−0.0148, 0.0238]). The direct difference was +0.0256 but imprecise (95% CI [−0.0107, 0.0618]).

## 5. Pseudo-reform results

Applying the identical expanding-window specification produced visible changes of −0.0158 for pseudo-2023, −0.0221 for pseudo-2024, −0.0034 for pseudo-2025, and +0.0301 for actual 2026. Actual 2026 ranks first and exceeds the largest pseudo estimate by 0.0335. With only three placebo dates, this is descriptive evidence of unusualness, not a randomization-inference p-value.

## 6. Visible versus placebo identification

The two composite indices correlate 0.787 (VIF 2.62), leaving 37.5% of visible-index variance after residualising against the placebo and temporal controls. More importantly, the clustered visible- and placebo-interaction estimates have correlation −0.894. Their negative covariance increases the variance of the difference: the contrast SE is 0.0185, larger than either component SE. The weak direct contrast is therefore explained mainly by limited independent separation of highly related performance information and the covariance of the two estimated interactions, compounded by having only 207 post-reform match clusters. It is not primarily index instability: all three alternative composites and every leave-one-visible-stat-out estimate remain positive. Nor do the aggregate indices show genuinely similar changes, although individual placebo statistics sometimes move.

## 7. Coach–umpire convergence

Player-match Spearman increased from a 2022–2025 annual mean of 0.552 to 0.617 (change +0.066, bootstrap CI [0.043, 0.089]). Within-match rank correlation increased by +0.066 [0.043, 0.089], top-three overlap by +0.054 [0.019, 0.087], and winner agreement by +0.083 [0.005, 0.153]. Extreme-disagreement match rate fell by 0.033, but its interval [−0.076, 0.015] includes zero. Coach voting is an independent assessment, not objective truth.

## 8. Robustness

Alternative 2026 visible changes are +0.0316 for within-match z-score equal weighting, +0.0404 for pre-2026 PCA1, and +0.0327 for pre-2026 predictive ridge weights. Their direct visible-minus-placebo contrasts remain positive but all intervals cross zero. Leave-one-visible-stat-out estimates range from +0.0219 to +0.0343, median +0.0289, with no sign reversal. Excluding 2022 strengthens rather than removes the visible pattern (+0.0354) but still leaves the direct contrast marginally imprecise.

## 9. Falsification

The aggregate placebo did not change clearly, but several individual non-visible measures did. Accordingly, the mechanism evidence is mixed: the official information index changes unusually and robustly, while the direct separation from correlated non-visible performance remains unresolved.

## 10. Negative findings

Development-period recognition persistence (+0.00613, CI [+0.00158,+0.01069]) did not replicate in frozen 2026 (−0.00180, CI [−0.00736,+0.00377]). Alternative reputation measures are null, and player fixed effects reverse the development association. This is a documented non-replication, not evidence that reputation effects were eliminated by the reform.

## 11. Limitations

There is one post-reform season, no untreated AFL control, correlated information sets, one unavailable official category, possible season-specific rule/style/composition changes, and unresolved data redistribution rights. The local extreme-disagreement reconstruction contains 24 events versus 23 in the AFL article; the extra record is Caleb Daniel's recorded 10 coach votes against Carlton in round four.

## 12. What can be claimed

Brownlow voting changed unusually in 2026: it became more sensitive to the supplied-statistics index and more aligned with independent coach assessment. The frozen expected-recognition instrument outperforms transparent baselines.

## 13. What cannot be claimed

The reform cannot be said to have caused the change, eliminated bias, or uniquely shifted voting toward only the supplied information. Residuals are not bias, coach votes are not ground truth, and the intervention is not human–AI collaboration.

## 14. Recommended abstract framing

Lead with an information-assisted human-evaluation question, introduce the prospective frozen measurement instrument briefly, report the unusual 2026 visible-stat break and multi-metric coach convergence, then disclose the imprecise visible-minus-placebo contrast. The strongest defensible framing is **“When Judges Get the Data: Recognition Around the 2026 Brownlow Medal Information Reform.”**
