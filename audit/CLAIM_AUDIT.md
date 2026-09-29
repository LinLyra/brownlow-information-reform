# Claim audit

## SUPPORTED

| Candidate statement | Status | Supporting result | Limitation | Recommended wording |
|---|---|---|---|---|
| 1. Brownlow voting became more sensitive to the statistics supplied to umpires in 2026. | SUPPORTED as a structural association | Visible-index change +0.0301, 95% CI [0.0122, 0.0480]; largest of the actual/pseudo-year estimates; all alternative composites and leave-one-stat-out estimates remain positive. | One post season; 16 of 17 official categories mapped; direct visible-minus-placebo contrast crosses zero. | “Brownlow voting became more sensitive in 2026 to the performance statistics supplied to umpires.” |
| 3. Coach and umpire recognition became more aligned in 2026. | SUPPORTED | Player-match Spearman increased from a 2022–25 mean of 0.552 to 0.617; change +0.066, bootstrap CI [0.043, 0.089]. Top-three overlap and winner agreement also increased. | Coaches are an independent assessment, not ground truth; season-level changes may have other causes. | “Umpire recognition became more aligned with independent coach assessment in 2026.” |
| 5. 2026 represents an unusual structural break relative to previous seasons. | SUPPORTED descriptively, not causally | 2026 visible change +0.0301 versus pseudo-2023 −0.0158, pseudo-2024 −0.0221, and pseudo-2025 −0.0034; 2026 ranks first and exceeds the largest pseudo estimate by 0.0335. | Only three pseudo years; not a formal randomization test. | “The 2026 change was unusually large relative to the three available pseudo-reform years.” |
| 7. Expected-recognition modelling materially outperforms simple baselines. | SUPPORTED | Overall RMSE 0.3465 versus ridge 0.3820, coach 0.3930, uniform 0.5360 and disposal 0.5550; frozen-2026 RMSE 0.3005. | This validates the measurement instrument; it is not the main scientific contribution. | “The frozen/OOT model materially outperformed four transparent benchmarks.” |

## SUGGESTIVE

| Candidate statement | Status | Supporting result | Limitation | Recommended wording |
|---|---|---|---|---|
| The 2026 change was concentrated specifically in the newly supplied information. | SUGGESTIVE | Visible change +0.0301; aggregate placebo +0.0045. | Direct contrast +0.0256 has CI [−0.0107, 0.0618]; several individual placebo variables also moved. Visible/placebo indices correlate 0.787. | “The pattern is consistent with, but does not isolate, a reform-specific information channel.” |

## NOT SUPPORTED

| Candidate statement | Status | Supporting result | Limitation | Recommended wording |
|---|---|---|---|---|
| 2. The information reform caused umpires to rely more on statistics. | NOT SUPPORTED | Timing and direction are consistent with the reform. | No untreated AFL control, one post season, contemporaneous season changes. | Do not use “caused”; use “changed following” or “was consistent with.” |
| 4. The reform eliminated umpire bias. | NOT SUPPORTED | Some alignment outcomes improved. | Residuals are model-relative deviations, not bias; disagreement remains; no causal design. | Do not use this claim. |
| 6. Reputation persistence exists in Brownlow voting. | NOT SUPPORTED as a replicated result | Development β=+0.00613, CI [+0.00158,+0.01069]. | Frozen 2026 β=−0.00180, CI [−0.00736,+0.00377]; alternative measures null; player-FE result reverses. | “A development-period association did not replicate in the frozen holdout.” |

