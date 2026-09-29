# Data dictionary

## Identifiers and outcomes

| Variable | Definition |
|---|---|
| `season` | AFL season year |
| `match_id` | Match identifier |
| `player_id` | Player identifier |
| `actual_votes` | Realised player-match Brownlow votes, 0–3 |
| `allocated_ev` | OOT/frozen model expected votes after six-vote match allocation |
| `coaches_votes` | AFLCA player-match coach votes, 0–10 |

## Visible-stat composite

`kicks`, `handballs`, `disposals`, `marks`, `contested_marks`, `tackles`, `goals`, `behinds`, `goal_assists`, `score_involvements`, `clearances`, `contested_possessions`, `hitouts`, `intercept_marks`, `intercepts`, `spoils`.

These map 16 of the 17 categories in the AFL announcement. `kick-ins` is unavailable and is not proxied. Repository `intercepts` maps to the announcement's intercept possessions.

## Non-visible placebo composite

`effective_disposals`, `uncontested_possessions`, `inside_fifties`, `rebounds`, `one_percenters`, `pressure_acts`, `ground_ball_gets`, `metres_gained`.

These are correlated performance measures, not an untreated control group.

## Context

| Variable | Definition |
|---|---|
| `team_won` | Player's team won the match |
| `position_group` | Harmonised Back, Forward, Interchange, Midfielder or Ruck category |
| `post2026` | Indicator for the actual reform season |

