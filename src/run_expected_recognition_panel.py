#!/usr/bin/env python3
"""Additive SSAC27 research analysis. Never retrains or overwrites the frozen model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/derived"
FIG = ROOT / "paper/figures/supporting"
RAW = ROOT / "data/private/brownlow_datathon_dataset.csv"
OOF = ROOT / "data/private/v2_1_coach_oof_predictions.csv"
FROZEN = ROOT / "data/private/2026_player_match_predictions.csv"
LABELS = ROOT / "data/private/brownlow_2026_actual_votes.csv"
SEED = 42
POSITION_LEVELS = ["Back", "Forward", "Interchange", "Midfielder", "Ruck"]
RIDGE_FEATURES = [
    "coaches_votes", "disposals", "goals", "clearances", "contested_possessions",
    "hitouts", "contested_marks", "tackles", "marks", "inside_fifties",
    "score_involvements", "metres_gained", "intercepts", "spoils",
    "supercoach_score", "afl_fantasy_score", "rating_points", "team_won", "abs_margin",
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def allocate(v: np.ndarray, total: float = 6.0, cap: float = 3.0) -> np.ndarray:
    v = np.asarray(v, float)
    lo, hi = float(v.min() - cap), float(v.max())
    for _ in range(200):
        tau = (lo + hi) / 2
        if np.clip(v - tau, 0, cap).sum() > total:
            lo = tau
        else:
            hi = tau
    return np.clip(v - (lo + hi) / 2, 0, cap)


def add_context(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    x["player_name"] = x["player_first_name"].fillna("") + " " + x["player_last_name"].fillna("")
    x["player_name"] = x["player_name"].str.strip()
    x["team_won"] = (x["player_team"] == x["match_winner"]).astype(int)
    x["abs_margin"] = x["match_margin"].abs().astype(float)
    x["close_game"] = (x["abs_margin"] <= 12).astype(int)
    pos = np.select(
        [
            x["player_position"].isin(["RK", "R", "RR"]),
            x["player_position"].isin(["SUB", "INT", "EMERG"]),
            x["is_midfielder"].eq(1), x["is_forward"].eq(1), x["is_back"].eq(1),
        ],
        ["Ruck", "Interchange", "Midfielder", "Forward", "Back"], default="Other",
    )
    x["position_group"] = pd.Categorical(pos, categories=POSITION_LEVELS)
    return x


def build_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = add_context(pd.read_csv(RAW))
    oof = pd.read_csv(OOF)
    hist_pred = oof[["season", "match_id", "player_id", "actual_brownlow_votes",
                     "coach_components_prediction_model", "coach_components_prediction_model_raw"]].rename(
        columns={"actual_brownlow_votes": "actual_votes",
                 "coach_components_prediction_model": "allocated_ev",
                 "coach_components_prediction_model_raw": "raw_ev"})
    hist_raw = raw[raw.season.between(2022, 2025)].drop(columns=["brownlow_votes"])
    hist = hist_pred.merge(hist_raw, on=["season", "match_id", "player_id"], how="left", validate="one_to_one")
    assert hist["player_team"].notna().all()

    frozen = pd.read_csv(FROZEN)
    labels = pd.read_csv(LABELS)[["season", "match_id", "player_id", "actual_brownlow_votes"]]
    fut = frozen.merge(labels, on=["season", "match_id", "player_id"], how="left", validate="one_to_one")
    fut["actual_votes"] = fut.pop("actual_brownlow_votes").fillna(0.0)
    fut = fut.rename(columns={"allocated_expected_votes": "allocated_ev", "raw_brownlow_prediction": "raw_ev"})
    raw26_cols = [c for c in raw.columns if c not in fut.columns or c in ["season", "match_id", "player_id"]]
    fut = fut.merge(raw[raw.season.eq(2026)][raw26_cols], on=["season", "match_id", "player_id"], how="left", validate="one_to_one")
    # Prefer frozen naming where duplicated semantics exist, then reconstruct context consistently.
    for c in ["player_first_name", "player_last_name", "player_team", "player_position"]:
        if c + "_x" in fut:
            fut[c] = fut[c + "_x"]
    fut = add_context(fut)
    assert np.allclose(fut.groupby("match_id").actual_votes.sum(), 6.0)

    keep_common = sorted(set(hist.columns) & set(fut.columns))
    panel = pd.concat([hist[keep_common], fut[keep_common]], ignore_index=True)
    panel["residual"] = panel["actual_votes"] - panel["allocated_ev"]
    panel["position_group"] = pd.Categorical(panel["position_group"], categories=POSITION_LEVELS)

    # Strict prior actual career measures from all raw history.
    actual = raw[raw.season.le(2025)].copy()
    actual["actual_votes_hist"] = actual["brownlow_votes"].fillna(0.0)
    ps = actual.groupby(["player_id", "season"], as_index=False).agg(
        season_actual_votes=("actual_votes_hist", "sum"), season_games=("match_id", "nunique"))
    grid = []
    for season in range(2022, 2027):
        prior = ps[ps.season < season]
        prior_no20 = ps[(ps.season < season) & (ps.season != 2020)]
        career = prior.groupby("player_id", as_index=False).agg(
            career_prior_votes=("season_actual_votes", "sum"), career_prior_games=("season_games", "sum"))
        career20 = prior_no20.groupby("player_id", as_index=False).agg(
            career_prior_votes_excl2020=("season_actual_votes", "sum"), career_prior_games_excl2020=("season_games", "sum"))
        prev = ps[ps.season.eq(season - 1)][["player_id", "season_actual_votes"]].rename(columns={"season_actual_votes": "prior_season_actual_votes"})
        z = pd.DataFrame({"player_id": panel.loc[panel.season.eq(season), "player_id"].unique()})
        z = z.merge(career, on="player_id", how="left").merge(career20, on="player_id", how="left").merge(prev, on="player_id", how="left")
        z["season"] = season
        grid.append(z)
    lag = pd.concat(grid, ignore_index=True)
    panel = panel.merge(lag, on=["season", "player_id"], how="left", validate="many_to_one")
    for c in ["career_prior_votes", "career_prior_games", "career_prior_votes_excl2020", "career_prior_games_excl2020", "prior_season_actual_votes"]:
        panel[c] = panel[c].fillna(0.0)
    panel["career_votes_per_game"] = panel["career_prior_votes"] / panel["career_prior_games"].replace(0, np.nan)
    panel["career_votes_per_game"] = panel["career_votes_per_game"].fillna(0.0)
    panel["career_votes_per_game_excl2020"] = panel["career_prior_votes_excl2020"] / panel["career_prior_games_excl2020"].replace(0, np.nan)
    panel["career_votes_per_game_excl2020"] = panel["career_votes_per_game_excl2020"].fillna(0.0)
    panel["rookie_or_no_history"] = panel["career_prior_games"].eq(0).astype(int)

    seas = panel.groupby(["player_id", "season"], as_index=False).agg(
        this_expected=("allocated_ev", "sum"), this_actual=("actual_votes", "sum"))
    seas["this_residual"] = seas["this_actual"] - seas["this_expected"]
    seas["season"] += 1
    seas = seas.rename(columns={"this_expected": "prior_season_expected_votes", "this_residual": "prior_season_residual"})
    panel = panel.merge(seas[["player_id", "season", "prior_season_expected_votes", "prior_season_residual"]], on=["player_id", "season"], how="left", validate="many_to_one")
    panel["prior_season_lag_missing"] = panel["prior_season_residual"].isna().astype(int)
    panel[["prior_season_expected_votes", "prior_season_residual"]] = panel[["prior_season_expected_votes", "prior_season_residual"]].fillna(0.0)
    return panel.sort_values(["season", "match_id", "player_id"]).reset_index(drop=True), raw


def provenance(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, g in panel.groupby("season"):
        rows.append({"season": season,
                     "prediction_source": str(OOF.relative_to(ROOT) if season < 2026 else FROZEN.relative_to(ROOT)),
                     "prediction_type": "ROLLING OOT" if season < 2026 else "FROZEN FUTURE HOLDOUT",
                     "rmse": np.sqrt(np.mean(g.residual ** 2)), "n_matches": g.match_id.nunique(),
                     "n_player_match_rows": len(g)})
    return pd.DataFrame(rows)


def design(df: pd.DataFrame, spec: int = 4, extra: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    cols = ["prior_season_residual", "prior_season_expected_votes", "rookie_or_no_history"]
    if spec >= 2:
        cols += [f"position_{p}" for p in POSITION_LEVELS if p != "Back"]
    if spec >= 3:
        cols += ["team_won", "abs_margin", "close_game"]
    if spec >= 4:
        cols += ["coaches_votes"]
    cols += extra or []
    z = df.copy()
    for p in POSITION_LEVELS:
        z[f"position_{p}"] = (z.position_group.astype(str) == p).astype(float)
    return np.column_stack([np.ones(len(z))] + [z[c].astype(float).to_numpy() for c in cols]), ["Intercept"] + cols


def clustered_ols(df: pd.DataFrame, spec: int = 4, extra: list[str] | None = None,
                  outcome: str = "residual") -> pd.DataFrame:
    X, names = design(df, spec, extra)
    y = df[outcome].astype(float).to_numpy()
    good = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y, clusters = X[good], y[good], df.loc[good, "match_id"].to_numpy()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ beta
    bread = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    unique = np.unique(clusters)
    for cl in unique:
        xu = X[clusters == cl].T @ u[clusters == cl]
        meat += np.outer(xu, xu)
    n, k, G = len(y), X.shape[1], len(unique)
    correction = (G / (G - 1)) * ((n - 1) / max(n - k, 1)) if G > 1 else 1.0
    cov = correction * bread @ meat @ bread
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    dof = max(G - 1, 1)
    p = 2 * stats.t.sf(np.abs(beta / np.where(se > 0, se, np.nan)), dof)
    crit = stats.t.ppf(.975, dof)
    r2 = 1 - np.sum(u ** 2) / np.sum((y - y.mean()) ** 2)
    return pd.DataFrame({"term": names, "coefficient": beta, "std_error": se,
                         "ci_low": beta - crit * se, "ci_high": beta + crit * se,
                         "p_raw": p, "n": n, "n_clusters": G, "r_squared": r2})


def holm(p: pd.Series) -> pd.Series:
    vals = p.to_numpy(float); order = np.argsort(vals); m = len(vals); out = np.empty(m); running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * vals[idx]); out[idx] = min(running, 1.0)
    return pd.Series(out, index=p.index)


def add_families(tbl: pd.DataFrame, strata: list[str] | None = None) -> pd.DataFrame:
    def fam(term: str) -> str:
        if term == "prior_season_residual": return "recognition_persistence"
        if term.startswith("position_"): return "position"
        if term in ["team_won", "abs_margin", "close_game"]: return "match_context"
        if term in ["prior_season_actual_votes", "career_votes_per_game", "career_votes_per_game_excl2020"]: return "alternative_reputation"
        return "covariate"
    tbl = tbl.copy(); tbl["family"] = tbl.term.map(fam); tbl["p_holm"] = tbl.p_raw
    group_cols = (strata or []) + ["family"]
    for key, idx in tbl.groupby(group_cols).groups.items():
        family = key[-1] if isinstance(key, tuple) else key
        if family != "covariate": tbl.loc[idx, "p_holm"] = holm(tbl.loc[idx, "p_raw"])
    return tbl


def prediction_metrics(df: pd.DataFrame, pred: str) -> dict:
    err = df[pred] - df.actual_votes
    hits, recalls = [], []
    for _, g in df.groupby("match_id"):
        top = set(g.nlargest(3, pred).player_id)
        top1 = set(g.nlargest(1, pred).player_id)
        winners = set(g.loc[g.actual_votes.eq(3), "player_id"])
        actual3 = set(g.loc[g.actual_votes.gt(0), "player_id"])
        hits.append(float(bool(top1 & winners)))
        recalls.append(len(top & actual3) / max(len(actual3), 1))
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "three_vote_winner_hit_rate": float(np.mean(hits)), "top3_recall": float(np.mean(recalls))}


def bootstrap_metrics(df: pd.DataFrame, pred: str, reps: int = 400) -> dict:
    rng = np.random.default_rng(SEED)
    rows = []
    for _, g in df.groupby("match_id"):
        e = g[pred].to_numpy() - g.actual_votes.to_numpy()
        top = set(g.nlargest(3, pred).player_id); top1 = set(g.nlargest(1, pred).player_id); winners = set(g.loc[g.actual_votes.eq(3), "player_id"]); actual3 = set(g.loc[g.actual_votes.gt(0), "player_id"])
        rows.append((len(g), np.sum(e*e), np.sum(np.abs(e)), float(bool(top1 & winners)), len(top & actual3)/max(len(actual3),1)))
    a = np.asarray(rows, float); vals = {k: [] for k in ["rmse","mae","three_vote_winner_hit_rate","top3_recall"]}
    for _ in range(reps):
        s = a[rng.integers(0, len(a), len(a))]
        vals["rmse"].append(np.sqrt(s[:,1].sum()/s[:,0].sum())); vals["mae"].append(s[:,2].sum()/s[:,0].sum())
        vals["three_vote_winner_hit_rate"].append(s[:,3].mean()); vals["top3_recall"].append(s[:,4].mean())
    return {k: (np.quantile(v, .025), np.quantile(v, .975)) for k, v in vals.items()}


def run_baselines(panel: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for season in range(2022, 2027):
        g = panel[panel.season.eq(season)].copy()
        g["uniform"] = g.groupby("match_id").player_id.transform(lambda s: 6 / len(s))
        g["disposal"] = 0.0
        for _, idx in g.groupby("match_id").groups.items():
            order = g.loc[idx].sort_values(["disposals", "player_id"], ascending=[False, True]).index
            g.loc[order[:3], "disposal"] = [3., 2., 1.]
        g["coach"] = g.groupby("match_id")["coaches_votes"].transform(
            lambda s: 6 * s / s.sum() if s.sum() > 0 else np.full(len(s), 6 / len(s)))
        train = add_context(raw[raw.season.lt(season) & raw.brownlow_votes.notna()].copy())
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
        model.fit(train[RIDGE_FEATURES], train.brownlow_votes)
        g["ridge_raw"] = model.predict(g[RIDGE_FEATURES])
        g["ridge"] = g.groupby("match_id", group_keys=False)["ridge_raw"].transform(lambda s: allocate(s.to_numpy()))
        g["full"] = g["allocated_ev"]
        for name in ["uniform", "disposal", "coach", "ridge", "full"]:
            met = prediction_metrics(g, name); ci = bootstrap_metrics(g, name)
            row = {"season": season, "model": name, **met}
            for k, (lo, hi) in ci.items(): row[k + "_ci_low"], row[k + "_ci_high"] = lo, hi
            pieces.append(row)
    return pd.DataFrame(pieces)


def allocation_ablation(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, g in panel.groupby("season"):
        for name, col in [("raw_model_output", "raw_ev"), ("six_vote_constrained", "allocated_ev")]:
            rows.append({"season": season, "variant": name, **prediction_metrics(g, col)})
    return pd.DataFrame(rows)


def plots(dev: pd.DataFrame, con: pd.DataFrame, prov: pd.DataFrame, bases: pd.DataFrame, panel: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    terms = ["prior_season_residual", "position_Forward", "position_Interchange", "position_Midfielder", "position_Ruck", "team_won", "abs_margin", "close_game"]
    d = dev[(dev.specification.eq("D4_full")) & dev.term.isin(terms)].copy(); d["period"] = "2023–25 development"
    c = con[con.term.isin(terms)].copy(); c["period"] = "2026 frozen holdout"
    z = pd.concat([d, c]); ymap = {t: i for i, t in enumerate(terms[::-1])}
    fig, ax = plt.subplots(figsize=(8, 5.8))
    for j, (period, gg) in enumerate(z.groupby("period")):
        yy = np.array([ymap[t] for t in gg.term]) + (-.12 if j == 0 else .12)
        ax.errorbar(gg.coefficient, yy, xerr=[gg.coefficient-gg.ci_low, gg.ci_high-gg.coefficient], fmt="o", capsize=3, label=period)
    ax.axvline(0, color="black", lw=.8); ax.set_yticks(range(len(terms)), terms[::-1]); ax.set_xlabel("Residual-vote coefficient (95% CI)")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG / "coefficient_development_vs_2026.png", dpi=220); plt.close(fig)

    bm = bases.groupby("model", as_index=False).rmse.mean().sort_values("rmse")
    fig, ax = plt.subplots(figsize=(7.5, 4.5)); ax.barh(bm.model, bm.rmse, color="#466b8a"); ax.invert_yaxis(); ax.set_xlabel("Mean player-match RMSE, 2022–2026")
    for i, v in enumerate(bm.rmse): ax.text(v + .002, i, f"{v:.3f}", va="center", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "baseline_validation.png", dpi=220); plt.close(fig)

    bins = pd.qcut(panel.allocated_ev, 10, duplicates="drop")
    cal = panel.assign(bin=bins).groupby("bin", observed=True).agg(expected=("allocated_ev", "mean"), realised=("actual_votes", "mean"), n=("actual_votes", "size")).reset_index()
    fig, ax = plt.subplots(figsize=(5.3, 5)); ax.plot(cal.expected, cal.realised, "o-", color="#9b3a32"); lim=max(cal.expected.max(),cal.realised.max()); ax.plot([0,lim],[0,lim],"--",color="grey"); ax.set(xlabel="Mean expected votes", ylabel="Mean realised votes")
    fig.tight_layout(); fig.savefig(FIG / "expected_vs_realised_calibration.png", dpi=220); plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    assert sha(FROZEN) == "5be4cc3381529c4bb88ca5c5197f5c5481b7e114f62a5c697ad61e71a1bfb5ad"
    panel, raw = build_panel()
    export_cols = ["season", "match_id", "player_id", "player_name", "player_team", "position_group", "allocated_ev", "raw_ev", "actual_votes", "residual", "team_won", "match_margin", "abs_margin", "close_game", "coaches_votes", "disposals", "goals", "clearances", "contested_possessions", "hitouts", "contested_marks", "goal_assists", "score_involvements", "intercepts", "spoils", "ruck_contests", "hitouts_to_advantage", "prior_season_actual_votes", "prior_season_expected_votes", "prior_season_residual", "prior_season_lag_missing", "career_prior_votes", "career_prior_games", "career_votes_per_game", "career_votes_per_game_excl2020", "rookie_or_no_history"]
    panel[export_cols].to_csv(OUT / "residual_panel.csv", index=False)
    prov = provenance(panel); prov.to_csv(OUT / "oot_provenance.csv", index=False)
    expected = {2022:.3153137594342417, 2023:.3809057262528894, 2024:.37062141099178547, 2025:.3566204989910542, 2026:.300540}
    for _, r in prov.iterrows(): assert abs(r.rmse - expected[int(r.season)]) < (2e-5 if r.season == 2026 else 1e-12)

    dev0 = panel[panel.season.between(2023, 2025)].copy()
    dev_parts = []
    for spec, label in [(1,"D1_persistence"),(2,"D2_plus_position"),(3,"D3_plus_context"),(4,"D4_full")]:
        r = clustered_ols(dev0, spec); r["specification"] = label; r["period"] = "2023-2025 development"; dev_parts.append(r)
    dev = add_families(pd.concat(dev_parts, ignore_index=True), strata=["specification"]); dev.to_csv(OUT / "development_regressions.csv", index=False)

    conf_data = panel[panel.season.eq(2026)].copy()
    conf = add_families(clustered_ols(conf_data, 4)); conf["specification"] = "LOCKED_CONFIRMATORY"; conf["period"] = "2026 frozen holdout"; conf.to_csv(OUT / "confirmatory_2026.csv", index=False)

    robust = []
    # Alternative strictly lagged recognition measures replace the primary residual term while retaining locked controls.
    for term, label in [("prior_season_actual_votes","prior_actual"),("career_votes_per_game","career_vpg"),("career_votes_per_game_excl2020","career_vpg_excl2020")]:
        q = conf_data.copy(); q["prior_season_residual"] = q[term]
        rr = clustered_ols(q, 4); rr = rr[rr.term.eq("prior_season_residual")].copy(); rr["term"] = term; rr["analysis"] = label; robust.append(rr)
    # Rich current position-specific controls test whether broad-position residual patterns attenuate.
    pos_stats = ["hitouts", "contested_marks", "goals", "score_involvements", "intercepts", "spoils", "ruck_contests", "hitouts_to_advantage"]
    rr = clustered_ols(conf_data, 4, extra=pos_stats); rr["analysis"] = "position_feature_sensitivity_2026"; robust.append(rr)
    # Development player fixed-effects via within-player demeaning for the continuous primary relationship.
    fe = dev0[["player_id", "match_id", "residual", "prior_season_residual", "prior_season_expected_votes", "team_won", "abs_margin", "close_game", "coaches_votes"]].copy()
    fe_cols = ["residual", "prior_season_residual", "prior_season_expected_votes", "team_won", "abs_margin", "close_game", "coaches_votes"]
    for c in fe_cols: fe[c] = fe[c] - fe.groupby("player_id")[c].transform("mean")
    fr = clustered_ols(fe.assign(rookie_or_no_history=0, position_group=pd.Categorical(["Back"]*len(fe), categories=POSITION_LEVELS)), 1, extra=["team_won","abs_margin","close_game","coaches_votes"])
    fr["analysis"] = "player_FE_within_development"; robust.append(fr)
    robust_df = add_families(pd.concat(robust, ignore_index=True)); robust_df.to_csv(OUT / "robustness.csv", index=False)

    stability = []
    for season in range(2023, 2027):
        q = panel[panel.season.eq(season)]
        rr = clustered_ols(q, 4); rr["season"] = season; stability.append(rr)
    stability = add_families(pd.concat(stability, ignore_index=True), strata=["season"]); stability.to_csv(OUT / "season_stability.csv", index=False)

    bases = run_baselines(panel, raw); bases.to_csv(OUT / "baselines.csv", index=False)
    abl = allocation_ablation(panel); abl.to_csv(OUT / "allocation_ablation.csv", index=False)

    # Explicit exploratory outputs, kept separate from confirmatory tables.
    explor = []
    for season, g in panel.groupby("season"):
        explor.append({"season": season, "analysis": "coach_umpire_player_match_spearman", "estimate": g[["coaches_votes","actual_votes"]].corr(method="spearman").iloc[0,1], "n": len(g)})
        explor.append({"season": season, "analysis": "prediction_sd", "estimate": g.allocated_ev.std(), "n": len(g)})
        explor.append({"season": season, "analysis": "actual_vote_sd", "estimate": g.actual_votes.std(), "n": len(g)})
        explor.append({"season": season, "analysis": "mean_ev_given_actual_3", "estimate": g.loc[g.actual_votes.eq(3),"allocated_ev"].mean(), "n": int(g.actual_votes.eq(3).sum())})
    pd.DataFrame(explor).to_csv(OUT / "exploratory.csv", index=False)
    plots(dev, conf, prov, bases, panel)

    notes = f"""# SSAC27 audit trail\n\nGenerated by `src/run_expected_recognition_panel.py` in this reproducibility repository.\n\n- Sources: `{RAW.relative_to(ROOT)}`, `{OOF.relative_to(ROOT)}`, `{FROZEN.relative_to(ROOT)}`, `{LABELS.relative_to(ROOT)}`.\n- Frozen forecast hash was verified before analysis. Existing forecasting files were read only.\n- Historical residuals use `coach_components_prediction_model` (the V2.1 `base_plus_components` champion); 2026 uses `allocated_expected_votes`.\n- 2026 absent sparse labels were zero-filled only after a one-to-one forecast-universe join and six-vote-per-match validation.\n- Primary lag limitation: 2022 has no valid prior-season residual because 2021 OOT predictions are absent; no fitted substitute was made.\n- Broad position mapping and 12-point close-game threshold were frozen in the protocol.\n- Regression inference uses match-clustered HC1 finite-sample covariance.\n- Ridge is a research benchmark trained separately for each target season; it does not alter the production model.\n- Player fixed effects are a within-player development-period sensitivity and absorb persistent player heterogeneity; the reputation coefficient is therefore weakly identified by within-player changes.\n- Publication blocker: raw datathon and AFL Tables redistribution permissions remain unverified.\n- Failed/null/ambiguous results remain in the exported tables; no coefficient-based specification search was performed.\n"""
    (OUT / "REPRODUCTION_NOTES.md").write_text(notes)
    print(prov.to_string(index=False))
    print("\nDevelopment primary:\n", dev[(dev.specification.eq("D4_full")) & dev.term.eq("prior_season_residual")].to_string(index=False))
    print("\nConfirmatory primary:\n", conf[conf.term.eq("prior_season_residual")].to_string(index=False))
    print("\nWrote", OUT)


if __name__ == "__main__":
    main()
