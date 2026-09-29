#!/usr/bin/env python3
"""Protocol-frozen information-reform analysis; reads but never changes forecasts."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_expected_recognition_panel import POSITION_LEVELS, add_context, holm  # noqa: E402

OUT = ROOT / "data/derived/information_reform"
FIG = ROOT / "paper/figures/supporting"
VISIBLE = ["kicks", "handballs", "disposals", "marks", "contested_marks", "tackles", "goals", "behinds",
           "goal_assists", "score_involvements", "clearances", "contested_possessions", "hitouts",
           "intercept_marks", "intercepts", "spoils"]
PLACEBO = ["effective_disposals", "uncontested_possessions", "inside_fifties", "rebounds", "one_percenters",
           "pressure_acts", "ground_ball_gets", "metres_gained"]
SEED, REPS = 42, 2000


def load_panel() -> pd.DataFrame:
    p = pd.read_csv(ROOT / "data/derived/residual_panel.csv")
    raw = add_context(pd.read_csv(ROOT / "data/private/brownlow_datathon_dataset.csv"))
    cols = ["season", "match_id", "player_id"] + VISIBLE + PLACEBO
    p = p.drop(columns=[c for c in VISIBLE + PLACEBO if c in p.columns])
    x = p.merge(raw[raw.season.between(2022, 2026)][cols], on=["season", "match_id", "player_id"], validate="one_to_one")
    assert x[VISIBLE + PLACEBO].notna().all().all()
    x["post2026"] = x.season.eq(2026).astype(float)
    for c in VISIBLE + PLACEBO:
        x[c + "_wmrank"] = x.groupby(["season", "match_id"])[c].rank(method="average", pct=True) - .5
    x["visible_index_raw"] = x[[c + "_wmrank" for c in VISIBLE]].mean(axis=1)
    x["placebo_index_raw"] = x[[c + "_wmrank" for c in PLACEBO]].mean(axis=1)
    for src, dst in [("visible_index_raw", "visible_index"), ("placebo_index_raw", "placebo_index")]:
        x[dst] = (x[src] - x[src].mean()) / x[src].std()
    return x


def controls(x: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    arrays, names = [np.ones(len(x))], ["Intercept"]
    for c in ["team_won"]:
        arrays.append(x[c].to_numpy(float)); names.append(c)
    for p in POSITION_LEVELS[1:]:
        arrays.append((x.position_group.astype(str) == p).to_numpy(float)); names.append("position_" + p)
    for s in [2023, 2024, 2025, 2026]:
        arrays.append(x.season.eq(s).to_numpy(float)); names.append("season_" + str(s))
    return np.column_stack(arrays), names


def cluster_fit(x: pd.DataFrame, columns: list[tuple[str, np.ndarray]]) -> tuple[pd.DataFrame, np.ndarray]:
    C, cn = controls(x); X = np.column_stack([C] + [a for _, a in columns]); names = cn + [n for n, _ in columns]
    y = x.actual_votes.to_numpy(float); beta = np.linalg.lstsq(X, y, rcond=None)[0]; u = y - X @ beta
    bread = np.linalg.pinv(X.T @ X); meat = np.zeros((X.shape[1], X.shape[1])); groups = x.match_id.to_numpy(); ug = np.unique(groups)
    for g in ug:
        xu = X[groups == g].T @ u[groups == g]; meat += np.outer(xu, xu)
    n, k, G = len(y), X.shape[1], len(ug); cov = (G/(G-1))*((n-1)/(n-k)) * bread @ meat @ bread
    se = np.sqrt(np.maximum(np.diag(cov), 0)); crit = stats.t.ppf(.975, G-1); p = 2*stats.t.sf(np.abs(beta/se), G-1)
    tab = pd.DataFrame({"term": names, "coefficient": beta, "std_error": se, "ci_low": beta-crit*se,
                        "ci_high": beta+crit*se, "p_raw": p, "n": n, "n_match_clusters": G})
    return tab, cov


def mechanism_models(x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        ("visible_index", x.visible_index.to_numpy()), ("placebo_index", x.placebo_index.to_numpy()),
        ("visible_x_post", (x.visible_index*x.post2026).to_numpy()),
        ("placebo_x_post", (x.placebo_index*x.post2026).to_numpy()),
    ]
    primary, cov = cluster_fit(x, cols)
    primary["family"] = "covariate"; primary["p_holm"] = primary.p_raw
    idx = primary.term.isin(["visible_x_post", "placebo_x_post"])
    primary.loc[idx, "family"] = "primary_interactions"; primary.loc[idx, "p_holm"] = holm(primary.loc[idx, "p_raw"])
    names = primary.term.tolist(); a, b = names.index("visible_x_post"), names.index("placebo_x_post")
    delta = primary.loc[a, "coefficient"] - primary.loc[b, "coefficient"]
    var = cov[a,a] + cov[b,b] - 2*cov[a,b]; se = np.sqrt(max(var,0)); dof=x.match_id.nunique()-1; crit=stats.t.ppf(.975,dof); p=2*stats.t.sf(abs(delta/se),dof)
    contrast = pd.DataFrame([{"term":"visible_minus_placebo_post_change", "coefficient":delta, "std_error":se,
                              "ci_low":delta-crit*se, "ci_high":delta+crit*se, "p_raw":p, "p_holm":p,
                              "family":"direct_mechanism_contrast", "n":len(x), "n_match_clusters":x.match_id.nunique()}])
    primary = pd.concat([primary, contrast], ignore_index=True)

    detail = []
    for family, features, other in [("visible",VISIBLE,"placebo_index"),("placebo",PLACEBO,"visible_index")]:
        for f in features:
            z = x[f + "_wmrank"]; z = (z-z.mean())/z.std()
            tab, _ = cluster_fit(x, [(f,z.to_numpy()), (other,x[other].to_numpy()), (f+"_x_post",(z*x.post2026).to_numpy())])
            row = tab[tab.term.eq(f+"_x_post")].copy(); row["feature"] = f; row["information_set"] = family; detail.append(row)
    detail = pd.concat(detail, ignore_index=True); detail["p_holm"] = detail.p_raw
    for fam, ids in detail.groupby("information_set").groups.items(): detail.loc[ids,"p_holm"] = holm(detail.loc[ids,"p_raw"])
    return primary, detail


def season_alignment(g: pd.DataFrame) -> dict:
    extreme_event = ((g.coaches_votes.eq(10)) & g.actual_votes.eq(0)) | ((g.actual_votes.eq(3)) & g.coaches_votes.eq(0))
    match_rows=[]
    for mid, q in g.groupby("match_id"):
        coach_top=set(q.sort_values(["coaches_votes","player_id"],ascending=[False,True]).head(3).player_id)
        brown_top=set(q[q.actual_votes.gt(0)].player_id)
        extreme=bool(((q.coaches_votes.eq(10))&q.actual_votes.eq(0)).any() or ((q.actual_votes.eq(3))&q.coaches_votes.eq(0)).any())
        rho=q[["coaches_votes","actual_votes"]].corr(method="spearman").iloc[0,1]
        match_rows.append((mid,len(coach_top&brown_top)/3,extreme,rho))
    m=pd.DataFrame(match_rows,columns=["match_id","top3_overlap","extreme_disagreement","match_rank_spearman"])
    return {"player_match_spearman":g[["coaches_votes","actual_votes"]].corr(method="spearman").iloc[0,1],
            "top3_overlap":m.top3_overlap.mean(), "extreme_disagreement_rate":m.extreme_disagreement.mean(),
            "match_rank_spearman":m.match_rank_spearman.mean(), "extreme_match_count":int(m.extreme_disagreement.sum()),
            "extreme_event_count":int(extreme_event.sum()), "extreme_events_per_match":float(extreme_event.sum()/len(m)), "n_matches":len(m)}


def weighted_corr(a: np.ndarray,b: np.ndarray,w: np.ndarray) -> float:
    sw=w.sum(); ma=(w*a).sum()/sw; mb=(w*b).sum()/sw
    return (w*(a-ma)*(b-mb)).sum()/np.sqrt((w*(a-ma)**2).sum()*(w*(b-mb)**2).sum())


def alignment_inference(x: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    annual=[]; cache={}
    for year,g in x.groupby("season"):
        pt=season_alignment(g); pt["season"]=year; annual.append(pt)
        gg=g.copy(); gg["crank"]=gg.coaches_votes.rank(method="average"); gg["vrank"]=gg.actual_votes.rank(method="average")
        matches=[]
        for mid,q in gg.groupby("match_id"):
            coach_top=set(q.sort_values(["coaches_votes","player_id"],ascending=[False,True]).head(3).player_id); brown=set(q[q.actual_votes.gt(0)].player_id)
            matches.append((mid,len(coach_top&brown)/3,float(((q.coaches_votes.eq(10)&q.actual_votes.eq(0)).any())|((q.actual_votes.eq(3)&q.coaches_votes.eq(0)).any())),q[["coaches_votes","actual_votes"]].corr(method="spearman").iloc[0,1]))
        cache[year]=(gg,pd.DataFrame(matches,columns=["match_id","top3_overlap","extreme_disagreement_rate","match_rank_spearman"]))
    annual=pd.DataFrame(annual).sort_values("season")
    rng=np.random.default_rng(SEED); boot={k:[] for k in ["player_match_spearman","top3_overlap","extreme_disagreement_rate","match_rank_spearman"]}
    for _ in range(REPS):
        vals={k:{} for k in boot}
        for year,(g,m) in cache.items():
            mids=m.match_id.to_numpy(); sampled=rng.choice(mids,len(mids),replace=True); counts=pd.Series(sampled).value_counts()
            w=g.match_id.map(counts).fillna(0).to_numpy(float)
            vals["player_match_spearman"][year]=weighted_corr(g.crank.to_numpy(),g.vrank.to_numpy(),w)
            mm=m.set_index("match_id"); ww=counts.reindex(mm.index).fillna(0).to_numpy(float)
            for metric in ["top3_overlap","extreme_disagreement_rate","match_rank_spearman"]: vals[metric][year]=np.average(mm[metric],weights=ww)
        for metric in boot: boot[metric].append(vals[metric][2026]-np.mean([vals[metric][y] for y in [2022,2023,2024,2025]]))
    rows=[]
    for metric,vals in boot.items():
        pre=annual[annual.season.lt(2026)][metric].mean(); post=annual.loc[annual.season.eq(2026),metric].iloc[0]; arr=np.asarray(vals)
        rows.append({"metric":metric,"pre_2022_2025_mean":pre,"post_2026":post,"delta":post-pre,
                     "ci_low":np.quantile(arr,.025),"ci_high":np.quantile(arr,.975),
                     "bootstrap_p_two_sided":2*min(np.mean(arr<=0),np.mean(arr>=0)),"bootstrap_reps":REPS})
    return annual, pd.DataFrame(rows)


def make_outputs(x, primary, detail, annual, align):
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    bas=pd.read_csv(ROOT/"data/derived/baselines.csv")
    table1=bas[bas.model.isin(["full","ridge","coach"])][["season","model","rmse","mae","three_vote_winner_hit_rate","top3_recall"]]
    table1.to_csv(OUT/"table1_predictive_instrument.csv",index=False)
    primary.to_csv(OUT/"mechanism_estimates.csv",index=False); detail.to_csv(OUT/"feature_level_interactions.csv",index=False); annual.to_csv(OUT/"alignment_by_season.csv",index=False)
    table2=pd.concat([
        primary.assign(section="visible_vs_placebo"),
        align.rename(columns={"metric":"term","delta":"coefficient"}).assign(section="coach_alignment"),
        detail.assign(section="feature_level_falsification")
    ],ignore_index=True,sort=False)
    table2.to_csv(OUT/"table2_robustness_falsification.csv",index=False)

    prov=pd.read_csv(ROOT/"data/derived/oot_provenance.csv")
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    axs[0].plot(annual.season,annual.player_match_spearman,"o-",color="#255f85"); axs[0].axvline(2025.5,color="grey",ls="--"); axs[0].set(title="Coach–umpire alignment",ylabel="Player-match Spearman",xticks=annual.season)
    axs[1].plot(annual.season,annual.extreme_event_count,"o-",color="#a14236"); axs[1].axvline(2025.5,color="grey",ls="--"); axs[1].set(title="Extreme-disagreement events",ylabel="Count",xticks=annual.season)
    axs[2].plot(prov.season,prov.rmse,"o-",color="#526c3f"); axs[2].axvline(2025.5,color="grey",ls="--"); axs[2].set(title="Expected-recognition RMSE",ylabel="RMSE",xticks=prov.season)
    fig.suptitle("Recognition before and after the 2026 information reform"); fig.tight_layout(); fig.savefig(FIG/"figure1_recognition_before_after.png",dpi=240); plt.close(fig)

    q=primary[primary.term.isin(["visible_x_post","placebo_x_post","visible_minus_placebo_post_change"])].copy()
    labels={"visible_x_post":"Visible statistics","placebo_x_post":"Non-visible placebo","visible_minus_placebo_post_change":"Visible − placebo"}
    fig,ax=plt.subplots(figsize=(7,3.8)); y=np.arange(len(q)); ax.errorbar(q.coefficient,y,xerr=[q.coefficient-q.ci_low,q.ci_high-q.coefficient],fmt="o",capsize=4,color="#255f85"); ax.axvline(0,color="black",lw=.8); ax.set(yticks=y,yticklabels=[labels[t] for t in q.term],xlabel="Change in vote sensitivity in 2026 (95% CI)"); fig.tight_layout(); fig.savefig(FIG/"figure2_visible_vs_placebo.png",dpi=240); plt.close(fig)


def main():
    x=load_panel(); primary,detail=mechanism_models(x); annual,align=alignment_inference(x); make_outputs(x,primary,detail,annual,align)
    print("PRIMARY\n",primary[primary.term.isin(["visible_x_post","placebo_x_post","visible_minus_placebo_post_change"])].to_string(index=False))
    print("\nALIGNMENT\n",align.to_string(index=False)); print("\nANNUAL EXTREME\n",annual[["season","extreme_event_count","extreme_match_count","player_match_spearman"]].to_string(index=False))


if __name__=="__main__": main()
