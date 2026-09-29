#!/usr/bin/env python3
"""Execute the locked SSAC27 final-identification protocol."""
from __future__ import annotations
from pathlib import Path
import sys, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT; HERE=ROOT; OUT=ROOT/"results"; FIG=ROOT/"paper/figures"
sys.path.insert(0,str(ROOT/"src"))
from run_information_reform import VISIBLE, PLACEBO, load_panel, cluster_fit, weighted_corr  # noqa

def extract_model(x, visible="visible_index", placebo="placebo_index", target=2026):
    post=x.season.eq(target).astype(float)
    tab,cov=cluster_fit(x,[("visible",x[visible].to_numpy()),("placebo",x[placebo].to_numpy()),("visible_change",(x[visible]*post).to_numpy()),("placebo_change",(x[placebo]*post).to_numpy())])
    names=tab.term.tolist(); a=names.index("visible_change"); b=names.index("placebo_change"); d=tab.loc[a,"coefficient"]-tab.loc[b,"coefficient"]
    se=np.sqrt(max(cov[a,a]+cov[b,b]-2*cov[a,b],0)); dof=x.match_id.nunique()-1; crit=stats.t.ppf(.975,dof)
    return tab,cov,{"visible_change":tab.loc[a,"coefficient"],"visible_ci_low":tab.loc[a,"ci_low"],"visible_ci_high":tab.loc[a,"ci_high"],
      "placebo_change":tab.loc[b,"coefficient"],"placebo_ci_low":tab.loc[b,"ci_low"],"placebo_ci_high":tab.loc[b,"ci_high"],
      "visible_minus_placebo":d,"contrast_ci_low":d-crit*se,"contrast_ci_high":d+crit*se,"contrast_se":se}

def pseudo_reforms(x):
    rows=[]
    align=pd.read_csv(BASE/"data/derived/information_reform/alignment_by_season.csv").set_index("season")
    for year in [2023,2024,2025,2026]:
        q=x[x.season.le(year)].copy(); _,_,r=extract_model(q,target=year)
        pre=align.loc[[y for y in range(2022,year)],"player_match_spearman"].mean(); post=align.loc[year,"player_match_spearman"]
        r.update(intervention_year=year,is_actual_reform=year==2026,n_player_matches=len(q),n_matches=q.match_id.nunique(),coach_alignment_change=post-pre)
        rows.append(r)
    z=pd.DataFrame(rows); actual=z.loc[z.intervention_year.eq(2026),"visible_change"].iloc[0]; pse=z[~z.is_actual_reform].visible_change
    z["actual_visible_rank_among_four"]=(z.visible_change.rank(ascending=False,method="min").where(z.is_actual_reform))
    z["actual_minus_max_pseudo_visible"]=np.where(z.is_actual_reform,actual-pse.max(),np.nan)
    return z

def identification_diagnostics(x,primary_tab,cov):
    rows=[]
    for label,q in [("full_2022_2026",x)]+[(str(y),x[x.season.eq(y)]) for y in range(2022,2027)]:
        a=q.visible_index; b=q.placebo_index; r=a.corr(b); rows += [
          {"scope":label,"diagnostic":"correlation","value":r},{"scope":label,"diagnostic":"covariance","value":a.cov(b)},
          {"scope":label,"diagnostic":"visible_variance","value":a.var()},{"scope":label,"diagnostic":"placebo_variance","value":b.var()},
          {"scope":label,"diagnostic":"two_predictor_vif","value":1/(1-r*r)}]
    # Effective independent variation after projecting visible on placebo plus locked controls.
    C=np.column_stack([np.ones(len(x)),x.placebo_index,x.team_won]+[x.season.eq(y) for y in [2023,2024,2025,2026]])
    resid=x.visible_index.to_numpy()-C@np.linalg.lstsq(C,x.visible_index.to_numpy(),rcond=None)[0]
    rows.append({"scope":"full_2022_2026","diagnostic":"visible_residual_variance_share","value":np.var(resid)/np.var(x.visible_index)})
    names=primary_tab.term.tolist(); a=names.index("visible_change"); b=names.index("placebo_change")
    rows += [{"scope":"interaction_covariance","diagnostic":"var_visible_change","value":cov[a,a]},
             {"scope":"interaction_covariance","diagnostic":"var_placebo_change","value":cov[b,b]},
             {"scope":"interaction_covariance","diagnostic":"cov_visible_placebo_changes","value":cov[a,b]},
             {"scope":"interaction_covariance","diagnostic":"corr_visible_placebo_changes","value":cov[a,b]/np.sqrt(cov[a,a]*cov[b,b])},
             {"scope":"interaction_covariance","diagnostic":"var_direct_contrast","value":cov[a,a]+cov[b,b]-2*cov[a,b]}]
    return pd.DataFrame(rows)

def add_z_composites(x):
    x=x.copy()
    for c in VISIBLE+PLACEBO:
        z=x.groupby(["season","match_id"])[c].transform(lambda s:(s-s.mean())/(s.std(ddof=0) or 1))
        x[c+"_wmz"]=z.fillna(0)
    pre=x.season.lt(2026)
    for fam,features in [("visible",VISIBLE),("placebo",PLACEBO)]:
        A=x[[c+"_wmz" for c in features]].to_numpy(); eq=A.mean(1); x[f"{fam}_r1"]=(eq-eq.mean())/eq.std()
        pca=PCA(n_components=1).fit(A[pre]); pc=pca.transform(A).ravel()
        if np.corrcoef(pc,eq)[0,1]<0: pc=-pc
        x[f"{fam}_r2"]=(pc-pc[pre].mean())/pc[pre].std()
        scaler=StandardScaler().fit(A[pre]); As=scaler.transform(A); ridge=Ridge(alpha=10).fit(As[pre],x.loc[pre,"actual_votes"]); pred=ridge.predict(As)
        x[f"{fam}_r3"]=(pred-pred[pre].mean())/pred[pre].std()
    return x

def composite_robustness(x):
    q=add_z_composites(x); rows=[]
    for label,v,p in [("R1_equal_weight_within_match_z","visible_r1","placebo_r1"),("R2_pre2026_PCA1","visible_r2","placebo_r2"),("R3_pre2026_ridge_weights","visible_r3","placebo_r3")]:
        _,_,r=extract_model(q,v,p,2026); r.update(construction=label,n_player_matches=len(q),n_matches=q.match_id.nunique()); rows.append(r)
    return pd.DataFrame(rows),q

def leave_one_out(x):
    rows=[]; primary=extract_model(x)[2]["visible_change"]
    for rem in VISIBLE:
        cols=[c+"_wmrank" for c in VISIBLE if c!=rem]; raw=x[cols].mean(1); name="loo_index"; q=x.copy(); q[name]=(raw-raw.mean())/raw.std()
        _,_,r=extract_model(q,name,"placebo_index",2026); rows.append({"removed_feature":rem,"visible_change":r["visible_change"],"ci_low":r["visible_ci_low"],"ci_high":r["visible_ci_high"],"change_from_primary":r["visible_change"]-primary})
    return pd.DataFrame(rows)

def feature_event_study(x):
    rows=[]
    for f in VISIBLE:
        z=x[f+"_wmrank"]; z=(z-z.mean())/z.std()
        for year in range(2022,2027):
            q=x[x.season.eq(year)].copy(); zz=z.loc[q.index]
            tab,_=cluster_fit(q,[("feature",zz.to_numpy()),("placebo_index",q.placebo_index.to_numpy())]); r=tab[tab.term.eq("feature")].iloc[0]
            rows.append({"feature":f,"family":("Possession" if f in ["disposals","kicks","handballs"] else "Contest" if f in ["contested_possessions","clearances"] else "Scoring" if f in ["goals","score_involvements"] else "Other"),"season":year,"coefficient":r.coefficient,"std_error":r.std_error,"ci_low":r.ci_low,"ci_high":r.ci_high,"p_raw":r.p_raw})
    return pd.DataFrame(rows)

def alignment_robustness(x,reps=2000):
    annual=[]; caches={}
    for year,g in x.groupby("season"):
        match=[]
        for mid,q in g.groupby("match_id"):
            coach=q.sort_values(["coaches_votes","player_id"],ascending=[False,True]); brown=q.sort_values(["actual_votes","player_id"],ascending=[False,True])
            topc=set(coach.head(3).player_id); topb=set(q[q.actual_votes.gt(0)].player_id); winagree=float(coach.iloc[0].player_id==brown.iloc[0].player_id)
            extreme=float((((q.coaches_votes==10)&(q.actual_votes==0))|((q.actual_votes==3)&(q.coaches_votes==0))).any())
            match.append((mid,len(topc&topb)/3,winagree,extreme,q[["coaches_votes","actual_votes"]].corr(method="spearman").iloc[0,1]))
        m=pd.DataFrame(match,columns=["match_id","top3_overlap","winner_agreement","extreme_disagreement_rate","within_match_rank_correlation"])
        pm=g[["coaches_votes","actual_votes"]].corr(method="spearman").iloc[0,1]
        annual.append({"season":year,"player_match_spearman":pm,**m.drop(columns="match_id").mean().to_dict(),"n_matches":len(m)})
        gg=g.copy(); gg["crank"]=gg.coaches_votes.rank(method="average"); gg["vrank"]=gg.actual_votes.rank(method="average"); caches[year]=(gg,m)
    annual=pd.DataFrame(annual); metrics=["player_match_spearman","within_match_rank_correlation","top3_overlap","winner_agreement","extreme_disagreement_rate"]
    rng=np.random.default_rng(42); boots={k:[] for k in metrics}
    for _ in range(reps):
        vals={k:{} for k in metrics}
        for y,(g,m) in caches.items():
            mids=m.match_id.to_numpy(); counts=pd.Series(rng.choice(mids,len(mids),replace=True)).value_counts(); w=g.match_id.map(counts).fillna(0).to_numpy()
            vals["player_match_spearman"][y]=weighted_corr(g.crank.to_numpy(),g.vrank.to_numpy(),w)
            mm=m.set_index("match_id"); ww=counts.reindex(mm.index).fillna(0).to_numpy()
            for k in metrics[1:]: vals[k][y]=np.average(mm[k],weights=ww)
        for k in metrics: boots[k].append(vals[k][2026]-np.mean([vals[k][y] for y in [2022,2023,2024,2025]]))
    rows=[]
    for _,r in annual.iterrows():
        for k in metrics: rows.append({"comparison":"annual_level","metric":k,"season":int(r.season),"estimate":r[k],"ci_low":np.nan,"ci_high":np.nan})
    for k,arr in boots.items():
        a=np.asarray(arr); pre=annual[annual.season<2026][k].mean(); post=annual[annual.season==2026][k].iloc[0]
        rows.append({"comparison":"2026_vs_equal_weight_pre_years","metric":k,"season":2026,"estimate":post-pre,"pre_reference":pre,"post_2026":post,"ci_low":np.quantile(a,.025),"ci_high":np.quantile(a,.975),"bootstrap_reps":reps})
    return pd.DataFrame(rows),annual

def pooled_validation():
    b=pd.read_csv(BASE/"data/derived/baselines.csv"); rows=[]
    groups={"2022-2025_OOT":[2022,2023,2024,2025],"2026_frozen":[2026],"2022-2026_overall":[2022,2023,2024,2025,2026]}
    nrows={2022:9108,2023:9522,2024:9522,2025:9476,2026:9522}; nm={2022:198,2023:207,2024:207,2025:206,2026:207}
    for period,years in groups.items():
      for model,g in b[b.season.isin(years)].groupby("model"):
        w=np.array([nrows[y] for y in g.season]); mw=np.array([nm[y] for y in g.season])
        rows.append({"period":period,"model":model,"rmse":np.sqrt(np.average(g.rmse**2,weights=w)),"mae":np.average(g.mae,weights=w),"three_vote_hit_rate":np.average(g.three_vote_winner_hit_rate,weights=mw),"top3_recall":np.average(g.top3_recall,weights=mw),"n_player_matches":w.sum(),"n_matches":mw.sum()})
    return pd.DataFrame(rows)

def disagreement_reconciliation(x):
    official={
      "Connor Macdonald","Harry Sheezel","Sam Flanders","Touk Miller","Zak Butters","Murphy Reid","Sam Walsh","Bailey Smith",
      "Wayne Milera","Charlie Comben","Kade Chandler","Jordan Dawson","Jarman Impey","Esava Ratugolea","Max Holmes","Caleb Daniel",
      "Jai Newcombe","Harris Andrews","Karl Worner","Francis Evans","Beau McCreery"}
    g=x[x.season.eq(2026)].copy(); mask=((g.coaches_votes.eq(10)&g.actual_votes.eq(0))|(g.actual_votes.eq(3)&g.coaches_votes.eq(0)))
    z=g[mask].copy(); z["rule_triggered"]=np.where(z.coaches_votes.eq(10),"10 coach / 0 Brownlow","3 Brownlow / 0 coach")
    # Murphy Reid and Sam Walsh appear twice officially; Caleb Daniel appears officially only in R15.
    z["official_article_match"]=z.player_name.isin(official)
    z.loc[(z.player_name.eq("Caleb Daniel")) & (z.match_id.eq(17088)),"official_article_match"]=False
    z["possible_explanation"]=np.where(z.official_article_match,"Listed by AFL article","Extra local event: Caleb Daniel received 10 recorded coach votes v Carlton in R4 but is absent from AFL article")
    return z[["match_id","player_name","player_team","coaches_votes","actual_votes","rule_triggered","official_article_match","possible_explanation"]]

def make_figures(pseudo,event,align,validation,primary):
    FIG.mkdir(parents=True,exist_ok=True)
    fig,axs=plt.subplots(1,3,figsize=(13,4.2))
    ann=align[align.comparison.eq("annual_level")&align.metric.eq("player_match_spearman")]; axs[0].plot(ann.season,ann.estimate,"o-",color="#245b78"); axs[0].axvline(2026,color="#9b3a32",lw=1.5); axs[0].set(title="A. Coach–umpire alignment",ylabel="Player-match Spearman",xticks=range(2022,2027))
    axs[1].errorbar(pseudo.intervention_year,pseudo.visible_change,yerr=[pseudo.visible_change-pseudo.visible_ci_low,pseudo.visible_ci_high-pseudo.visible_change],fmt="o",capsize=3,color="#555555"); axs[1].scatter([2026],pseudo.loc[pseudo.intervention_year.eq(2026),"visible_change"],color="#9b3a32",zorder=3); axs[1].axhline(0,color="black",lw=.7); axs[1].set(title="B. Actual and pseudo reforms",ylabel="Visible-stat sensitivity change",xticks=range(2023,2027))
    full=validation[(validation.model.eq("full"))]; axs[2].plot([2022,2023,2024,2025,2026],pd.read_csv(BASE/"data/derived/oot_provenance.csv").rmse,"o-",color="#526c3f"); axs[2].axvline(2026,color="#9b3a32",lw=1.5); axs[2].set(title="C. Expected-recognition error",ylabel="RMSE",xticks=range(2022,2027))
    fig.suptitle("Recognition around the 2026 information reform"); fig.tight_layout(); fig.savefig(FIG/"FIGURE_1_recognition_around_reform.png",dpi=300); fig.savefig(FIG/"FIGURE_1_recognition_around_reform.pdf"); plt.close(fig)
    q=primary.set_index("term").loc[["visible_change","placebo_change","visible_minus_placebo"]].reset_index(); labels=["Visible statistics","Non-visible placebo","Visible − placebo"]
    fig,ax=plt.subplots(figsize=(7.2,3.8)); y=np.arange(3); ax.errorbar(q.coefficient,y,xerr=[q.coefficient-q.ci_low,q.ci_high-q.coefficient],fmt="o",capsize=4,color="#245b78"); ax.axvline(0,color="black",lw=.8); ax.set(yticks=y,yticklabels=labels,xlabel="Change in voting sensitivity (95% CI)"); ax.invert_yaxis(); fig.tight_layout(); fig.savefig(FIG/"FIGURE_2_change_in_voting_sensitivity.png",dpi=300); fig.savefig(FIG/"FIGURE_2_change_in_voting_sensitivity.pdf"); plt.close(fig)
    chosen=event[event.family.isin(["Possession","Contest","Scoring"])]
    fig,axs=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,(fam,g) in zip(axs,chosen.groupby("family",sort=False)):
      for f,h in g.groupby("feature"): ax.plot(h.season,h.coefficient,"o-",label=f)
      ax.axvline(2026,color="#9b3a32",lw=1.2); ax.axhline(0,color="black",lw=.6); ax.set(title=fam,xticks=range(2022,2027)); ax.legend(frameon=False,fontsize=8)
    axs[0].set_ylabel("Vote-sensitivity coefficient"); fig.suptitle("Feature-level event study (predefined families)"); fig.tight_layout(); fig.savefig(FIG/"feature_event_study.png",dpi=300); fig.savefig(FIG/"feature_event_study.pdf"); plt.close(fig)

def main():
    OUT.mkdir(parents=True,exist_ok=True); x=load_panel()
    primary_tab,cov,pr=extract_model(x); primary=pd.DataFrame([
      {"term":"visible_change","coefficient":pr["visible_change"],"ci_low":pr["visible_ci_low"],"ci_high":pr["visible_ci_high"]},
      {"term":"placebo_change","coefficient":pr["placebo_change"],"ci_low":pr["placebo_ci_low"],"ci_high":pr["placebo_ci_high"]},
      {"term":"visible_minus_placebo","coefficient":pr["visible_minus_placebo"],"ci_low":pr["contrast_ci_low"],"ci_high":pr["contrast_ci_high"]}])
    pseudo=pseudo_reforms(x); pseudo.to_csv(OUT/"pseudo_reform_results.csv",index=False)
    diag=identification_diagnostics(x,primary_tab,cov); diag.to_csv(OUT/"composite_identification_diagnostics.csv",index=False)
    corr=x[[c+"_wmrank" for c in VISIBLE+PLACEBO]].corr(); corr.to_csv(OUT/"feature_cross_correlation_matrix.csv")
    fig,ax=plt.subplots(figsize=(10,8)); im=ax.imshow(corr.loc[[c+"_wmrank" for c in VISIBLE],[c+"_wmrank" for c in PLACEBO]],vmin=-1,vmax=1,cmap="coolwarm"); ax.set_xticks(range(len(PLACEBO)),PLACEBO,rotation=45,ha="right",fontsize=7); ax.set_yticks(range(len(VISIBLE)),VISIBLE,fontsize=7); fig.colorbar(im,ax=ax,label="Correlation"); fig.tight_layout(); fig.savefig(FIG/"composite_correlation_heatmap.png",dpi=300); plt.close(fig)
    robust,xz=composite_robustness(x); robust.to_csv(OUT/"composite_robustness.csv",index=False)
    loo=leave_one_out(x); loo.to_csv(OUT/"leave_one_visible_feature_out.csv",index=False)
    event=feature_event_study(x); event.to_csv(OUT/"feature_event_study.csv",index=False)
    sal=event[event.family.isin(["Possession","Contest","Scoring"])].pivot(index=["feature","family"],columns="season",values="coefficient").reset_index(); sal["change_2026_vs_2022_2025_mean"]=sal[2026]-sal[[2022,2023,2024,2025]].mean(axis=1); sal.to_csv(OUT/"exploratory_salience_descriptive.csv",index=False)
    align,annual=alignment_robustness(x); align.to_csv(OUT/"coach_umpire_alignment_robustness.csv",index=False)
    validation=pooled_validation(); validation.to_csv(OUT/"expected_recognition_validation.csv",index=False)
    dev=pd.read_csv(BASE/"data/derived/development_regressions.csv"); con=pd.read_csv(BASE/"data/derived/confirmatory_2026.csv"); rob=pd.read_csv(BASE/"data/derived/robustness.csv")
    rep=pd.concat([dev[(dev.specification.eq("D4_full"))&dev.term.eq("prior_season_residual")].assign(result="Development 2023-2025"),con[con.term.eq("prior_season_residual")].assign(result="Frozen 2026"),rob[rob.analysis.isin(["prior_actual","career_vpg","player_FE_within_development"])].assign(result=lambda d:d.analysis)],ignore_index=True); rep.to_csv(OUT/"reputation_non_replication.csv",index=False)
    pre=[]
    for start in [2022,2023]:
      q=x[x.season.ge(start)]; _,_,r=extract_model(q,target=2026); pre.append({"pre_period":f"{start}-2025","visible_change":r["visible_change"],"visible_ci_low":r["visible_ci_low"],"visible_ci_high":r["visible_ci_high"],"placebo_change":r["placebo_change"],"contrast":r["visible_minus_placebo"],"contrast_ci_low":r["contrast_ci_low"],"contrast_ci_high":r["contrast_ci_high"],"coach_spearman_pre_mean":annual[annual.season.between(start,2025)].player_match_spearman.mean(),"coach_spearman_2026":annual.loc[annual.season.eq(2026),"player_match_spearman"].iloc[0],"extreme_rate_pre_mean":annual[annual.season.between(start,2025)].extreme_disagreement_rate.mean(),"extreme_rate_2026":annual.loc[annual.season.eq(2026),"extreme_disagreement_rate"].iloc[0]})
    pd.DataFrame(pre).to_csv(OUT/"preperiod_sensitivity.csv",index=False)
    rec=disagreement_reconciliation(x); rec.to_csv(OUT/"official_disagreement_reconciliation.csv",index=False)
    make_figures(pseudo,event,align,validation,primary)
    # Compact abstract table.
    v26=validation[(validation.period.eq("2026_frozen"))&validation.model.eq("full")].iloc[0]; ar=align[(align.comparison.str.startswith("2026"))].set_index("metric")
    absrows=[("Frozen 2026 RMSE",v26.rmse,np.nan,np.nan),("Frozen 2026 3-vote hit",v26.three_vote_hit_rate,np.nan,np.nan),("Coach–umpire Spearman change",ar.loc["player_match_spearman","estimate"],ar.loc["player_match_spearman","ci_low"],ar.loc["player_match_spearman","ci_high"]),("Top-3 overlap change",ar.loc["top3_overlap","estimate"],ar.loc["top3_overlap","ci_low"],ar.loc["top3_overlap","ci_high"]),("Visible-stat sensitivity change",pr["visible_change"],pr["visible_ci_low"],pr["visible_ci_high"]),("Non-visible placebo change",pr["placebo_change"],pr["placebo_ci_low"],pr["placebo_ci_high"]),("Visible-minus-placebo",pr["visible_minus_placebo"],pr["contrast_ci_low"],pr["contrast_ci_high"]),("2026 minus max pseudo visible change",pseudo.loc[pseudo.is_actual_reform,"actual_minus_max_pseudo_visible"].iloc[0],np.nan,np.nan)]
    pd.DataFrame(absrows,columns=["result","estimate","ci_low","ci_high"]).to_csv(OUT/"SSAC_ABSTRACT_RESULTS.csv",index=False)
    # Machine-readable diagnostic conclusion.
    summary={"actual_visible_change":pr,"actual_visible_rank":int(pseudo.loc[pseudo.is_actual_reform,"actual_visible_rank_among_four"].iloc[0]),"loo_min":loo.visible_change.min(),"loo_max":loo.visible_change.max(),"loo_median":loo.visible_change.median(),"loo_sign_reversal":bool((loo.visible_change<0).any()),"composite_robustness":robust.to_dict("records"),"extra_disagreement_event":rec.loc[~rec.official_article_match,"player_name"].tolist()}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2))
    print("PSEUDO\n",pseudo.to_string(index=False)); print("\nROBUST\n",robust.to_string(index=False)); print("\nLOO RANGE",summary["loo_min"],summary["loo_median"],summary["loo_max"],"reverse",summary["loo_sign_reversal"]); print("\nDIAG\n",diag.to_string(index=False)); print("\nEXTRA\n",rec[~rec.official_article_match].to_string(index=False))

if __name__=="__main__": main()
