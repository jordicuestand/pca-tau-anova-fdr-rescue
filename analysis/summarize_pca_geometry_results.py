#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, itertools
from pathlib import Path
import numpy as np
import pandas as pd

def smean(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    return float(x.mean()) if len(x) else np.nan

def ssd(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    return float(x.std(ddof=1)) if len(x) > 1 else np.nan

def abs_cos(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    na=np.linalg.norm(a); nb=np.linalg.norm(b)
    return np.nan if na==0 or nb==0 else abs(float(np.dot(a,b)/(na*nb)))

def loading_stability(df):
    vecs=[]
    for _,g in df.groupby(["repeat","outer_fold"]):
        vecs.append(g.sort_values("feature_index_1based")["loading"].to_numpy(float))
    vals=[abs_cos(a,b) for a,b in itertools.combinations(vecs,2) if len(a)==len(b)]
    vals=np.asarray([v for v in vals if np.isfinite(v)])
    return (float(vals.mean()), float(vals.std(ddof=1)) if len(vals)>1 else np.nan) if len(vals) else (np.nan,np.nan)

def jaccard_sets(sets):
    vals=[]
    for a,b in itertools.combinations(sets,2):
        u=a|b
        vals.append(1.0 if not u else len(a&b)/len(u))
    return np.asarray(vals,float)

def reconstruct_covs(eigen, loads):
    covs={}
    for (rep,fold), es in eigen.groupby(["repeat","outer_fold"]):
        es=es.sort_values("pc_1based")
        rows=[]
        ok=True
        for pc in es["pc_1based"].astype(int):
            g=loads[(loads.repeat==rep)&(loads.outer_fold==fold)&(loads.pc_1based==pc)]
            g=g.sort_values("feature_index_1based")
            if len(g)==0:
                ok=False; break
            rows.append(g["loading"].to_numpy(float))
        if ok:
            V=np.vstack(rows)
            lam=es["explained_variance"].to_numpy(float)
            if V.shape[0]==len(lam):
                covs[(rep,fold)]=V.T@np.diag(lam)@V
    return covs

def frob_dists(covs):
    vals=[]
    for k1,k2 in itertools.combinations(sorted(covs),2):
        A,B=covs[k1],covs[k2]
        den=0.5*(np.linalg.norm(A,"fro")+np.linalg.norm(B,"fro"))
        if den>0: vals.append(np.linalg.norm(A-B,"fro")/den)
    return np.asarray(vals,float)

def spectrum_dists(eigen):
    specs=[]
    for _,g in eigen.groupby(["repeat","outer_fold"]):
        specs.append(g.sort_values("pc_1based")["explained_variance_ratio"].to_numpy(float))
    vals=[]
    for a,b in itertools.combinations(specs,2):
        m=min(len(a),len(b))
        if m: vals.append(np.linalg.norm(a[:m]-b[:m]))
    return np.asarray(vals,float)

def subspace_similarity(loads,k):
    bases=[]
    for _,g in loads.groupby(["repeat","outer_fold"]):
        rows=[]
        for pc in range(1,k+1):
            h=g[g.pc_1based==pc].sort_values("feature_index_1based")
            if len(h)==0: rows=[]; break
            rows.append(h["loading"].to_numpy(float))
        if rows: bases.append(np.vstack(rows))
    vals=[]
    for A,B in itertools.combinations(bases,2):
        if A.shape==B.shape:
            s=np.linalg.svd(A@B.T,compute_uv=False)
            vals.append(float(np.clip(s,0,1).mean()))
    vals=np.asarray(vals,float)
    return (float(vals.mean()), float(vals.std(ddof=1)) if len(vals)>1 else np.nan) if len(vals) else (np.nan,np.nan)

def perf_summary(path):
    out={}
    if not path.exists(): return out
    r=pd.read_csv(path)
    for metric in ["auc","mcc","f1"]:
        if metric not in r.columns: continue
        for method in ["PCA","HYBRID"]:
            x=r.loc[r.method==method,metric]
            out[f"{method.lower()}_{metric}_mean"]=smean(x)
            out[f"{method.lower()}_{metric}_sd"]=ssd(x)
        p=r[r.method=="PCA"].set_index(["repeat","outer_fold"])
        h=r[r.method=="HYBRID"].set_index(["repeat","outer_fold"])
        common=p.index.intersection(h.index)
        if len(common):
            d=h.loc[common,metric]-p.loc[common,metric]
            out[f"delta_{metric}_mean"]=smean(d)
            out[f"delta_{metric}_sd"]=ssd(d)
    return out

def summarize(indir,outdir,ds):
    diag=pd.read_csv(indir/f"{ds}_all_pc_diagnostics.csv")
    loads=pd.read_csv(indir/f"{ds}_all_pc_loadings.csv")
    eigen=pd.read_csv(indir/f"{ds}_eigen_spectrum.csv")
    fold=pd.read_csv(indir/f"{ds}_fold_geometry.csv")

    pcrows=[]
    for pc,g in diag.groupby("pc_1based"):
        lg=loads[loads.pc_1based==pc]
        lm,ls=loading_stability(lg)
        sig=g["is_anova_significant"].astype(bool)
        res=g["is_rescued"].astype(bool)
        pcrows.append({
            "dataset":ds,"pc_1based":int(pc),"folds":len(g),
            "significant_rate":float(sig.mean()),"rescue_rate":float(res.mean()),
            "explained_variance_ratio_mean":smean(g.explained_variance_ratio),
            "relative_eigengap_mean":smean(g.relative_eigengap),
            "relative_eigengap_sd":ssd(g.relative_eigengap),
            "abs_cohens_d_mean":smean(g.abs_cohens_d),
            "abs_cohens_d_sd":ssd(g.abs_cohens_d),
            "q_median":float(pd.to_numeric(g.q_value,errors="coerce").median()),
            "q_sd":ssd(g.q_value),
            "loading_abs_cosine_mean":lm,"loading_abs_cosine_sd":ls
        })
    pcs=pd.DataFrame(pcrows)

    sets=[]
    for _,g in diag.groupby(["repeat","outer_fold"]):
        sets.append(set(g.loc[g.is_rescued.astype(bool),"pc_1based"].astype(int)))
    jac=jaccard_sets(sets)

    covd=frob_dists(reconstruct_covs(eigen,loads))
    eigd=spectrum_dists(eigen)

    outside=diag[~diag.inside_pca_tau.astype(bool)]
    rescued=diag[diag.is_rescued.astype(bool)]

    around=[]
    for _,g in diag.groupby(["repeat","outer_fold"]):
        ntau=int(g.n_pca_tau.iloc[0])
        for pc in [ntau,ntau+1]:
            h=g[g.pc_1based==pc]
            around += pd.to_numeric(h.relative_eigengap,errors="coerce").dropna().tolist()

    s={
        "dataset":ds,
        "n_folds":len(fold),
        "n_pca_tau_mean":smean(fold.n_pca_tau),
        "n_pca_tau_sd":ssd(fold.n_pca_tau),
        "effective_rank_mean":smean(fold.effective_rank),
        "effective_rank_sd":ssd(fold.effective_rank),
        "spectral_entropy_mean":smean(fold.spectral_entropy),
        "spectral_entropy_sd":ssd(fold.spectral_entropy),
        "tail_variance_after_tau_mean":smean(fold.tail_variance_after_tau),
        "relative_eigengap_after_tau_mean":smean(outside.relative_eigengap),
        "relative_eigengap_around_tau_mean":smean(around),
        "relative_eigengap_rescued_mean":smean(rescued.relative_eigengap),
        "n_rescued_mean":smean(fold.n_rescued),
        "n_rescued_sd":ssd(fold.n_rescued),
        "rescue_fold_rate":float((pd.to_numeric(fold.n_rescued)>0).mean()),
        "rescue_jaccard_mean":float(jac.mean()) if len(jac) else np.nan,
        "rescue_jaccard_median":float(np.median(jac)) if len(jac) else np.nan,
        "covariance_distance_mean":float(covd.mean()) if len(covd) else np.nan,
        "covariance_distance_sd":float(covd.std(ddof=1)) if len(covd)>1 else np.nan,
        "eigen_spectrum_distance_mean":float(eigd.mean()) if len(eigd) else np.nan,
        "eigen_spectrum_distance_sd":float(eigd.std(ddof=1)) if len(eigd)>1 else np.nan,
        "rescued_abs_cohens_d_mean":smean(rescued.abs_cohens_d),
        "rescued_q_median":float(pd.to_numeric(rescued.q_value,errors="coerce").median()) if len(rescued) else np.nan,
        "rescued_variance_ratio_mean":smean(rescued.explained_variance_ratio)
    }

    maxpc=int(loads.pc_1based.max())
    for k in [1,2,3,4,5,6,10]:
        if maxpc>=k:
            m,sd=subspace_similarity(loads,k)
            s[f"subspace{k}_similarity_mean"]=m
            s[f"subspace{k}_similarity_sd"]=sd
    for pc in range(1,min(10,maxpc)+1):
        m,sd=loading_stability(loads[loads.pc_1based==pc])
        s[f"pc{pc}_loading_similarity_mean"]=m
        s[f"pc{pc}_loading_similarity_sd"]=sd

    s.update(perf_summary(indir/f"{ds}_nested_results.csv"))
    sdf=pd.DataFrame([s])

    outdir.mkdir(parents=True,exist_ok=True)
    sdf.to_csv(outdir/f"{ds}_geometry_summary.csv",index=False)
    pcs.to_csv(outdir/f"{ds}_pc_summary.csv",index=False)
    return sdf,pcs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--indir",required=True)
    ap.add_argument("--outdir",default=None)
    ap.add_argument("--datasets",nargs="*",default=None)
    args=ap.parse_args()

    indir=Path(args.indir).expanduser().resolve()
    outdir=Path(args.outdir).expanduser().resolve() if args.outdir else indir/"geometry_summaries"
    datasets=args.datasets or sorted(p.name.replace("_all_pc_diagnostics.csv","") for p in indir.glob("*_all_pc_diagnostics.csv"))
    if not datasets: raise RuntimeError("No *_all_pc_diagnostics.csv found")

    gs=[]; gps=[]
    for ds in datasets:
        print("Summarizing",ds)
        s,p=summarize(indir,outdir,ds)
        gs.append(s); gps.append(p)

    pd.concat(gs,ignore_index=True).to_csv(outdir/"geometry_global_summary.csv",index=False)
    pd.concat(gps,ignore_index=True).to_csv(outdir/"pc_global_summary.csv",index=False)
    print("Done:",outdir)

if __name__=="__main__":
    main()
