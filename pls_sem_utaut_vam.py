#!/usr/bin/env python3
"""
PLS-SEM for the integrated UTAUT2-VAM biogas-adoption model (reflective, Mode A,
path-weighting scheme), written from scratch with numpy/pandas/scipy.

Usage
-----
    python pls_sem_utaut_vam.py dataset.xlsx --sheet 2_Data_Kuesioner --boot 5000 --seed 42 --out results
    python pls_sem_utaut_vam.py --selftest          # parameter-recovery check on simulated data

Outputs (Excel workbook + console summary)
    loadings, reliability (alpha, rho_A, CR, AVE), Fornell-Larcker, HTMT, cross-loadings,
    outer/inner VIF, R2/adj-R2, f2, paths (bootstrap SE, t, p, percentile CI),
    specific indirect effects H19-H25 (H24 = two specific paths + total),
    Harman test, full-collinearity VIF, regional ANOVA + Tukey HSD, herd-size
    correlations and Keil et al. (2000) parametric multigroup test.

Not implemented (do these before comparing groups): MICOM, PLSpredict/CVPAT, SRMR.
"""
import argparse
import re
import sys
import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------- model
CONSTRUCTS = ["PE", "EE", "SI", "FC", "HM", "PV", "HT", "PB", "PS", "NV", "BI"]
# structural model: endogenous -> predictors  (H1-H18)
PATHS = {
    "PB": ["PE", "SI", "HM"],                                        # H1-H3
    "PS": ["EE", "FC", "PV"],                                        # H4-H6
    "NV": ["PB", "PS", "HM", "HT"],                                  # H7-H10
    "BI": ["NV", "PE", "EE", "SI", "FC", "HM", "PV", "HT"],          # H11-H18
}
HYP = {("PE", "PB"): "H1", ("SI", "PB"): "H2", ("HM", "PB"): "H3",
       ("EE", "PS"): "H4", ("FC", "PS"): "H5", ("PV", "PS"): "H6",
       ("PB", "NV"): "H7", ("PS", "NV"): "H8", ("HM", "NV"): "H9", ("HT", "NV"): "H10",
       ("NV", "BI"): "H11", ("PE", "BI"): "H12", ("EE", "BI"): "H13", ("SI", "BI"): "H14",
       ("FC", "BI"): "H15", ("HM", "BI"): "H16", ("PV", "BI"): "H17", ("HT", "BI"): "H18"}
# specific indirect effects (H19-H25); H24 = sum of two specific paths
INDIRECT = {
    "H19": [["PE", "PB", "NV", "BI"]],
    "H20": [["SI", "PB", "NV", "BI"]],
    "H21": [["EE", "PS", "NV", "BI"]],
    "H22": [["FC", "PS", "NV", "BI"]],
    "H23": [["PV", "PS", "NV", "BI"]],
    "H24": [["HM", "PB", "NV", "BI"], ["HM", "NV", "BI"]],
    "H25": [["HT", "NV", "BI"]],
}
EXOG = [c for c in CONSTRUCTS if c not in PATHS]


def item_map(columns):
    """{construct: [item columns]} from names such as PE1, PE2, ..."""
    m = {c: [] for c in CONSTRUCTS}
    for col in columns:
        g = re.fullmatch(r"([A-Z]{2})(\d)", str(col))
        if g and g.group(1) in m:
            m[g.group(1)].append(col)
    return {k: sorted(v) for k, v in m.items()}


# --------------------------------------------------------------------------- PLS core
def zscore(a):
    return (a - a.mean(0)) / a.std(0, ddof=1)


def ols_beta(Y, X):
    """standardized OLS (inputs already standardized)"""
    return np.linalg.lstsq(X, Y, rcond=None)[0]


def pls_fit(R, blocks, tol=1e-9, maxit=500):
    """
    PLS-SEM estimation in correlation space (identical to the case-wise algorithm for
    standardized indicators, but much faster). R: indicator correlation matrix (p x p);
    blocks: {construct: [column idx]}. Outer Mode A, inner path-weighting scheme (Lohmoeller).
    """
    cs = list(blocks); m = len(cs); p = R.shape[0]; ix = {c: i for i, c in enumerate(cs)}
    owner = np.empty(p, dtype=int)
    for c in cs:
        owner[blocks[c]] = ix[c]
    E_succ = [[ix[k] for k, pr in PATHS.items() if c in pr] for c in cs]
    W = np.zeros((p, m))                                   # block-diagonal outer-weight matrix

    def normalise(W):
        v = np.einsum("ij,ij->j", W, R @ W)                # LV variances
        return W / np.sqrt(v)

    for c in cs:
        W[blocks[c], ix[c]] = 1.0
    W = normalise(W)
    for it in range(maxit):
        RW = R @ W
        C = W.T @ RW
        E = np.zeros((m, m))
        for c in cs:
            i = ix[c]
            if c in PATHS:
                Pi = [ix[q] for q in PATHS[c]]
                E[i, Pi] = np.linalg.solve(C[np.ix_(Pi, Pi)], C[Pi, i])
            for j in E_succ[i]:
                E[i, j] = C[i, j]
        cov_xp = RW @ E.T                                  # cov(x, inner proxy of each construct)
        scale = np.sqrt(np.einsum("ij,jk,ik->i", E, C, E))
        Wn = np.zeros_like(W)
        for c in cs:
            i = ix[c]; Wn[blocks[c], i] = cov_xp[blocks[c], i] / scale[i]
        Wn = normalise(Wn)
        delta = np.abs(Wn - W).max(); W = Wn
        if delta < tol:
            break
    load_all = R @ W
    for c in cs:                                           # orient: majority of loadings > 0
        if load_all[blocks[c], ix[c]].sum() < 0:
            W[:, ix[c]] *= -1
    load_all = R @ W
    C = W.T @ load_all
    coef = {}
    for k, P in PATHS.items():
        Pi = [ix[q] for q in P]
        b = np.linalg.solve(C[np.ix_(Pi, Pi)], C[Pi, ix[k]])
        for q, bq in zip(P, b):
            coef[(q, k)] = bq
    w = {c: W[blocks[c], ix[c]].copy() for c in cs}
    load = {c: load_all[blocks[c], ix[c]].copy() for c in cs}
    return dict(w=w, load=load, C=C, coef=coef, iters=it + 1)


def indirect_effects(coef):
    out = {}
    for h, plist in INDIRECT.items():
        parts = []
        for path in plist:
            parts.append(np.prod([coef[(a, b)] for a, b in zip(path[:-1], path[1:])]))
        out[h] = parts
    return out


def htmt_matrix(R, blocks):
    cs = list(blocks)
    H = pd.DataFrame(np.nan, index=cs, columns=cs)
    mono = {}
    for c in cs:
        idx = blocks[c]
        sub = np.abs(R[np.ix_(idx, idx)])
        mono[c] = sub[np.triu_indices(len(idx), 1)].mean()
    for i, a in enumerate(cs):
        for b in cs[i + 1:]:
            hh = np.abs(R[np.ix_(blocks[a], blocks[b])]).mean()
            H.loc[a, b] = H.loc[b, a] = hh / np.sqrt(mono[a] * mono[b])
    return H


def reliability(Xs, blocks, fit):
    R = np.corrcoef(Xs, rowvar=False)
    rows = []
    for c in blocks:
        idx = blocks[c]; k = len(idx)
        S = R[np.ix_(idx, idx)]
        rbar = S[np.triu_indices(k, 1)].mean()
        alpha = k * rbar / (1 + (k - 1) * rbar)
        lam = fit["load"][c]
        cr = lam.sum() ** 2 / (lam.sum() ** 2 + (1 - lam ** 2).sum())
        ave = (lam ** 2).mean()
        w = fit["w"][c]
        off = lambda M: M - np.diag(np.diag(M))
        denom = w @ off(np.outer(w, w)) @ w
        rho_a = (w @ w) ** 2 * (w @ off(S) @ w) / denom if abs(denom) > 1e-12 else np.nan
        rows.append([c, k, alpha, rho_a, cr, ave])
    return pd.DataFrame(rows, columns=["Construct", "Items", "Alpha", "rho_A", "CR", "AVE"]).set_index("Construct")


def r2_f2(C):
    """R2 and f2 from the latent-variable correlation matrix C (CONSTRUCTS order)."""
    ix = {c: i for i, c in enumerate(CONSTRUCTS)}
    def r2_of(k, P):
        if not P:
            return 0.0
        Pi = [ix[p] for p in P]; c = C[Pi, ix[k]]
        return float(c @ np.linalg.solve(C[np.ix_(Pi, Pi)], c))
    r2, f2 = {}, {}
    for k, P in PATHS.items():
        full = r2_of(k, P); r2[k] = full
        for p in P:
            f2[(p, k)] = (full - r2_of(k, [q for q in P if q != p])) / (1 - full)
    return r2, f2


def vif_inner(C):
    ix = {c: i for i, c in enumerate(CONSTRUCTS)}
    out = {}
    for k, P in PATHS.items():
        Pi = [ix[p] for p in P]
        d = np.diag(np.linalg.inv(C[np.ix_(Pi, Pi)]))
        for p, v in zip(P, d):
            out[(p, k)] = v
    return out


# --------------------------------------------------------------------------- bootstrap
def run_bootstrap(X_raw, blocks, B, seed, ref_fit):
    """Case resampling; each resample re-estimates the whole model. Signs are aligned to the original weights."""
    rng = np.random.default_rng(seed)
    n = X_raw.shape[0]
    paths = list(ref_fit["coef"])
    store = {"coef": [], "ind": [], "load": {c: [] for c in blocks}, "r2": []}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        Xb = X_raw[idx]
        if np.any(Xb.std(0) == 0):
            continue
        f = pls_fit(np.corrcoef(Xb, rowvar=False), blocks)
        flip = {c: (np.sign(f["w"][c] @ ref_fit["w"][c]) or 1.0) for c in blocks}
        coef = {(a, b): f["coef"][(a, b)] * flip[a] * flip[b] for (a, b) in paths}
        store["coef"].append([coef[p] for p in paths])
        ind = indirect_effects(coef)
        store["ind"].append([sum(v) for v in ind.values()] + [x for v in ind.values() for x in v])
        for c in blocks:
            store["load"][c].append(f["load"][c] * flip[c])
        store["r2"].append(list(r2_f2(f["C"])[0].values()))
    return paths, store


def boot_table(est, boots, labels, n):
    boots = np.asarray(boots)
    se = boots.std(0, ddof=1)
    t = est / se
    p = 2 * stats.t.sf(np.abs(t), df=n - 1)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return pd.DataFrame({"Estimate": est, "SE": se, "t": t, "p": p, "CI2.5": lo, "CI97.5": hi}, index=labels)


# --------------------------------------------------------------------------- group analyses
def anova_tukey(df, scores, group="Wilayah"):
    rows, post = [], []
    for c in scores.columns:
        groups = [scores.loc[df[group] == g, c].values for g in sorted(df[group].unique())]
        F, p = stats.f_oneway(*groups)
        lev = stats.levene(*groups).pvalue
        rows.append([c, F, p, lev] + [g.mean() for g in groups])
        tk = stats.tukey_hsd(*groups)
        names = sorted(df[group].unique())
        k = len(groups)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                # Games-Howell (robust to unequal variances)
                v1, v2 = groups[i].var(ddof=1) / len(groups[i]), groups[j].var(ddof=1) / len(groups[j])
                dfw = (v1 + v2) ** 2 / (v1 ** 2 / (len(groups[i]) - 1) + v2 ** 2 / (len(groups[j]) - 1))
                q = abs(groups[i].mean() - groups[j].mean()) / np.sqrt((v1 + v2) / 2)
                gh = stats.studentized_range.sf(q, k, dfw)
                post.append([c, names[i], names[j], groups[i].mean() - groups[j].mean(), tk.pvalue[i, j], gh])
    cols = ["Construct", "F", "p", "Levene_p"] + [f"M_{g}" for g in sorted(df[group].unique())]
    return (pd.DataFrame(rows, columns=cols).set_index("Construct"),
            pd.DataFrame(post, columns=["Construct", "Group_A", "Group_B", "Mean_diff", "Tukey_p", "GamesHowell_p"]))


def mga_keil(X_raw, blocks, mask, B, seed, ref_paths):
    res = {}
    for name, m in (("A", mask), ("B", ~mask)):
        Xg = X_raw[m]; f = pls_fit(np.corrcoef(Xg, rowvar=False), blocks)
        p, st = run_bootstrap(Xg, blocks, B, seed, f)
        res[name] = (f["coef"], np.asarray(st["coef"]).std(0, ddof=1), m.sum())
    rows = []
    n1, n2 = res["A"][2], res["B"][2]
    for i, pth in enumerate(ref_paths):
        b1, b2 = res["A"][0][pth], res["B"][0][pth]; s1, s2 = res["A"][1][i], res["B"][1][i]
        sp = np.sqrt(((n1 - 1) ** 2 / (n1 + n2 - 2)) * s1 ** 2 + ((n2 - 1) ** 2 / (n1 + n2 - 2)) * s2 ** 2) * np.sqrt(1 / n1 + 1 / n2)
        t = (b1 - b2) / sp
        rows.append([HYP[pth], f"{pth[0]}->{pth[1]}", b1, b2, b1 - b2, t, 2 * stats.t.sf(abs(t), n1 + n2 - 2)])
    return pd.DataFrame(rows, columns=["H", "Path", "Beta_A", "Beta_B", "Diff", "t", "p"]), (n1, n2)


# --------------------------------------------------------------------------- main analysis
def analyse(df, B=5000, seed=42, out="results", boot_group=2000, verbose=True):
    imap = item_map(df.columns)
    cols = [c for k in CONSTRUCTS for c in imap[k]]
    X_raw = df[cols].astype(float).values
    blocks, pos = {}, 0
    for k in CONSTRUCTS:
        blocks[k] = list(range(pos, pos + len(imap[k]))); pos += len(imap[k])
    n = len(df)
    Xs = zscore(X_raw)
    R = np.corrcoef(Xs, rowvar=False)
    fit = pls_fit(R, blocks)
    fit["lv"] = {c: Xs[:, blocks[c]] @ fit["w"][c] for c in CONSTRUCTS}
    T = {}
    # --- measurement model
    T["Reliability_AVE"] = reliability(Xs, blocks, fit)
    T["Loadings"] = pd.DataFrame({"Construct": [k for k in CONSTRUCTS for _ in blocks[k]], "Item": cols,
                                  "Loading": np.concatenate([fit["load"][k] for k in CONSTRUCTS])}).set_index("Item")
    scores = pd.DataFrame(fit["lv"])[CONSTRUCTS]
    cl = pd.DataFrame({k: [np.corrcoef(Xs[:, j], fit["lv"][k])[0, 1] for j in range(Xs.shape[1])] for k in CONSTRUCTS}, index=cols)
    T["Cross_loadings"] = cl
    LVc = scores.corr()
    fl = LVc.copy()
    for k in CONSTRUCTS:
        fl.loc[k, k] = np.sqrt(T["Reliability_AVE"].loc[k, "AVE"])
    T["Fornell_Larcker"] = fl
    T["HTMT"] = htmt_matrix(R, blocks)
    # --- structural model
    r2, f2 = r2_f2(fit["C"])
    T["R2"] = pd.DataFrame({"R2": r2, "R2_adj": {k: 1 - (1 - r2[k]) * (n - 1) / (n - len(PATHS[k]) - 1) for k in PATHS}})
    T["VIF_inner"] = pd.Series(vif_inner(fit["C"])).rename("VIF").to_frame()
    T["VIF_inner"].index = [f"{a}->{b}" for a, b in T["VIF_inner"].index]
    # --- bootstrap
    paths, st = run_bootstrap(X_raw, blocks, B, seed, fit)
    est = np.array([fit["coef"][p] for p in paths])
    pt = boot_table(est, st["coef"], [f"{HYP[p]} {p[0]}->{p[1]}" for p in paths], n)
    pt["f2"] = [f2[p] for p in paths]
    pt["Supported(p<.05)"] = np.where(pt["p"] < .05, "Yes", "No")
    T["Paths"] = pt
    ind = indirect_effects(fit["coef"])
    est_i = [sum(v) for v in ind.values()] + [x for v in ind.values() for x in v]
    labs = [f"{h} total" for h in ind] + [f"{h}{'abc'[i] if len(v) > 1 else ''} " + "->".join(INDIRECT[h][i])
                                           for h, v in ind.items() for i in range(len(v))]
    it = boot_table(np.array(est_i), st["ind"], labs, n)
    # mediation classification (Nitzl et al., 2016): direct effect per antecedent
    cls = []
    for lab in it.index:
        h = lab.split()[0]
        if lab.endswith("total"):
            ant = INDIRECT[h][0][0]
            d = pt.loc[[i for i in pt.index if i.endswith(f"{ant}->BI")][0]]
            sig_ind = it.loc[lab, "CI2.5"] * it.loc[lab, "CI97.5"] > 0
            sig_dir = d["p"] < .05
            same = np.sign(d["Estimate"]) == np.sign(it.loc[lab, "Estimate"])
            cls.append("none" if not sig_ind else ("indirect-only (full)" if not sig_dir else
                       ("complementary partial" if same else "competitive partial")))
        else:
            cls.append("")
    it["Mediation"] = cls
    T["Indirect"] = it
    # --- diagnostics
    ev = np.linalg.eigvalsh(R)[::-1]
    T["Harman"] = pd.DataFrame({"First_factor_variance_%": [100 * ev[0] / R.shape[0]]})
    fc = {}
    for k in CONSTRUCTS:
        others = [o for o in CONSTRUCTS if o != k]
        b = ols_beta(scores[k].values, scores[others].values)
        r = 1 - (scores[k].values - scores[others].values @ b).var(ddof=1) / scores[k].values.var(ddof=1)
        fc[k] = 1 / (1 - r)
    T["Full_collinearity_VIF"] = pd.Series(fc).rename("VIF").to_frame()
    T["Item_descriptives"] = pd.DataFrame({"Mean": df[cols].mean(), "SD": df[cols].std(),
                                           "Skew": df[cols].skew(), "Kurt": df[cols].kurt()})
    T["Construct_descriptives"] = pd.DataFrame({"Mean": scores.mean(), "SD": scores.std()})  # z-scores; see raw below
    raw = pd.DataFrame({k: df[imap[k]].mean(axis=1) for k in CONSTRUCTS})
    T["Construct_descriptives"] = pd.DataFrame({"Mean": raw.mean(), "SD": raw.std(), "Skew": raw.skew(), "Kurt": raw.kurt()})
    # --- group analyses
    if "Wilayah" in df:
        an, post = anova_tukey(df, raw)
        T["ANOVA_region"], T["Tukey_region"] = an, post
    if "Jumlah_Ternak_ekor" in df:
        T["Herd_corr"] = pd.DataFrame([[k, *stats.pearsonr(df["Jumlah_Ternak_ekor"], raw[k])] for k in CONSTRUCTS],
                                      columns=["Construct", "r", "p"]).set_index("Construct")
        mask = (df["Jumlah_Ternak_ekor"] <= 7).values
        mg, (n1, n2) = mga_keil(X_raw, blocks, mask, boot_group, seed, paths)
        T["MGA_herd_Keil"] = mg
        T["MGA_herd_Keil"].attrs["n"] = (n1, n2)
    T["Run_info"] = pd.DataFrame({"n": [n], "bootstrap": [B], "seed": [seed], "PLS_iterations": [fit["iters"]],
                                  "bootstrap_group": [boot_group]})
    if out:
        import os
        os.makedirs(out, exist_ok=True)
        with pd.ExcelWriter(f"{out}/pls_sem_results.xlsx") as xw:
            for k, v in T.items():
                v.to_excel(xw, sheet_name=k[:31])
        scores.to_csv(f"{out}/latent_scores.csv", index=False)
    if verbose:
        pd.set_option("display.width", 200, "display.float_format", lambda x: f"{x:.3f}")
        print(f"n = {n}, bootstrap = {B}, seed = {seed}\n")
        print("RELIABILITY / AVE\n", T["Reliability_AVE"], "\n")
        print("R2\n", T["R2"], "\n")
        print("PATHS\n", T["Paths"], "\n")
        print("INDIRECT\n", T["Indirect"], "\n")
        print("Max HTMT:", np.nanmax(T["HTMT"].values).round(3), "| Harman first factor %:",
              T["Harman"].iloc[0, 0].round(1), "\n")
        if "ANOVA_region" in T:
            print("ANOVA\n", T["ANOVA_region"], "\n")
    return T



def validate_items(df, drop_invalid=False):
    """Likert items must be integers 1-5. Report offending cells; optionally drop those respondents (listwise)."""
    imap = item_map(df.columns)
    cols = [c for k in CONSTRUCTS for c in imap[k]]
    vals = df[cols]
    bad = ~vals.isin([1, 2, 3, 4, 5])
    if bad.values.any():
        cells = [(df.iloc[i, 0], cols[j]) for i, j in zip(*np.where(bad.values))]
        print(f"WARNING: {len(cells)} invalid Likert cell(s) (not 1-5), respondents/items: {cells}")
        if drop_invalid:
            keep = ~bad.any(axis=1)
            print(f"  -> dropping {int((~keep).sum())} respondent(s) listwise; n = {int(keep.sum())}")
            return df[keep].reset_index(drop=True)
        print("  -> kept as is. Fix the data, or rerun with --drop-invalid.")
    return df

# --------------------------------------------------------------------------- self-test
def simulate(n=3000, seed=1):
    """Generate data from the model with known paths/loadings to verify parameter recovery."""
    rng = np.random.default_rng(seed)
    true = {("PE", "PB"): .30, ("SI", "PB"): .25, ("HM", "PB"): .33, ("EE", "PS"): -.25, ("FC", "PS"): -.20,
            ("PV", "PS"): -.15, ("PB", "NV"): .40, ("PS", "NV"): -.20, ("HM", "NV"): .10, ("HT", "NV"): .15,
            ("NV", "BI"): .32, ("PE", "BI"): .13, ("EE", "BI"): .07, ("SI", "BI"): .09, ("FC", "BI"): .05,
            ("HM", "BI"): .15, ("PV", "BI"): .08, ("HT", "BI"): .12}
    L = {c: rng.standard_normal(n) for c in EXOG}
    for k in ["PB", "PS", "NV", "BI"]:
        v = sum(true[(p, k)] * L[p] for p in PATHS[k])
        L[k] = v + rng.standard_normal(n) * np.sqrt(max(1 - v.var(), .05))
        L[k] = (L[k] - L[k].mean()) / L[k].std() if k != "x" else L[k]
    nit = {"PE": 4, "EE": 4, "SI": 3, "FC": 4, "HM": 3, "PV": 3, "HT": 4, "PB": 4, "PS": 4, "NV": 3, "BI": 4}
    cols = {}
    for c in CONSTRUCTS:
        for i in range(nit[c]):
            lam = rng.uniform(.78, .88)
            x = lam * L[c] + np.sqrt(1 - lam ** 2) * rng.standard_normal(n)
            cols[f"{c}{i+1}"] = np.clip(np.round(3 + 0.9 * x), 1, 5)
    return pd.DataFrame(cols), true


def selftest():
    df, true = simulate()
    T = analyse(df, B=200, seed=7, out=None, verbose=False)
    pt = T["Paths"]
    comp = pd.DataFrame({"true": [true[p] for p in true],
                         "estimated": [pt.loc[[i for i in pt.index if i.endswith(f"{a}->{b}")][0], "Estimate"] for (a, b) in true]},
                        index=[f"{a}->{b}" for a, b in true])
    comp["abs_err"] = (comp.estimated - comp.true).abs()
    print(comp.round(3)); print("\nmean abs error:", comp.abs_err.mean().round(3))
    print(T["Reliability_AVE"].round(3).head(3))
    print("(estimates are slightly attenuated vs. latent truth, as expected for composite-based PLS-SEM)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?")
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--boot-group", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    ap.add_argument("--sheet", default=0, help="Excel sheet name or index with the raw data")
    ap.add_argument("--drop-invalid", action="store_true", help="drop respondents with Likert values outside 1-5")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); sys.exit()
    if not a.data:
        ap.error("data file required")
    d = pd.read_excel(a.data, sheet_name=(int(a.sheet) if str(a.sheet).isdigit() else a.sheet)) if a.data.endswith(("xlsx", "xls")) else pd.read_csv(a.data)
    d = validate_items(d, a.drop_invalid)
    analyse(d, B=a.boot, seed=a.seed, out=a.out, boot_group=a.boot_group)
