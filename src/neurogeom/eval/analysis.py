"""Preregistered analysis: summaries with CIs and hypothesis tests H1-H5 from the per-sample tables.

Global context: reads results/<run>/synth/per_sample.csv (one row per sample x system) and
results/<run>/real/real_results.json, and writes results/<run>/analysis.json. Every number in README,
reports and paper comes from that file via report/numbers.py (no hand-typed numbers).
Every synthetic CI is a cluster bootstrap over generator chunks (clouds of one chunk share N and the R^D-embedding
draw; audit 2026-11-14), and Wilcoxon tests use per-chunk mean differences. Post-hoc checks (not preregistered) are
kept apart under "posthoc" and the "*_posthoc" keys; those of the 2026-11-22 audit (stratified cluster bootstrap,
refinement by coarse class, Chamfer discretisation floor) are under "posthoc" -> "design_checks".
Comparisons with baselines are paired on the baseline subset (identical samples); comparisons with
ablations that are separate models (A1, A2, A4, A6) are paired on the fixed method subset that
scripts/sweep.sh evaluates (eval.method_subset_per_cell = 100 samples per family and cell, same samples
and order from the fixed `test` stream).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neurogeom.data.testsets import CHUNK
from neurogeom.eval.stats import (binom_ci, bootstrap_ci, bootstrap_ci_pairs, ece, fit_temperature, holm, macro_f1, paired_bootstrap,
                                  paired_bootstrap_f1, wilcoxon_p)
from neurogeom.logging_utils import get_logger
from neurogeom.realdata.common import is_prereg
from neurogeom.shapes.registry import INDEX

log = get_logger(__name__)


def cluster_ids(sample_ids: pd.Series, chunk: int) -> pd.Series:
    """Cluster of dependent test clouds = the generator chunk a cloud came from (its clouds share N and the R^D-embedding
    draw, data/testsets.py). sample_id is '<n_bin>-<noise_bin>-<index in cell>'; each family fills a consecutive block of
    test_per_cell clouds generated in chunks of `chunk`, so index // chunk identifies the chunk (audit 2026-11-14)."""
    parts = sample_ids.str.split("-", expand=True)
    return parts[0] + "-" + parts[1] + "-" + (parts[2].astype(int) // chunk).astype(str)


def _code(names: pd.Series) -> np.ndarray:
    """Family names -> integer indices, so the bootstrapped macro-F1 sorts integers instead of strings."""
    return names.map(INDEX).to_numpy()


def system_summary(df: pd.DataFrame, reps: int) -> dict:
    out = {}
    for sysname, g in df.groupby("system"):
        cl = g.cluster.values                                                      # cluster bootstrap over generator chunks
        acc = bootstrap_ci(g.correct.astype(float), reps=reps, groups=cl)
        f1 = bootstrap_ci_pairs(_code(g.family_true), _code(g.family_pred), macro_f1, reps=reps, groups=cl)   # same replicate count as every CI (prereg Sec. 5.2)
        cd = bootstrap_ci(g.chamfer.values, stat=np.median, reps=reps, groups=cl)
        out[sysname] = {
            "n": int(len(g)), "fine_acc": acc, "macro_f1": f1, "coarse_acc": bootstrap_ci(g.coarse_correct.astype(float), reps=reps, groups=cl),
            "superset_acc": bootstrap_ci(g.superset_ok.astype(float), reps=reps, groups=cl), "median_chamfer": cd,
            "mean_chamfer": float(g.chamfer.mean()), "mean_residual": float(g.residual.mean()),
            "median_axis_err_deg": float(np.nanmedian(g.axis_err_deg)) if g.axis_err_deg.notna().any() else None,
            "median_logscale_err": float(np.nanmedian(g.logscale_err)) if g.logscale_err.notna().any() else None,
        }
    return out


def breakdown(df: pd.DataFrame, key: str, reps: int) -> dict:
    """Fine accuracy (with a cluster-bootstrap 95% CI, for the figure error bars) and median Chamfer per system and bin."""
    return {s: {int(k): {"fine_acc": float(v.correct.mean()), "median_chamfer": float(v.chamfer.median()), "n": int(len(v)),
                         "fine_acc_ci": list(bootstrap_ci(v.correct.astype(float).values, reps=reps, groups=v.cluster.values)[1:])}
                for k, v in g.groupby(key)} for s, g in df.groupby("system")}


def per_family(df: pd.DataFrame) -> dict:
    return {s: {f: {"recall": float(v.correct.mean()), "median_chamfer": float(v.chamfer.median())} for f, v in g.groupby("family_true")}
            for s, g in df.groupby("system")}


def confusion(df: pd.DataFrame, system: str) -> dict:
    g = df[df.system == system]
    return pd.crosstab(g.family_true, g.family_pred).to_dict()


def paired(df: pd.DataFrame, a: str, b: str, reps: int) -> dict | None:
    ga, gb = df[df.system == a].set_index("sample_id"), df[df.system == b].set_index("sample_id")
    common = ga.index.intersection(gb.index)
    if len(common) == 0:
        return None
    ga, gb = ga.loc[common], gb.loc[common]
    cl = ga.cluster.values                                                         # resample whole generator chunks
    f1 = paired_bootstrap_f1(_code(ga.family_true), _code(ga.family_pred), _code(gb.family_pred), reps=reps, groups=cl)
    cd = paired_bootstrap(ga.chamfer.values, gb.chamfer.values, stat=np.median, reps=reps, groups=cl)
    return {"n": int(len(common)), "n_clusters": int(len(set(cl))), "macro_f1_diff": f1, "median_chamfer_diff": cd,
            "fine_acc_diff": paired_bootstrap(ga.correct.astype(float).values, gb.correct.astype(float).values, reps=reps, groups=cl),
            "wilcoxon_p_chamfer": wilcoxon_p(ga.chamfer.values, gb.chamfer.values, groups=cl)}


def _f1_pvalue(lo: float, hi: float, est: float) -> float:
    """Two-sided bootstrap p-value approximation from the CI (normal approximation of the bootstrap SE)."""
    from scipy.stats import norm
    se = max((hi - lo) / (2 * 1.96), 1e-12)
    return float(2 * norm.sf(abs(est) / se))   # survival function: no 1 - cdf cancellation in the tail


def hypotheses_synth(df: pd.DataFrame, reps: int, ablation_frames: dict) -> dict:
    h = {}
    c1 = paired(df, "method", "B1_classical", reps)
    if c1:
        est, lo, hi = c1["macro_f1_diff"]
        h["H1"] = {"comparison": "macro-F1(method) - macro-F1(B1)", "estimate": est, "ci": [lo, hi], "supported": lo > 0,
                   "p": _f1_pvalue(lo, hi, est), "n": c1["n"]}
        est, lo, hi = c1["median_chamfer_diff"]
        h["H2"] = {"comparison": "median Chamfer(method) - median Chamfer(B1)", "estimate": est, "ci": [lo, hi],
                   "falsified": lo > 0, "supported": lo <= 0 and est <= 0, "p": c1["wilcoxon_p_chamfer"], "n": c1["n"]}
    c3 = paired(df, "method", "A3_norefine", reps)
    if c3:
        est, lo, hi = c3["median_chamfer_diff"]
        h["H5b_refinement"] = {"comparison": "median Chamfer(method) - median Chamfer(no refinement)", "estimate": est, "ci": [lo, hi],
                               "supported": hi < 0, "p": c3["wilcoxon_p_chamfer"], "n": c3["n"]}
    if "A1_paramloss" in ablation_frames and "A0_reference" in ablation_frames:
        # matched budget: the unchanged method retrained with the ablation budget (A0) vs A1
        both = pd.concat([ablation_frames["A0_reference"], ablation_frames["A1_paramloss"]])
        c = paired(both, "A0_reference", "A1_paramloss", reps)
        est, lo, hi = c["median_chamfer_diff"]
        h["H5a_chamfer_loss"] = {"comparison": "median Chamfer(A0 method, short budget) - median Chamfer(A1 parameter-space loss)", "estimate": est,
                                 "ci": [lo, hi], "supported": hi < 0, "p": c["wilcoxon_p_chamfer"], "n": c["n"]}
    adj = holm({k: v["p"] for k, v in h.items() if v.get("p") is not None})
    for k, p in adj.items():
        h[k]["p_holm"] = p
    return h


def posthoc_synth(df: pd.DataFrame, reps: int) -> dict:
    """Post-hoc checks added by the 2026-11-14 audit (not preregistered, outside the Holm family):
    - blob recall and blob predictions per system: geometric AIC with the noise level estimated from the best
      non-volumetric candidate can never select the d = 3 blob (B1, A5_gaic; reports/deviations.md);
    - the method-vs-B1 comparison restricted to clouds whose true family is not blob;
    - the comparison with the corrected classical baseline B1-fixed, if it has been run."""
    out = {"blob_recall": {s: float(g[g.family_true == "blob"].correct.mean()) for s, g in df.groupby("system")},
           "blob_predicted": {s: int((g.family_pred == "blob").sum()) for s, g in df.groupby("system")}}
    nb = df[df.family_true != "blob"]
    out["nonblob"] = {"accuracy": {s: float(g.correct.mean()) for s, g in nb.groupby("system")},
                      "macro_f1": {s: macro_f1(g.family_true, g.family_pred) for s, g in nb.groupby("system")},
                      "method_vs_B1": paired(nb, "method", "B1_classical", reps)}
    if (df.system == "B1_fixed").any():
        out["method_vs_B1_fixed"] = paired(df, "method", "B1_fixed", reps)
        out["B1_fixed_vs_B1"] = paired(df, "B1_fixed", "B1_classical", reps)
    return out


def posthoc_design_checks(df: pd.DataFrame, reps: int) -> dict:
    """Post-hoc checks added by the 2026-11-22 audit (not preregistered, outside the Holm family):
    - stratified cluster bootstrap of the full-test-set statistics: chunks are resampled within their family x N-bin x
      noise-bin stratum (10 chunks each), as the test design fixes them. The reported cluster bootstrap resamples chunks
      across strata, so its CIs also carry the spread between stratum means and are conservative (eval/stats.py,
      _sampler). In the baseline subset every stratum is a single chunk, so no stratified CI exists there.
    - refinement (H5b) split by the coarse class of the true family (each chunk holds one family, so every class is a
      set of whole chunks and the cluster bootstrap applies unchanged);
    - the discretisation floor of the evaluation Chamfer distance (eval/metrics.py, chamfer_floor)."""
    from neurogeom.eval.metrics import chamfer_floor
    from neurogeom.shapes.registry import COARSE, COARSE_NAMES
    out = {"chamfer_floor": chamfer_floor()}
    if not {"method", "A3_norefine"} <= set(df.system):
        return out
    a = df[df.system == "method"].set_index("sample_id")
    b = df[df.system == "A3_norefine"].set_index("sample_id").loc[a.index]
    strata = (a.n_bin.astype(str) + "-" + a.noise_bin.astype(str) + "-" + a.family_true).values
    out["stratified_ci"] = {
        "method_fine_acc": bootstrap_ci(a.correct.astype(float).values, reps=reps, groups=a.cluster.values, strata=strata),
        "method_median_chamfer": bootstrap_ci(a.chamfer.values, stat=np.median, reps=reps, groups=a.cluster.values, strata=strata),
        "H5b_median_chamfer_diff": paired_bootstrap(a.chamfer.values, b.chamfer.values, stat=np.median, reps=reps,
                                                    groups=a.cluster.values, strata=strata),
        "n_strata": int(len(set(strata))), "n_clusters": int(a.cluster.nunique())}
    coarse = a.family_true.map(lambda f: COARSE_NAMES[COARSE[INDEX[f]]]).values
    out["refinement_by_coarse"] = {c: {"n": int((coarse == c).sum()),
                                       "median_chamfer_method": float(np.median(a.chamfer.values[coarse == c])),
                                       "median_chamfer_A3": float(np.median(b.chamfer.values[coarse == c])),
                                       "median_chamfer_diff": paired_bootstrap(a.chamfer.values[coarse == c], b.chamfer.values[coarse == c],
                                                                               stat=np.median, reps=reps, groups=a.cluster.values[coarse == c])}
                                   for c in COARSE_NAMES if (coarse == c).any()}
    return out


def calibration(res_dir: Path) -> dict | None:
    """ECE (15 bins) and NLL of the classifier on the synthetic test set before (T = 1) and after temperature scaling
    (preregistration Sec. 5.4). probs.npz holds p = softmax(l / T), so l = T log p up to a per-sample constant.
    `t_test_optimal` is the temperature that would minimise NLL on the test set itself (a check of how well the
    validation-fitted T transfers; it is never used for selection)."""
    pz, mz = res_dir / "synth" / "probs.npz", res_dir / "synth" / "meta.json"
    if not (pz.exists() and mz.exists()):
        return None
    z, t = np.load(pz), json.load(open(mz))["temperature"]
    logits, y = t * np.log(np.clip(z["probs"], 1e-12, 1.0)), z["labels"]

    def scored(temp):
        lg = logits / temp
        q = np.exp(lg - lg.max(1, keepdims=True))
        q /= q.sum(1, keepdims=True)
        return {"ece": ece(q, y), "nll": float(-np.log(q[np.arange(len(y)), y] + 1e-12).mean())}
    t_test = fit_temperature(torch.as_tensor(logits), torch.as_tensor(y, dtype=torch.long))   # same LBFGS fitter as on `val`
    return {"temperature": t, "t_test_optimal": t_test, "n": int(len(y)), "before": scored(1.0), "after": scored(t)}


def hypotheses_real(results: list[dict], seed: int = 0) -> dict:
    pre = [r for r in results if r["id"].startswith("R") and is_prereg(r) and r["meta"].get("expected_coarse")]
    lab = lambda r: r.get("stability", {}).get("coarse_majority", r["coarse"])
    agree = [lab(r) == r["meta"]["expected_coarse"] for r in pre]
    k, n = int(sum(agree)), len(agree)
    rng = np.random.default_rng(seed)
    preds = np.array([lab(r) for r in pre])
    exp = np.array([r["meta"]["expected_coarse"] for r in pre])
    null = [int((rng.permutation(preds) == exp).sum()) for _ in range(10000)]
    p_perm = float((np.array(null) >= k).mean())
    fine_ok = lambda fam, r: fam in (r["meta"].get("expected_fine") or [])
    h3 = {"agree": k, "n": n, "rate": k / max(n, 1), "ci": binom_ci(k, n) if n else None, "p_permutation": p_perm,
          # post hoc (not preregistered): the same count with the full-cloud label instead of the subsample majority
          "agree_full_cloud_posthoc": int(sum(r["coarse"] == r["meta"]["expected_coarse"] for r in pre)),
          # post hoc (audit 2026-11-14): trivial baseline = always predict the most frequent expected coarse label,
          # and agreement at the level of the exact family (subsample majority / full cloud)
          "constant_majority_agree_posthoc": int(max(np.unique(exp, return_counts=True)[1])) if n else 0,
          "fine_agree_subsample_posthoc": int(sum(fine_ok(r.get("stability", {}).get("fine_majority", r["family"]), r) for r in pre)),
          "fine_agree_full_cloud_posthoc": int(sum(fine_ok(r["family"], r) for r in pre)),
          "supported": n > 0 and k / n >= 0.5 and p_perm < 0.05,
          "per_dataset": {r["id"]: {"predicted_coarse": lab(r), "expected_coarse": r["meta"]["expected_coarse"],
                                    "predicted_fine": r.get("stability", {}).get("fine_majority", r["family"]),
                                    "expected_fine": r["meta"].get("expected_fine")} for r in pre}}
    c2 = [r["C2_matched_gaussian"].get("blob", 0.0) for r in results if "C2_matched_gaussian" in r and r["id"].startswith("R")]
    c1 = [r for r in results if r["id"].startswith("C1_")]
    ring = float(np.mean([r["family"] == "ringing" for r in c1])) if c1 else None
    helix = float(np.mean([r["family"] == "helix" for r in c1])) if c1 else None
    blob_rate = float(np.mean(c2)) if c2 else None
    # post-hoc breakdown (not preregistered): smooth Gaussian processes and random walks separately
    by_proc = {k: {fam: float(np.mean([r["family"] == fam for r in c1 if r["id"].startswith(f"C1_{k}_")]))
                   for fam in ("helix", "ringing")} for k in ("gp", "walk") if any(r["id"].startswith(f"C1_{k}_") for r in c1)}
    # post hoc (audit 2026-11-14): the preregistered criterion read per dataset ("blob in >= 80%"), over preregistered datasets only
    c2_pre = [r["C2_matched_gaussian"].get("blob", 0.0) for r in results if "C2_matched_gaussian" in r and r["id"].startswith("R") and is_prereg(r)]
    h4 = {"C2_blob_rate_mean": blob_rate, "C2_per_dataset": {r["id"]: r["C2_matched_gaussian"] for r in results if "C2_matched_gaussian" in r},
          "C2_blob_rate_mean_prereg_posthoc": float(np.mean(c2_pre)) if c2_pre else None,
          "C2_prereg_at_least_80_posthoc": int(sum(v >= 0.8 for v in c2_pre)), "C2_prereg_n_posthoc": len(c2_pre),
          "C1_ringing_rate": ring, "C1_helix_rate": helix, "C1_labels": {r["id"]: r["family"] for r in c1}, "C1_by_process_posthoc": by_proc,
          "supported": blob_rate is not None and blob_rate >= 0.8 and ring is not None and ring > helix}
    c3 = {r["id"]: r["family"] for r in results if r["id"].startswith("C3_")}
    # preregistered two-part helix rule (p(helix)/p(ringing) >= 3 and lower held-out residual than ringing), recorded for
    # every full-cloud helix fit that had held-out folds: True = helix claim accepted, False = rejected
    rule = {r["id"]: bool(r["helix_claim_supported"]) for r in results if r.get("helix_claim_supported") is not None}
    non_helix = [k for k in rule if "helix" not in (next(r for r in results if r["id"] == k)["meta"].get("expected_fine") or ["helix"])]
    return {"H3": h3, "H4": h4, "C3_random_init": c3, "helix_rule": rule, "helix_rule_non_helix_expected": non_helix}


def run_analysis(cfg: dict, ablation_runs: dict | None = None) -> dict:
    res_dir = Path(cfg["paths"]["results"])
    reps = cfg["eval"]["bootstrap_reps"]
    df = pd.read_csv(res_dir / "synth" / "per_sample.csv", low_memory=False)
    assert cfg["eval"]["test_per_cell"] % CHUNK == 0, "cluster ids assume every family block is a whole number of generator chunks"
    df["cluster"] = cluster_ids(df.sample_id, CHUNK)
    # the method restricted to the baseline subset, so every row of the main table can be compared on the same clouds
    base_ids = set(df[df.system == "B1_classical"].sample_id)
    if base_ids:
        df = pd.concat([df, df[(df.system == "method") & df.sample_id.isin(base_ids)].assign(system="method_subset")])
    abl = {}
    for name, run in (ablation_runs or {}).items():
        p = res_dir.parent / run / "synth" / "per_sample.csv"
        if p.exists():
            a = pd.read_csv(p, low_memory=False)
            a = a[a.system == "method"].copy()
            a["system"] = name
            a["cluster"] = cluster_ids(a.sample_id, CHUNK)
            abl[name] = a
    full = pd.concat([df] + list(abl.values()))
    out = {"systems": system_summary(full, reps), "by_n_bin": breakdown(full, "n_bin", reps), "by_noise_bin": breakdown(full, "noise_bin", reps),
           "per_family": per_family(full), "confusion_method": confusion(df, "method"),
           "hypotheses": hypotheses_synth(df, reps, abl), "posthoc": posthoc_synth(df, reps), "calibration": calibration(res_dir),
           "ci_method": f"cluster bootstrap over generator chunks of {CHUNK} clouds (audit 2026-11-14)"}
    out["posthoc"]["design_checks"] = posthoc_design_checks(df, reps)   # audit 2026-11-22
    tau_p = res_dir / "synth" / "b1_fixed_tau.json"
    if tau_p.exists():   # B1-fixed's blob threshold, fitted on the val stream (eval/synth_eval.py, tune_blob_tau)
        t = json.load(open(tau_p))
        out["posthoc"]["b1_fixed_tau"] = {k: t[k] for k in ("tau", "val_macro_f1", "val_macro_f1_never_blob", "n_val")}
    if "ambiguous" in df:
        ids = set(df[(df.system == "B1_classical") & (df.ambiguous == True)].sample_id)  # noqa: E712
        sub = df[(df.system != "method_subset") & ~df.sample_id.isin(ids) & df.sample_id.isin(base_ids)]
        out["systems_unambiguous_subset"] = system_summary(sub, reps)
        out["ambiguous_fraction"] = float(len(ids) / max(1, (df.system == "B1_classical").sum()))
    rp = res_dir / "real" / "real_results.json"
    if rp.exists():
        out["real"] = hypotheses_real(json.load(open(rp)))
    json.dump(out, open(res_dir / "analysis.json", "w"), indent=2, default=lambda o: o if not isinstance(o, np.generic) else o.item())
    log.info("analysis written: %s", res_dir / "analysis.json")
    return out
