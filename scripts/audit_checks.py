"""Independent audit checks of 2026-11-22: recompute headline numbers and verify text claims from the stored results.

Global context: reports/reviews/audit-2026-11-22.md cites the numbers this script writes to
results/main_s0/audit_2026-11-22.json. It reads only results files (no model, no inference) and uses library
implementations (sklearn f1_score, scipy beta) instead of the project's own metric code, so that it is an
independent check of eval/stats.py and eval/analysis.py.
Run: uv run python scripts/audit_checks.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import f1_score

RES = Path(__file__).resolve().parents[1] / "results" / "main_s0"


def recompute_synthetic(df: pd.DataFrame) -> dict:
    """Accuracy, macro-F1 (sklearn, over the true classes) and median Chamfer per system, plus design sanity checks."""
    out = {}
    for s, g in df.groupby("system"):
        labels = sorted(g.family_true.unique())
        out[s] = {"n": int(len(g)), "fine_acc": float(g.correct.mean()), "median_chamfer": float(g.chamfer.median()),
                  "macro_f1": float(f1_score(g.family_true, g.family_pred, average="macro", labels=labels, zero_division=0))}
    m = df[df.system == "method"].copy()
    p = m.sample_id.str.split("-", expand=True)
    m["chunk"] = p[0] + "-" + p[1] + "-" + (p[2].astype(int) // 25).astype(str)           # generator chunk (data/testsets.py)
    by = m.groupby("chunk")
    out["_design"] = {"duplicate_rows": int(df.duplicated(["system", "sample_id"]).sum()),
                      "n_chunks": int(m.chunk.nunique()),
                      "N_constant_within_chunk": bool((by.N.nunique() == 1).all()),
                      "embedding_constant_within_chunk": bool((by.embed_dim.nunique() == 1).all()),
                      "family_constant_within_chunk": bool((by.family_true.nunique() == 1).all()),
                      "true_family_identical_across_systems": bool(
                          (df.pivot_table(index="sample_id", columns="system", values="family_true", aggfunc="first").nunique(axis=1) <= 1).all())}
    return out


def recompute_h3(analysis: dict, perms: int = 200_000, seed: int = 1) -> dict:
    """H3 agreement, Monte-Carlo permutation p (more permutations than the preregistered 10,000) and Clopper-Pearson CI."""
    pre = analysis["real"]["H3"]["per_dataset"]
    pred = np.array([v["predicted_coarse"] for v in pre.values()])
    exp = np.array([v["expected_coarse"] for v in pre.values()])
    k, n = int((pred == exp).sum()), len(pred)
    rng = np.random.default_rng(seed)
    null = np.array([(rng.permutation(pred) == exp).sum() for _ in range(perms)])
    return {"agree": k, "n": n, "p_permutation": float((null >= k).mean()), "permutations": perms,
            "clopper_pearson": [float(beta.ppf(0.025, k, n - k + 1)), float(beta.ppf(0.975, k + 1, n - k))]}


def _viz(did: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = json.load(open(RES / "real" / "viz" / f"{did}.json"))
    return np.array(v["points"]), np.array(v["labels"]), np.array(v["shape"])


def real_claims(results: dict) -> dict:
    """Quantities behind the post-hoc descriptions of real clouds in the paper and exp02 (top-3 PCA coordinates)."""
    out = {}
    x, lab, _ = _viz("R9_llama_truth_cities")                                  # label 1 = true statement (cities.csv)
    rms = {l: float(np.sqrt(((x[lab == l] - x[lab == l].mean(0)) ** 2).sum(1).mean())) for l in np.unique(lab)}
    out["R9_rms_spread"] = {"false": rms["0"], "true": rms["1"], "ratio_false_over_true": rms["0"] / rms["1"]}
    for did in ("R5a_llama_weekdays", "R5b_llama_months", "C3_llama_weekdays_init", "C3_llama_months_init"):
        x, lab, _ = _viz(did)
        cent = np.array([x[lab == l].mean(0) for l in np.unique(lab)])
        within = np.sqrt(np.mean([((x[lab == l] - x[lab == l].mean(0)) ** 2).sum(1).mean() for l in np.unique(lab)]))
        between = np.sqrt(((cent - cent.mean(0)) ** 2).sum(1).mean())
        out[f"{did}_token_clusters"] = {"n_tokens": int(len(cent)), "between_over_within_rms": float(between / within)}
    x, lab, shape = _viz("C3_llama_counting_init")
    d = np.linalg.norm(x - np.median(x, 0), axis=1)
    far = np.argsort(-d)[:3]
    out["C3_llama_counting_init"] = {"farthest_counts": [int(lab[i]) for i in far], "farthest_dist": [float(d[i]) for i in far],
                                     "median_dist": float(np.median(d)),
                                     "max_dist_fitted_centroid_to_nearest_point": float(max(np.linalg.norm(x - c, axis=1).min() for c in shape))}
    x, lab, _ = _viz("R8b_llama_age")
    o = np.argsort(lab.astype(int))
    jump = np.linalg.norm(np.diff(x[o], axis=0), axis=1)
    top = np.sort(np.argsort(-jump)[:8])
    out["R8b_largest_jumps"] = {"pairs": [[int(lab[o][i]), int(lab[o][i + 1])] for i in top], "median_jump": float(np.median(jump))}
    x, _, _ = _viz("R1_gpt2_wpos")
    p = results["R1_gpt2_wpos"]["params"]
    dist = np.abs(np.linalg.norm(x - np.array(p["center"]), axis=1) - p["radius"])
    out["R1_point_to_fitted_sphere"] = {"median": float(np.median(dist)), "q90": float(np.quantile(dist, 0.9)), "radius": float(p["radius"])}
    x, _, _ = _viz("R3_tms_n6")
    cos = (x @ x.T) / np.outer(np.linalg.norm(x, axis=1), np.linalg.norm(x, axis=1))
    out["R3_tms_n6_offdiag_cosines"] = sorted({round(float(c), 3) for c in cos[~np.eye(len(x), dtype=bool)]})
    out["helix_fit_handedness"] = {k: v["params"]["handedness"] for k, v in results.items() if v["family"] == "helix"}
    return out


def main() -> None:
    df = pd.read_csv(RES / "synth" / "per_sample.csv", low_memory=False)
    analysis = json.load(open(RES / "analysis.json"))
    results = {r["id"]: r for r in json.load(open(RES / "real" / "real_results.json"))}
    out = {"synthetic": recompute_synthetic(df), "H3": recompute_h3(analysis), "real_claims": real_claims(results)}
    path = RES / "audit_2026-11-22.json"
    json.dump(out, open(path, "w"), indent=2)
    print(json.dumps(out, indent=1))
    print("written", path)


if __name__ == "__main__":
    main()
