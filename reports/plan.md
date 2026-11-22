# Plan and status

This file tracks the project plan and its status. Results live in [summary.md](summary.md), every change to the preregistered plan in [deviations.md](deviations.md), and the preregistration itself in [preregistration.md](preregistration.md).

## Stages (order of work)

| Stage | Content | Status |
|---|---|---|
| 1 | Literature catalog, preregistration, core code, README | done (2026-10-31) |
| 2 | First model `main_s0`, with the controls and baselines that need no training; paper, reports, site, Hugging Face | done (2026-10-31); fact-checks 2026-11-01 and 2026-11-04 |
| 2a | Independent soundness audit of stage 2 and its fixes (below) | done (2026-11-15; PR #10) |
| 2b | Second independent audit (no model runs): recomputation, design checks, fact-check (below) | done (2026-11-22) |
| 3 | Trained ablations A0, A1, A2, A4, A6 (`bash scripts/sweep.sh ablations`), including H5a | postponed by the author |
| 4 | Seeds 1–2 and hyperparameter sensitivity (`bash scripts/sweep.sh sensitivity`); hierarchical bootstrap over seeds | postponed by the author |
| 5 | OOD synthetic set (4D flat torus, two-frequency Fourier circles, multi-period number helix) | postponed |
| 6 | Next models suggested by the failures: coverage check, Fourier-curve family, more cluster centroids, helix test against circles and clusters, abstention | not started |

## Stage 2a: audit of 2026-11-14

**Question.** Is any methodological error invalidating the findings of `main_s0`?

**Findings.** None flips a preregistered verdict. Details are in [deviations.md](deviations.md), section "Found in the independent audit of 2026-11-14".
1. Geometric AIC, as implemented, can never select the blob (B1, A5-G-AIC). This inflates the H1 effect size and the share of the A5 drop attributed to model selection.
2. The test clouds are clustered: chunks of 25 share N and the R^D-embedding draw. The i.i.d. CIs were too narrow.
3. The "unambiguous" subset is defined by B1's own fits, which favours B1.
4. H3 had no trivial baseline. A constant "curve" prediction does at least as well as the method. Fine-level agreement was never stated.
5. H4's "supported" depends on averaging over datasets; the per-dataset criterion fails for the smallest clouds.

**Fixes.**
- [x] Cluster bootstrap for every synthetic CI. Wilcoxon on per-chunk mean differences.
- [x] Post-hoc outputs: blob recall, non-blob H1/H2, H3 constant baseline and fine agreement, H4 per-dataset count.
- [x] B1-fixed implemented with tests: per-cloud cluster candidates, exact point-to-shape J, and a blob null test with τ fitted on `val`. Smoke-tested.
- [x] Deviations written, including the B1-fixed pre-specification (before the test run).
- [x] Push the code and pre-specification before the long runs (branch `fix/audit-2026-11-14`, 2026-11-15 01:17).
- [x] `run.py tune-b1-fixed` on 780 `val` clouds (60 min GPU; tau saved in `results/main_s0/synth/b1_fixed_tau.json`).
- [x] `run.py eval-b1-fixed` on the 3,120-cloud baseline subset (138 min GPU). B1-fixed is only slightly better than B1; H1/H2 hold against it.
- [x] Generators: headline lines, H4 verdict text, post-hoc rows, LaTeX macros, error bars on the by-N and by-noise figures. Macro-F1 is now a bincount implementation identical to sklearn, and the analysis takes about 2 min instead of more than 30.
- [x] Text: README, summary, exp01, exp02, methodology, paper ("strongest" wording, A5 attribution, "Successes" → "Coarse agreements", B1-conditioned unambiguous table, cluster bootstrap, limitations, README to-do).
- [x] Text that depends on the B1-fixed result: interpretation in exp01, the paper's results, abstract and conclusion, the README and the summary.
- [x] Regenerate the report, site and paper PDF. Publish the Hugging Face card. PR and merge (PR #10, 2026-11-15).

## Stage 2b: audit of 2026-11-22

**Question.** Is any methodological error invalidating the findings? Are the hypotheses evaluated correctly, and are methods, definitions and citations used correctly? No model inference or training.

**Findings.** None flips a verdict. The review is in [reviews/audit-2026-11-22.md](reviews/audit-2026-11-22.md); see also the deviations section "Found in the independent audit of 2026-11-22".
1. The Chamfer metric has a sampling-grid floor for surfaces, of the order of the reported medians.
2. The pooled H5b effect hides opposite signs: refinement helps point sets, curves and surfaces and hurts blobs.
3. The cluster-bootstrap CIs are conservative (collapsed strata); a stratified sensitivity check narrows them.
4. H4's preregistered wording admits a pooled and a per-dataset reading; the paper had claimed the per-dataset one.
5. Two descriptions of real clouds were wrong (the random-init counting control, the true/false statements), as were one bibliography author order and one year label.

**Fixes.**
- [x] `scripts/audit_checks.py`: an independent recomputation with sklearn and scipy, plus checks of the real-cloud descriptions. Output: `results/main_s0/audit_2026-11-22.json`.
- [x] `posthoc.design_checks` in `analysis.json`: stratified cluster bootstrap, refinement by coarse class, Chamfer floor. Unit tests, a headline line and LaTeX macros come with it.
- [x] Text: paper (statistics, H5b, H4, R9, limitations, date), methodology, exp01, exp02, deviations, bibliography, catalog corrections.
- [x] Regenerate the report and the paper PDF. PR and merge. Hugging Face card.

## Open to-do after stages 2a and 2b
- Re-run the real-data B1 column with B1-fixed (currently the preregistered B1).
- Recompute the ambiguity tag with B1-fixed fits, or with a ground-truth-based definition.
- A B1-fixed variant with k < N cluster candidates. With k ≤ N, G-AIC picks degenerate one-point-per-cluster fits on the smallest clouds. This would be another post-hoc choice, so if run it must be reported next to the pre-specified B1-fixed.
- Evaluate the Chamfer metric on a finer grid (needs a re-evaluation with the model), or store predicted parameters so that metrics can be recomputed without the model.
- Stages 3–6 above.
