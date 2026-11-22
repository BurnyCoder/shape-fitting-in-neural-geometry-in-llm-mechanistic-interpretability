# Experiment 1: training the fitter and evaluating it on synthetic shapes

**Run:** `main_s0`
- Config: `configs/main.yaml` on top of `configs/base.yaml`.
- Commands, in order:
  - `uv run python run.py train --config main.yaml`
  - `uv run python run.py eval-synth --config main.yaml`
  - `uv run python run.py report --config main.yaml`
- Logs: `logs/*_main_s0_*.log`.

Numbers between the markers are generated from `results/main_s0/` and must not be edited by hand.

## Question
The network is trained only on synthetic shapes. Does it identify the shape family and recover the shape better than a classical pipeline (B1) that fits every family and picks one with geometric AIC? This is preregistered as H1 and H2. Does test-time refinement help (H5b)? B1 as implemented has three handicaps (see the audit learning below), so the comparison is repeated post hoc against a repaired version, B1-fixed.

## Setup
- **Model:** Set Transformer (ISAB×2 with 16 inducing points, PMA, d = 128) with 13 per-family parameter heads. The parameter count is in the models table below.
- **Data:** on-the-fly synthetic shapes from the `train` seed stream, with the nuisances listed in [../docs/methodology.md](../docs/methodology.md).
- **Loss:** CE (label smoothing 0.05) + Chamfer(predicted shape, noise-free ground-truth shape) + 0.5·Huber residual (δ = 0.05) + 0.1·CE on the cluster count / polytope template + annealed auxiliary terms. The geometric terms are averaged within each family, then over families.
- **Optimization:**
  - AdamW, lr 3e-4, weight decay 1e-4, 2k-step warmup, then a cosine schedule planned over 25k steps;
  - batch = max(64, min(512, 262144 // N)) clouds, sliced per family (about 507 per step at small N);
  - wall-clock cap of 110 minutes on an RTX 5070 Laptop (8 GB), which stopped training before the end of the cosine schedule (steps in the models table).
- **Test set:** stream `test`, 250 clouds per cell of family × N-bin × noise-bin, 39,000 clouds in total. Clouds are generated in chunks of 25 that share N and the R^D-embedding draw, so every CI is a cluster bootstrap over chunks.
- **Baselines:** run on a fixed subset of 20 clouds per family per cell (3,120 clouds, i.e. 156 chunks); see [deviations.md](deviations.md).
- **B1-fixed (post hoc):** B1 with per-cloud cluster counts, exact point-to-shape residuals, and a blob threshold on the RMS residual of the best surface fit, fitted on 780 `val` clouds. Same starts and steps as B1; same 3,120 test clouds. It was pre-specified in [deviations.md](deviations.md) before it ran on the test set (`run.py tune-b1-fixed eval-b1-fixed`).

## Models trained

<!-- BEGIN:models_table -->
| Run | Encoder | Training objective | Data / curriculum | Params | Steps | Minutes | Val. accuracy | Test fine accuracy | Test median Chamfer |
|---|---|---|---|---|---|---|---|---|---|
| main_s0 | Set Transformer | CE + Chamfer + residual + aux | full synthetic mix | 1,712,093 | 22643 | 110 | 90.2% | 87.9% | 0.156 |
<!-- END:models_table -->

## Results

<!-- BEGIN:results_table -->
| System | n | Fine accuracy [95% CI] | Macro-F1 [95% CI] | Coarse accuracy | Superset-OK accuracy | Median Chamfer to GT [95% CI] | Mean residual | Median axis error (deg) |
|---|---|---|---|---|---|---|---|---|
| Method (network + refinement) | 39000 | 87.9% [86.9, 88.9] | 0.879 [0.869, 0.889] | 92.7% | 88.4% | 0.156 [0.148, 0.164] | 0.105 | 1.1 |
| Method, baseline subset | 3120 | 87.4% [83.7, 90.9] | 0.873 [0.829, 0.904] | 92.3% | 87.9% | 0.154 [0.133, 0.180] | 0.106 | 1.0 |
| A3: no refinement | 39000 | 87.9% [86.9, 88.9] | 0.879 [0.869, 0.889] | 92.7% | 88.4% | 0.159 [0.152, 0.167] | 0.120 | 1.5 |
| A5: G-AIC selection | 39000 | 64.6% [63.0, 66.0] | 0.627 [0.616, 0.637] | 77.5% | 68.8% | 0.182 [0.173, 0.192] | 0.108 | 0.8 |
| A5: hybrid selection | 39000 | 85.6% [84.4, 86.8] | 0.858 [0.846, 0.869] | 91.1% | 86.6% | 0.157 [0.149, 0.165] | 0.103 | 1.1 |
| B1: classical fit + G-AIC | 3120 | 52.2% [47.2, 57.5] | 0.499 [0.453, 0.533] | 70.7% | 58.1% | 0.209 [0.183, 0.249] | 0.117 | 0.7 |
| B1-fixed: repaired classical fit (post hoc) | 3120 | 54.0% [49.0, 58.7] | 0.535 [0.486, 0.571] | 70.2% | 59.7% | 0.215 [0.190, 0.253] | 0.098 | 0.9 |
| B2: features + boosting | 3120 | 64.5% [59.0, 70.3] | 0.645 [0.582, 0.691] | 85.1% | 68.0% | 0.237 [0.194, 0.300] | 0.120 | 1.1 |
| B3a: always blob | 3120 | 7.7% [3.8, 12.2] | 0.011 [0.006, 0.017] | 7.7% | 7.7% | 0.543 [0.524, 0.573] | 0.120 | 11.7 |
| B3b: always arc (B1 arc fit) | 3120 | 7.7% [3.8, 12.2] | 0.011 [0.006, 0.017] | 38.5% | 7.7% | 0.545 [0.521, 0.570] | 0.276 | 0.5 |
| B3c: random shape | 3120 | 7.9% [7.0, 8.9] | 0.079 [0.068, 0.087] | 32.9% | 10.9% | 1.116 [1.074, 1.159] | 0.589 | 58.3 |
<!-- END:results_table -->

The table above restricted to test clouds that no simpler family fits equally well, judged by B1's own per-family fits (the "unambiguous" subset). A cloud whose true family B1 fits badly also counts as ambiguous. This subset therefore conditions on B1's fitting success and favours B1; read it as a check, not as intrinsic ambiguity:

<!-- BEGIN:unambiguous_table -->
| System | n | Fine accuracy [95% CI] | Macro-F1 [95% CI] | Coarse accuracy | Superset-OK accuracy | Median Chamfer to GT [95% CI] | Mean residual | Median axis error (deg) |
|---|---|---|---|---|---|---|---|---|
| Method (network + refinement) | 2060 | 94.0% [91.6, 95.9] | 0.922 [0.888, 0.942] | 96.4% | 94.3% | 0.119 [0.094, 0.139] | 0.091 | 0.7 |
| A3: no refinement | 2060 | 94.0% [91.6, 95.9] | 0.922 [0.888, 0.942] | 96.4% | 94.3% | 0.127 [0.102, 0.141] | 0.101 | 1.1 |
| A5: G-AIC selection | 2060 | 72.0% [67.8, 75.8] | 0.654 [0.622, 0.673] | 81.5% | 76.5% | 0.138 [0.119, 0.153] | 0.097 | 0.6 |
| A5: hybrid selection | 2060 | 92.6% [90.1, 94.8] | 0.902 [0.871, 0.925] | 94.8% | 93.1% | 0.119 [0.094, 0.140] | 0.090 | 0.7 |
| B1: classical fit + G-AIC | 2060 | 69.7% [64.8, 74.2] | 0.636 [0.587, 0.668] | 82.5% | 75.4% | 0.154 [0.139, 0.170] | 0.103 | 0.6 |
| B1-fixed: repaired classical fit (post hoc) | 2060 | 70.1% [65.6, 74.3] | 0.673 [0.623, 0.704] | 82.5% | 76.2% | 0.159 [0.145, 0.174] | 0.086 | 0.7 |
| B2: features + boosting | 2060 | 72.3% [66.4, 77.9] | 0.697 [0.634, 0.735] | 92.5% | 76.2% | 0.152 [0.136, 0.173] | 0.099 | 0.6 |
| B3a: always blob | 2060 | 4.5% [1.9, 7.6] | 0.007 [0.003, 0.011] | 4.5% | 4.5% | 0.523 [0.497, 0.543] | 0.119 | 10.2 |
| B3b: always arc (B1 arc fit) | 2060 | 7.3% [3.2, 12.0] | 0.011 [0.005, 0.016] | 36.3% | 7.3% | 0.528 [0.495, 0.555] | 0.273 | 0.3 |
| B3c: random shape | 2060 | 8.3% [7.2, 9.5] | 0.079 [0.067, 0.089] | 34.1% | 11.6% | 1.114 [1.064, 1.173] | 0.582 | 58.6 |
<!-- END:unambiguous_table -->

### Preregistered hypotheses

<!-- BEGIN:hypotheses_table -->
| Hypothesis | Comparison | Estimate [95% CI] | p (Holm) | n | Verdict |
|---|---|---|---|---|---|
| H1 | macro-F1(method) - macro-F1(B1) | 0.3741 [0.3410, 0.4095] | < 1e-16 | 3120 | supported |
| H2 | median Chamfer(method) - median Chamfer(B1) | -0.0554 [-0.0766, -0.0412] | < 1e-16 | 3120 | supported |
| H5b_refinement | median Chamfer(method) - median Chamfer(no refinement) | -0.0033 [-0.0055, -0.0011] | < 1e-16 | 39000 | supported |
| H1, post hoc | macro-F1(method) - macro-F1(B1), clouds whose true family is not blob | 0.3191 [0.2848, 0.3560] | - | 2880 | method better |
| H1, post hoc | macro-F1(method) - macro-F1(B1-fixed) | 0.3381 [0.3062, 0.3746] | - | 3120 | method better |
| H2, post hoc | median Chamfer(method) - median Chamfer(B1-fixed) | -0.0608 [-0.0846, -0.0454] | < 1e-16 (unadjusted) | 3120 | method better |
| H3 | Coarse-label agreement (subsample majority) with literature on preregistered real datasets | 7/12 = 58% [28, 85] (exact binomial); post hoc: constant 'curve' 8/12, exact family 1/12 | perm. p = 0.125 | 12 | not supported |
| H4 | Controls: matched Gaussians -> blob (mean over 17 real datasets); ordinal processes -> ringing (not helix) | blob rate 86%; ringing 60% vs helix 40% | - | - | supported (mean criterion); 10/12 preregistered datasets reach 80% |
<!-- END:hypotheses_table -->

### Figures
- Training curves: ![training](../docs/figures/fig_training.png)
- Systems compared: ![systems](../docs/figures/fig_systems.png)
- Accuracy against N and against noise: ![by N](../docs/figures/fig_by_n.png) ![by noise](../docs/figures/fig_by_noise.png)
- Confusion matrix: ![confusion](../docs/figures/fig_confusion.png)
- Calibration: ![reliability](../docs/figures/fig_reliability.png)
- Example fits, method versus classical B1: ![gallery](../docs/figures/fig_gallery_synth.png)

## Learnings
These notes record what we tried, what went wrong, and how each problem was found and fixed. The timings were measured with `torch.profiler` and wall-clock timers during development; they are development observations, not results.

1. **Batched small linear algebra on the GPU was the bottleneck, not the model.**
   - The first generator took about 0.3 s per training step. The torch profiler showed `linalg_qr` accounting for 75% of CPU time with 68k kernel launches: in our profile, batched QR on CUDA launched about one kernel per matrix.
   - Fix: random rotations from normalized 4D Gaussians, i.e. uniformly distributed unit quaternions (Shoemake 1992, Graphics Gems III; [`urot.c`, `Qt_RandomG`](https://github.com/erich666/GraphicsGems/blob/master/gemsiii/urot.c)); determinants via the triple product; tiny SVDs and eigendecompositions moved to the CPU.
   - FP64 `eigh` on the GPU was also slow, because consumer GPUs (compute capability 8.6, 8.9, 12.0) run FP64 at 1/64 of FP32 throughput ([CUDA Programming Guide, Compute Capabilities, row "Ratio of FP32 to FP64 Throughput": 64:1](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)). It now runs in FP32 on the GPU, with FP64 kept for real data on the CPU.
   - Result: the generator became about 7× faster, at 0.04 s per step.
2. **Training is kernel-launch bound.**
   - The GPU was busy for only about 20% of each step. A batch of 512 costs about the same wall time as a batch of 256, so throughput doubled.
   - A batch of 512 with N = 1024 exceeded 8 GB, and Windows silently spilled into shared memory at 3.3 s per step. A token budget (batch × N ≤ 262144) prevents this.
3. **PCA of a 3D subspace of R^D can return a mirrored frame.**
   - In 3D we force det = +1. With D > 3, however, the map from the shape's 3D frame to the PCA frame can be a reflection.
   - Ground-truth parameters therefore need a mirror-aware transform: a y-mirror for most families, which flips the helix pitch sign, and a z-mirror for the ringing curve, which flips a3. A regression test checks reflections for every family.
4. **Flat shapes plus a 3rd PC that is pure noise make the frame map rank-deficient.**
   - For an arc embedded in R^64 with no noise, the third PC is an arbitrary null direction. The 3×3 map from the shape frame to the PCA frame then has singular values (1, 1, 0).
   - Using the mean singular value as the scale was wrong. The correct scale is exactly 1/RMS, because both maps are isometries on the shape's plane.
   - The Chamfer target never had this problem, because it is computed from exactly projected ground-truth samples.
5. **Test-design pitfalls caught by TDD.**
   - Comparing shapes point by point fails after a mirror, because the traversal direction flips. Shapes must be compared as point sets.
   - Symmetry tests need a sampling much finer than the grid's half-spacing. Otherwise a rotated phase grid looks like an error.

### Learnings from the results

The numbers behind these points are in the generated tables and figures above; none is typed here.

6. **The learned fitter beats the classical pipeline on every point-count bin (H1, H2 supported).**
   - On the same baseline-subset clouds, the method has higher macro-F1 and a lower median Chamfer distance to the ground-truth shape than B1, which fits every family and selects with geometric AIC. Both paired-bootstrap CIs exclude 0.
   - Against B1 the accuracy gap grows with N: it is smallest for the smallest clouds (5–12 points), where every system struggles. Against B2 the gap is largest for the smallest clouds. See the accuracy-by-N figure.
   - The feature-based classifier B2 sits between the two overall and for N ≥ 13; for 5–12 points it is below B1 (accuracy-by-N figure).
   - **The conclusion does not depend on B1's handicaps (post hoc).** The repaired baseline B1-fixed is only slightly better than B1, and the method's lead over it is about as large as over B1 (B1-fixed lines of the headline and the post-hoc rows of the hypotheses table).
7. **A large part of the gap is model selection, and part of that is structural.**
   - A5 keeps the network's refined candidates but picks among them by geometric AIC instead of the classifier, and accuracy drops close to B2's level.
   - Part of this drop is guaranteed: as implemented, geometric AIC can never select the blob (audit learning 12). B1 and A5 therefore miss every blob cloud, 1/13 of the test set. On the clouds whose true family is not blob, A5's drop is smaller but still large (Selection-rule line of the summary headline).
   - A5 still chooses only among the classifier's top-3 candidates, so it is not a purely classical selector. The remaining difference between A5 and B1 also contains fitting and candidate effects.
   - With nested families and realistic noise, a residual-versus-complexity criterion often prefers the more flexible family. The classifier has learned the prior over families and noise that generated the data; on real activations that prior does not hold (exp02).
8. **Refinement helps the fit through the points, not family identification.**
   - Refinement never changes the selected family, so A3 has exactly the method's accuracy.
   - It lowers the mean point-to-shape residual clearly, but the median Chamfer distance to the ground truth only slightly. H5b is supported, with a small effect.
   - The network's direct parameter prediction is already close to the ground-truth shape; refinement mostly snaps the shape onto the noisy points.
   - **Post hoc (audit 2026-11-22): the pooled H5b estimate mixes opposite effects.** Split by the true family's dimension, refinement lowers the median Chamfer distance for point sets, curves and surfaces by similar amounts, each larger than the pooled estimate. For blob clouds it raises it ("Audit 2026-11-22" line of the summary headline). A likely reason, not tested: the two-sided point-to-sample objective is not an estimate of a Gaussian's covariance. It moves the blob's fixed set of 256 Gaussian draws toward the noisy points, away from the shape the network predicted. For curves, whose grid floor is small (learning 13), the improvement is a real gain in shape recovery.
9. **Hardest families.**
   - Cone, helix and paraboloid have the lowest recall; see the confusion matrix.
   - The largest single confusions are helix ↔ cone (sparsely sampled helix turns lie on a cylinder- or cone-like surface), paraboloid ↔ cone and sphere ↔ cone (curved sheets around an axis). Even so, most helix and paraboloid errors go to families outside this trio.
   - The fine-versus-coarse gap in the results table shows that fewer than half of the errors stay within the right dimension (curve, surface, point set): coarse accuracy is only a few points above fine accuracy.
10. **Calibration.**
    - The raw classifier is slightly underconfident (the fitted T is below 1), consistent with the 0.05 label smoothing, which "creates artificially underconfident models" ([Minderer et al. 2021](https://arxiv.org/abs/2106.07998), Appendix D).
    - A single temperature fitted on the `val` stream ([Guo et al. 2017](https://arxiv.org/abs/1706.04599)) roughly halves the expected calibration error. The NLL-optimal temperature on the test set itself is almost the same (both values are in the Calibration line of the summary headline).
11. **Engineering: autograd memory in refinement.**
    - The first full evaluation stalled at 21.5k of 39k clouds. Refining 3 candidate surfaces for 512 clouds of 256 points kept several (B, N, M) distance matrices for the backward pass. That needed more than 8 GB, and Windows paged GPU memory to system RAM.
    - Fix: find the nearest-point index without autograd, in chunks, and differentiate only the gathered pairs. This is the split of pytorch3d's [`knn_points` + `knn_gather`](https://pytorch3d.readthedocs.io/en/latest/modules/ops.html). Training Chamfer and the evaluation metric still use the full distance matrix. Value and gradient are identical up to ties (unit test), and peak memory fell by more than 10×.
    - Reported numbers come from the restarted evaluation; see [deviations.md](deviations.md).
12. **Independent audit of 2026-11-14: what was found and how.** The audit re-read the evaluation code against the preregistration and recomputed the per-sample results. Details and the old values are in [deviations.md](deviations.md).
    - **Geometric AIC cannot output blob.** Found because B1's confusion matrix has an empty blob column. Kanatani's noise estimate ε² = J_g / ((3 − d_g)N − p_g) is taken from the best surface fit g. Substituting it gives AIC(blob) − AIC(g) = J_blob + J_g + 2·p_blob·ε² > 0 for every cloud, so the d = 3 model always loses. A noise level estimated from a model cannot be used to reject that model. Kanatani's criterion assumes the general model is correct ([Kanatani 2004](https://iim.cs.tut.ac.jp/member/kanatani/papers/pamodel2.pdf)), which a blob violates.
    - **Fix: B1-fixed.** We chose a null test over a full likelihood comparison of a Gaussian blob against "manifold + noise" densities. Such a comparison needs the shape's area normalisation and a sampling finer than the noise, which a 40 × 40 surface grid does not provide at σ = 0.003. In B1-fixed the blob is chosen when the RMS residual of the best surface fit exceeds a threshold τ fitted on `val`. A first version thresholded the dof-corrected ε̂. A smoke run showed that it explodes for N ≤ p, where the degrees of freedom are clamped to 1, so the RMS residual replaced it before the full run. B1-fixed also repairs the batch-wide cluster-count skip and replaces the nearest-grid-sample residual with an orthogonal projection (Gauss–Newton point inversion, backtracked; unit-tested against analytic circle and sphere distances).
    - **Clustered test set.** Found by reading `data/testsets.py`. N is drawn once per chunk of 25 clouds, and the R^D embedding once per generator call. The baseline subset is exactly the first 20 clouds of one chunk per family and cell. Resampling clouds as if independent made every CI too narrow. The cluster bootstrap over chunks widens them (the deviations row lists the old values); no verdict changes.
    - **Result of B1-fixed.** B1-fixed recovers about half of the blob clouds and does better than B1 on the largest clouds. Its macro-F1 gain over B1 is small and its accuracy gain is not significant, because two effects offset the repairs (accuracy-by-N figure, per-family recall in `analysis.json`). (1) Its false blob calls cost recall in other families. (2) The per-cloud cluster repair allows k ≤ N, which matches the generator: with 5 points it can draw 5 clusters. But a k = N fit has zero residual, and the G-AIC penalty of a point set has no dN term, so on the smallest clouds G-AIC often prefers these degenerate one-point-per-cluster fits. B1-fixed is slightly worse than B1 there. This explanation came after the run. A k < N variant would be another post-hoc choice and was not run (open item in [plan.md](plan.md)).
    - **Engineering.** The cluster bootstrap made the 2000-replicate macro-F1 CIs the slowest part of the report: sklearn's `f1_score` on 39k string labels took about 0.3 s per call. A bincount implementation gives identical values (unit test against sklearn) and took 0.04 s on the same strings. With integer family codes, the whole analysis ran in under 2 minutes instead of more than 30 (development timings).
13. **Independent audit of 2026-11-22: what was found and how.** This audit recomputed the headline numbers from `per_sample.csv` with sklearn and scipy. Accuracy, macro-F1, median Chamfer, the H3 count, its permutation p and its Clopper–Pearson CI all reproduced. It then asked what each metric and interval actually measures. Values are in the "Audit 2026-11-22" line of the summary headline; details are in [deviations.md](deviations.md) and [reviews/audit-2026-11-22.md](reviews/audit-2026-11-22.md).
    - **The cluster-bootstrap CIs are conservative.** Found by asking which units are random in the test design. Families and cells are fixed by design (10 chunks per family × N-bin × noise-bin cell). Resampling chunks across cells treats the family mix as random and adds the spread between cell means. A stratified cluster bootstrap (chunks resampled within their cell) gives clearly narrower full-test-set CIs. It cannot be computed on the baseline subset, where each cell is one chunk, so the reported CIs stay as they are: valid, but wider than necessary. No verdict changes.
    - **The Chamfer metric has a grid floor for surfaces.** Found by computing the metric between a shape and itself. Both shapes enter through fixed dense samples: 256 per curve and a 24 × 24 grid per surface. Against a 16× denser sampling of the same shape, the evaluation sampling scores an order of magnitude more for surfaces than for curves. A sphere, cone or torus rotated about its own axis scores well above zero against itself. Absolute median Chamfer values of surface families therefore partly measure the grid. The floor enters every system's score, and the rotation about a symmetry axis cannot be inferred from the points, so we expect it to add noise to H2 rather than favour one system. A finer evaluation grid would remove it, but that means re-evaluating with the model, which this audit did not do.
    - **H5b by coarse class.** See learning 8.
