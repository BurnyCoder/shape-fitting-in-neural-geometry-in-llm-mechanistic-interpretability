"""Tests for refinement, classical fitting, model selection and statistics."""

import numpy as np
import torch

from neurogeom.baselines.classical import fit_one_family, kmeans
from neurogeom.config import load_config
from neurogeom.data.synth import generate_family
from neurogeom.eval.stats import _sampler, bootstrap_ci, ece, fit_temperature, holm, paired_bootstrap, wilcoxon_p
from neurogeom.geometry.chamfer import chamfer
from neurogeom.infer.refine import refine
from neurogeom.infer.select import gaic_scores
from neurogeom.seeding import make_generator
from neurogeom.shapes.registry import INDEX, family

CFG = load_config()
CLEAN = dict(CFG["synth"], outlier_prob=0.0, aniso_noise_prob=0.0, embed_prob=0.0, density_modes=["uniform"])
DENSE = {"curve": 256, "surface_side": 24, "blob": 256}


def _data(name, b=8, n=200, sigma=0.0, seed=0):
    return generate_family(name, b, n, CLEAN, DENSE, make_generator(seed), "cpu", sigma=torch.full((b,), sigma), embed_dim=None)


def test_refine_recovers_ground_truth_from_perturbed_start():
    d = _data("helix")
    f = family("helix")
    raw_gt = f.encode(d["gt"])
    raw0 = raw_gt + 0.15 * torch.randn_like(raw_gt)
    raw, _ = refine(f, raw0, d["x"], d["mask"], None, steps=300, lr=1e-2)
    pts, w = f.dense(f.decode(raw), DENSE)
    assert chamfer(pts, w, d["gt_dense"], d["gt_w"]).median() < 0.05


def test_refine_does_not_inflate_half_circle():
    """A half circle must not be 'explained' by a full circle: the shape->points term forbids it."""
    f = family("arc")
    d = _data("arc", b=16)
    half = d["gt"]["extent_frac"] < 0.6
    raw0 = f.encode(d["gt"])
    raw0[:, -1] = 4.0                                              # start from a (nearly) full circle
    raw, _ = refine(f, raw0, d["x"], d["mask"], None, steps=400, lr=2e-2)
    frac = f.decode(raw)["extent_frac"]
    assert (frac[half] < d["gt"]["extent_frac"][half] + 0.15).all()


def test_classical_recovers_noiseless_circle_and_line():
    for name in ("arc", "line"):
        d = _data(name, b=6)
        c = fit_one_family(INDEX[name], d["x"], d["mask"], steps=300, back_w=0.25, gen=make_generator(1))
        f = family(name)
        pts, w = f.dense(f.decode(c["raw"]), DENSE)
        assert chamfer(pts, w, d["gt_dense"], d["gt_w"]).median() < 0.05, name


def test_kmeans_separates_clusters():
    x = torch.cat([torch.randn(1, 30, 3) * 0.05 + torch.tensor([3.0, 0, 0]), torch.randn(1, 30, 3) * 0.05 - torch.tensor([3.0, 0, 0])], 1)
    c = kmeans(x, torch.ones(1, 60, dtype=torch.bool), 2, gen=torch.Generator().manual_seed(0))
    assert torch.allclose(c[0, :, 0].abs().sort().values, torch.tensor([3.0, 3.0]), atol=0.1)


def test_gaic_prefers_simpler_model_when_fit_is_equal():
    n = torch.tensor([100.0])
    cands = [{"fi": INDEX["arc"], "J": torch.tensor([1.0]), "p": torch.tensor([8.0])},
             {"fi": INDEX["helix"], "J": torch.tensor([1.0]), "p": torch.tensor([9.0])},
             {"fi": INDEX["sphere"], "J": torch.tensor([0.9]), "p": torch.tensor([7.0])}]
    aic = gaic_scores(cands, n)[0]
    assert aic[0] < aic[1]                                         # same J, fewer parameters wins
    assert aic[0] < aic[2]                                         # 2D model pays 2 N eps^2 more


def test_bootstrap_and_holm_and_ece_and_temperature():
    rng = np.random.default_rng(0)
    est, lo, hi = bootstrap_ci(rng.normal(1.0, 1.0, 400), reps=500)
    assert lo < 1.0 < hi and lo < est < hi
    d, lo, hi = paired_bootstrap(np.ones(50) * 2, np.ones(50), reps=200)
    assert d == 1.0 and lo == 1.0
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == 0.03 and adj["c"] == 0.06 and adj["b"] == 0.06
    probs = np.eye(3)[rng.integers(0, 3, 300)]
    assert ece(probs, probs.argmax(1)) < 1e-9
    logits = torch.randn(500, 4) * 5
    labels = torch.distributions.Categorical(logits=logits / 5).sample()
    assert abs(fit_temperature(logits, labels) - 5.0) < 1.0      # recovers the true temperature


def test_gaic_can_never_select_the_volumetric_blob():
    # Audit 2026-11-14: with eps^2 estimated from the best non-volumetric candidate g, AIC(blob) - AIC(g) =
    # J_blob + J_g + 2 p_blob eps^2 > 0, so G-AIC never picks the d = 3 blob, whatever the residuals (reports/deviations.md).
    gen = torch.Generator().manual_seed(0)
    fams = ["blob", "sphere", "disc", "arc", "line", "clusters"]
    for n in (5, 12, 100, 1000):
        cands = [{"fi": INDEX[f], "J": torch.rand(64, generator=gen) * (0.01 if f == "blob" else 10.0),
                  "p": torch.full((64,), 9.0 if f == "blob" else 7.0)} for f in fams]
        aic = gaic_scores(cands, torch.full((64,), float(n)))
        assert not bool((aic.argmin(-1) == 0).any())        # blob (column 0) never wins, even with a 1000x smaller residual


def test_cluster_bootstrap_resamples_whole_groups():
    groups = np.repeat(np.arange(10), np.arange(1, 11))       # group g has g + 1 members
    draw = _sampler(len(groups), groups, np.random.default_rng(0))
    for _ in range(20):
        counts = np.bincount(groups[draw()], minlength=10)
        assert all(c % (g + 1) == 0 for g, c in enumerate(counts))   # every group enters whole (possibly several times)


def test_cluster_ci_is_wider_than_iid_ci_for_clustered_data():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), 25)
    v = rng.normal(0, 1, 40)[groups] + rng.normal(0, 0.1, len(groups))   # strong shared effect per cluster
    _, lo_i, hi_i = bootstrap_ci(v, reps=400)
    _, lo_c, hi_c = bootstrap_ci(v, reps=400, groups=groups)
    assert (hi_c - lo_c) > 2.5 * (hi_i - lo_i)
    a, b = v + 0.5, v                                          # constant paired shift
    d, lo, hi = paired_bootstrap(a, b, reps=200, groups=groups)
    assert abs(d - 0.5) < 1e-12 and lo > 0
    assert wilcoxon_p(a + rng.normal(0, 0.01, len(a)), b, groups=groups) < 1e-6   # test on 40 per-cluster mean differences


# ---------------------------------------------------------------- design checks (post hoc, audit 2026-11-22)
def test_stratified_cluster_bootstrap_keeps_whole_groups_and_groups_per_stratum():
    groups = np.repeat(np.arange(6), 3)                                   # 6 groups of 3 samples
    strata = np.repeat(np.array([0, 0, 1, 1, 1, 2]), 3)                   # strata hold 2, 3 and 1 groups
    draw = _sampler(len(groups), groups, np.random.default_rng(0), strata)
    for _ in range(20):
        idx = draw()
        assert np.bincount(strata[idx], minlength=3).tolist() == [6, 9, 3]   # the design's group count per stratum
        assert all(c % 3 == 0 for c in np.bincount(groups[idx], minlength=6))  # every group enters whole
        assert set(groups[idx][strata[idx] == 2]) == {5}                  # a one-group stratum always returns that group


def test_stratified_ci_is_narrower_than_collapsed_ci_when_stratum_means_differ():
    rng = np.random.default_rng(1)
    strata, groups = np.repeat(np.arange(10), 50), np.repeat(np.arange(100), 5)   # 10 strata x 10 groups x 5 samples
    v = strata + 0.1 * rng.normal(size=500)                               # large fixed differences between strata
    width = lambda t: t[2] - t[1]
    assert width(bootstrap_ci(v, reps=400, groups=groups, strata=strata)) < 0.5 * width(bootstrap_ci(v, reps=400, groups=groups))


def test_chamfer_floor_is_negligible_for_curves_but_not_for_surface_grids():
    from neurogeom.eval.metrics import chamfer_floor
    fl = chamfer_floor(n_shapes=8)
    assert set(fl) == {"line", "arc", "ellipse", "helix", "ringing", "disc", "sphere", "paraboloid", "cone", "torus"}
    assert max(v["vs_fine"] for v in fl.values() if v["coarse"] == "1d") < 0.03
    assert min(v["vs_fine"] for v in fl.values() if v["coarse"] == "2d") > 0.05
    assert fl["arc"]["vs_symmetric"] < 1e-3            # a reversed arc maps the midpoint grid onto itself
    assert fl["sphere"]["vs_symmetric"] > 0.02         # a sphere rotated about its pole does not


# ---------------------------------------------------------------- B1-fixed (post hoc, audit 2026-11-14)
def _unit(name, **spec):
    """A unit, axis-aligned, origin-centred instance of a posed family with the given intrinsic parameters."""
    p = {"center": torch.zeros(1, 3), "rot": torch.eye(3).unsqueeze(0)}
    p.update({k: torch.tensor([float(v)]) for k, v in spec.items()})
    return p


def test_projected_distance_matches_analytic_distance_and_never_exceeds_grid():
    from neurogeom.infer.refine import grid_dist, projected_dist
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(1, 300, 3, generator=gen) * 0.7
    dense = {"curve": 64, "surface_side": 12, "blob": 256}             # deliberately coarse grids
    circle = _unit("arc", radius=1.0, extent_frac=0.999)
    exact_c = torch.sqrt((x[..., :2].norm(dim=-1) - 1) ** 2 + x[..., 2] ** 2)
    keep = x[..., 2] > -0.6                                              # stay away from the sphere's tiny hole at the -z pole
    sphere = _unit("sphere", radius=1.0, cap_frac=0.999)
    exact_s = (x.norm(dim=-1) - 1).abs()
    for name, p, exact, m in (("arc", circle, exact_c, torch.ones_like(keep)), ("sphere", sphere, exact_s, keep)):
        f = family(name)
        d_grid, d_proj = grid_dist(f, p, x, dense), projected_dist(f, p, x, dense)
        assert bool((d_proj <= d_grid + 1e-6).all())                    # never worse than the nearest grid sample
        err_proj, err_grid = float((d_proj - exact)[m].abs().max()), float((d_grid - exact)[m].abs().max())
        assert err_proj < 2e-3                                           # close to the true orthogonal distance
        assert err_grid > 5 * err_proj                                   # while the coarse grid alone is clearly worse


def test_per_cloud_cluster_candidates_keep_k_up_to_each_clouds_own_n():
    from neurogeom.baselines.classical import fit_all_families
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(2, 30, 3, generator=gen)
    mask = torch.ones(2, 30, dtype=torch.bool)
    mask[0, 5:] = False                                                  # cloud 0 has 5 points, cloud 1 has 30
    x = x * mask.unsqueeze(-1)
    cands = fit_all_families(x, mask, 5, 0.25, gen, families=[INDEX["clusters"]], per_cloud_k=True)
    by_k = {int(c["select"][0]) + 2: c for c in cands}                   # k_idx j -> k = j + 2
    assert set(by_k) == {2, 3, 4, 5, 6}                                  # no k is dropped for the whole batch
    assert torch.isinf(by_k[6]["J"][0]) and torch.isfinite(by_k[6]["J"][1])   # k = 6 only for the 30-point cloud
    assert torch.isfinite(by_k[5]["J"]).all()                           # k = 5 <= 5 points is allowed


def test_blob_null_test_selects_blob_only_above_tau():
    from neurogeom.baselines.classical import classical_select
    n = torch.tensor([100.0, 100.0])
    cands = [{"fi": INDEX["blob"], "J": torch.tensor([1.0, 1.0]), "p": torch.tensor([9.0, 9.0])},
             {"fi": INDEX["sphere"], "J": torch.tensor([0.01, 4.0]), "p": torch.tensor([7.0, 7.0])},
             {"fi": INDEX["arc"], "J": torch.tensor([0.5, 9.0]), "p": torch.tensor([8.0, 8.0])}]
    mask = torch.ones(2, 100, dtype=torch.bool)
    sel = classical_select(cands, mask, blob_tau=0.1)
    # eps_hat = sqrt(J_sphere / N) = 0.01 (cloud 0, a surface) and 0.2 (cloud 1, too noisy for any surface)
    assert sel["cand_index"].tolist() == [1, 0]
    assert torch.allclose(sel["eps"], torch.sqrt(torch.tensor([0.01, 4.0]) / 100.0))
    assert classical_select(cands, mask)["cand_index"].tolist()[1] != 0   # preregistered B1: never blob


def test_blob_tau_is_fitted_to_separate_noisy_clouds_and_ties_go_to_the_smallest_tau():
    import pandas as pd
    from neurogeom.eval.synth_eval import tune_blob_tau
    rng = np.random.default_rng(0)
    eps = np.r_[rng.uniform(0.3, 0.5, 20), rng.uniform(0.0, 0.2, 80)]                # blobs imply large surface noise
    df = pd.DataFrame({"eps_hat": eps, "family_true": ["blob"] * 20 + ["arc"] * 40 + ["sphere"] * 40,
                       "family_pred": ["sphere"] * 20 + ["arc"] * 40 + ["sphere"] * 40})   # G-AIC never says blob
    res = tune_blob_tau(df)
    assert 0.2 <= res["tau"] < 0.3 and res["val_macro_f1"] == 1.0
    assert res["tau"] == min(t for t, s in zip(res["grid"], res["grid_macro_f1"]) if s == 1.0)
    assert res["val_macro_f1_never_blob"] < 1.0


def test_fast_macro_f1_equals_sklearn_including_predictions_outside_the_true_classes():
    from sklearn.metrics import f1_score
    from neurogeom.eval.stats import macro_f1
    rng = np.random.default_rng(0)
    names = np.array(["arc", "blob", "cone", "disc", "helix"])
    for _ in range(20):
        yt = names[rng.integers(0, 4, 300)]                      # "helix" never true ...
        yp = names[rng.integers(0, 5, 300)]                      # ... but sometimes predicted
        ref = f1_score(yt, yp, average="macro", labels=np.unique(yt), zero_division=0)
        assert abs(macro_f1(yt, yp) - ref) < 1e-12
