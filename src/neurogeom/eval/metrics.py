"""Per-sample evaluation metrics on synthetic data with known ground truth.

Global context: preregistered metrics (reports/preregistration.md Sec. 4):
- fine / coarse / acceptable-superset family accuracy (macro-F1 is computed from the per-sample table);
- Chamfer distance between the predicted shape and the noise-free ground-truth shape (primary
  geometric metric; symmetric, unsquared, the sum of the two directed mean distances, in units of the normalised
  cloud's RMS radius; both shapes are compared through their EVAL_DENSE samples, whose discretisation floor
  `chamfer_floor` measures);
- residual: mean distance from the observed points to the predicted shape;
- when the family is right: unsigned axis angle error (degrees) and |log scale ratio|.
"""

from __future__ import annotations

import math

import torch

from neurogeom.geometry.chamfer import chamfer, residual
from neurogeom.shapes.registry import COARSE, FAMILIES, NAMES, SUPERSETS

EVAL_DENSE = {"curve": 256, "surface_side": 24, "blob": 256}


@torch.no_grad()
def shape_metrics(pred_fi: list[int], raws: list[torch.Tensor], batch: dict) -> list[dict]:
    """pred_fi[i], raws[i]: predicted family and raw parameters for sample i of `batch` (on device)."""
    b = len(pred_fi)
    out = [dict() for _ in range(b)]
    fam_true = batch["family"].tolist()
    for fi in set(pred_fi):
        rows = [i for i in range(b) if pred_fi[i] == fi]
        r = torch.tensor(rows, device=batch["x"].device)
        f = FAMILIES[fi]
        p = f.decode(torch.stack([raws[i] for i in rows]).to(batch["x"].device))
        pts, w = f.dense(p, EVAL_DENSE)
        cd = chamfer(pts, w, batch["gt_dense"][r], batch["gt_w"][r])
        res = residual(batch["x"][r], batch["mask"][r], pts, w)
        axis, logscale = f.aux(p)
        for j, i in enumerate(rows):
            t = fam_true[i]
            rec = {"family_pred": NAMES[fi], "family_true": NAMES[t], "correct": fi == t,
                   "coarse_correct": COARSE[fi] == COARSE[t],
                   "superset_ok": fi == t or NAMES[fi] in SUPERSETS.get(NAMES[t], set()),
                   "chamfer": float(cd[j]), "residual": float(res[j]), "axis_err_deg": math.nan, "logscale_err": math.nan}
            if fi == t:
                if axis is not None and bool(batch["has_axis"][i]):
                    c = float((axis[j] * batch["gt_axis"][i]).sum().abs().clamp(max=1.0))
                    rec["axis_err_deg"] = math.degrees(math.acos(c))
                rec["logscale_err"] = float((logscale[j] - batch["gt_logscale"][i]).abs())
            out[i] = rec
    return out


@torch.no_grad()
def chamfer_floor(n_shapes: int = 64, sigma: float = 0.01, seed: int = 0, fine: int = 16) -> dict:
    """Discretisation floor of the evaluation Chamfer distance (post hoc, audit 2026-11-22; geometry only, no model).

    The metric compares EVAL_DENSE samples of the predicted and of the ground-truth shape, so a geometrically perfect
    prediction need not score 0. For every curve and surface family, `n_shapes` ground-truth shapes are drawn as the
    generator draws them (label margins at relative noise `sigma`) and scaled to RMS radius 1, then:
      vs_fine:      CD(evaluation samples, `fine` x as many samples of the same shape): one evaluation grid against
                    (nearly) the continuous shape;
      vs_symmetric: CD between the evaluation samples of two equivalent parameterisations of the same shape (the family's
                    listed symmetries, e.g. a sphere, cone or torus rotated by 1 rad about its axis); this is what a perfect
                    prediction scores when its parameterisation differs from the ground truth's.
    Medians over shapes. Point sets (exact vertices) and the blob (a fixed set of Gaussian draws) have no grid."""
    from neurogeom.seeding import make_generator
    gen = make_generator(seed)
    dense_fine = {"curve": EVAL_DENSE["curve"] * fine, "surface_side": int(EVAL_DENSE["surface_side"] * fine ** 0.5), "blob": EVAL_DENSE["blob"]}
    out = {}
    for f in FAMILIES:
        if not f.posed or f.name == "blob":
            continue
        vf, vs = [], []
        for _ in range(max(1, n_shapes // 8)):                         # batches of 8 keep the (8, M, 16 M) distances small
            p = f.sample_valid(8, gen, torch.full((8,), sigma), "cpu")
            p = f.transform(p, torch.eye(3).expand(8, 3, 3), torch.zeros(8, 3), 1.0 / f.rms_radius(p))   # RMS radius 1
            c, wc = f.dense(p, EVAL_DENSE)
            vf += chamfer(c, wc, *f.dense(p, dense_fine)).tolist()
            for q in f.symmetries(p):
                vs += chamfer(c, wc, *f.dense(q, EVAL_DENSE)).tolist()
        med = lambda v: float(torch.tensor(v).quantile(0.5))          # interpolated median, as numpy's
        out[f.name] = {"coarse": f.coarse, "vs_fine": med(vf), "vs_symmetric": med(vs) if vs else None, "n": len(vf)}
    return out


def sample_meta(cell: dict, ii: torch.Tensor, batch: dict) -> list[dict]:
    return [{"n_bin": cell["n_bin"], "noise_bin": cell["noise_bin"], "sample_id": f"{cell['n_bin']}-{cell['noise_bin']}-{int(i)}",
             "N": int(batch["n"][j]), "sigma": float(batch["sigma"][j]), "embed_dim": int(batch["embed_dim"][j])}
            for j, i in enumerate(ii.tolist())]
