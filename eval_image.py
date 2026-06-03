"""
eval_image.py -- end-to-end image evaluation for the Reference-Aware
Perturbation Explanation method (paper SS6 + SS8.2).

Pipeline
--------
1. Load a torchvision pretrained classifier and one image.
2. Run the method (RefLIME.explain) over the FULL reference family.
   - This produces, for every method reference rho_m, an attribution map
     per_reference[rho_m].attr and the SS6 SNR score gamma.
   - The paper's selection criterion (max gamma) picks `best_reference`.
3. Measure insertion / deletion faithfulness.
   IMPORTANT -- there are TWO independent uses of a reference here:
     (a) the METHOD reference rho_m : the masking operator used while fitting
         the surrogate / producing an attribution map. The method already
         selects the best one by SNR, but we keep every map so we can audit.
     (b) the MEASURE baseline rho_b : the masking operator used to occlude /
         reveal cells while sweeping the insertion & deletion curves.
   We evaluate the full cross product: every attribution map (one per method
   reference) is scored under every measure baseline in the family.
4. For each attribution map we aggregate ins/del AUC across measure baselines
   and report mean +/- std in two modes:
     - "all"      : average over ALL measure baselines.
     - "exclude"  : average over baselines EXCEPT the one matching the method
                    reference that produced the map (avoids the SS8.2
                    circularity of measuring with the same reference you fit
                    on). For the selected explanation this is the headline
                    number.

The decisive falsifiable test is still synthetic_verify.py (paper SS8.1);
ins/del here is secondary evidence (Theorem 1 is recoverability, not
faithfulness; see SS8.2).

Usage
-----
    python eval_image.py --image cat.jpg --model resnet50 --grid 12 \
        --n-samples 2000 --device cuda --steps 50 --measure-repeat 4 \
        --out-dir out/

Note: this script does NOT touch torch on import; all torch work happens
inside main() when you invoke it. It will download pretrained weights on
first run.
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def build_argparser():
    p = argparse.ArgumentParser(
        description="Reference-aware explanation + cross-baseline ins/del eval."
    )
    p.add_argument("--image", required=True, help="path to an RGB image")
    p.add_argument("--model", default="resnet50",
                   help="torchvision model name (resnet50, vit_b_16, ...)")
    p.add_argument("--device", default="cpu", help="cpu | cuda")
    p.add_argument("--grid", type=int, default=12, help="grid is (grid,grid)")
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--val-frac", type=float, default=0.3)
    p.add_argument("--c", type=float, default=1.26,
                   help="leakage constant (see synthetic_verify.py lemma1)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target", type=int, default=None,
                   help="class index; default = model's top-1")
    p.add_argument("--references", default="black,gray,mean,blur,inpaint",
                   help="comma-separated subset of the reference family used "
                        "BOTH as method references and as measure baselines")
    p.add_argument("--steps", type=int, default=50,
                   help="ins/del curve resolution (number of reveal chunks)")
    p.add_argument("--measure-repeat", type=int, default=4,
                   help="repeats per stochastic measure baseline (averaged) to "
                        "tame sampling variance of inpaint; >=1")
    p.add_argument("--out-dir", default="out")
    return p


# --------------------------------------------------------------------------- #
#  Insertion / Deletion faithfulness.
#
#  Reworked from real_select.insertion_deletion_auc to:
#    * accept an arbitrary MEASURE baseline rho_b (decoupled from the method
#      reference that produced `attr`);
#    * batch the forward passes over the curve;
#    * average over `repeat` draws when rho_b is stochastic.
#  The attribution `attr` is an (H,W) painted coefficient map; we reduce it to
#  one score per grid cell (mean over the cell's pixels), then rank cells.
# --------------------------------------------------------------------------- #
def _cell_order_from_attr(attr, lib):
    """High-attribution-first ordering of cell ids from an (H,W) map."""
    ids = lib.cell_ids.detach().cpu().numpy()           # (H,W)
    n_cells = lib.n_cells
    cell_scores = np.full(n_cells, -np.inf, dtype=np.float64)
    for c in range(n_cells):
        m = ids == c
        if m.any():
            cell_scores[c] = float(attr[m].mean())
    return np.argsort(-cell_scores)                     # high first


def insertion_deletion_auc(model, x, attr, rho_b, target, lib, device,
                           steps=50, repeat=1):
    """Insertion & deletion AUC for one attribution map under one MEASURE
    baseline rho_b. Returns mean over `repeat` draws (repeat folded to 1 for
    deterministic rho_b)."""
    import torch
    import torch.nn.functional as F

    n_cells = lib.n_cells
    order = _cell_order_from_attr(attr, lib)
    is_stoch = bool(getattr(rho_b, "is_stochastic", False))
    reps = max(1, repeat) if is_stoch else 1

    def one_curve(insertion: bool, seed_offset: int):
        # keep_cells: 1 = reveal original pixel, 0 = replace by rho_b
        keep_cells = np.zeros(n_cells) if insertion else np.ones(n_cells)
        chunk = max(1, n_cells // steps)
        ys = []
        # Build the sequence of cell-keep vectors along the sweep, then batch.
        keep_states = []
        s = 0
        while True:
            keep_states.append(keep_cells.copy())
            if s >= n_cells:
                break
            nxt = order[s:s + chunk]
            keep_cells[nxt] = 1.0 if insertion else 0.0
            s += chunk
        Z = torch.tensor(np.stack(keep_states), dtype=torch.float32)  # (T,n_cells)
        with torch.no_grad():
            for b in range(0, Z.shape[0], 64):
                zb = Z[b:b + 64]
                keep = lib.to_pixel_keep(zb)             # (B,1,H,W)
                comp = rho_b(x, keep)                    # apply MEASURE baseline
                p = F.softmax(model(comp.to(device)), dim=1)[:, target]
                ys.extend(p.detach().cpu().numpy().tolist())
        return float(np.trapezoid(ys) / len(ys))

    ins_vals, del_vals = [], []
    for r in range(reps):
        ins_vals.append(one_curve(True, r))
        del_vals.append(one_curve(False, r))
    return {
        "insertion_auc": float(np.mean(ins_vals)),
        "deletion_auc": float(np.mean(del_vals)),
    }


# --------------------------------------------------------------------------- #
#  Aggregation across measure baselines
# --------------------------------------------------------------------------- #
def _agg(per_baseline, key, exclude=None):
    """mean/std of `key` over baselines, optionally excluding one name."""
    vals = [v[key] for name, v in per_baseline.items() if name != exclude]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=0)),
            "n": len(vals)}


# --------------------------------------------------------------------------- #
#  Within-image agreement between the selection signal and faithfulness.
#
#  These test the SS8.2 conjecture *on this one image* across the method
#  references. With ~5 references this is descriptive only -- no p-value; the
#  real inference is across many images (dataset loop, added later).
#
#  SS8.2 conjecture predicts: lower m_hat  -> better faithfulness, i.e.
#      m_hat   vs insertion_auc : NEGATIVE rho   (less leakage -> higher ins)
#      m_hat   vs deletion_auc  : POSITIVE rho   (less leakage -> lower  del)
#  and the SNR gamma, being inverse to the floor, flips both signs.
# --------------------------------------------------------------------------- #
def _spearman(a, b):
    """Spearman rank correlation, ties averaged. Pure numpy. nan if degenerate."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2:
        return float("nan")

    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        # average ties
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, r)
        avg = sums / counts
        return avg[inv]

    ra, rb = rank(a), rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((ra * rb).sum() / denom)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    args = build_argparser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # torch imports live inside main() so importing this module never touches torch
    import torchvision as tv
    from PIL import Image

    from lime import RefLIME, default_reference_family

    device = args.device

    # ---- model + image ----
    weights = tv.models.get_model_weights(args.model).DEFAULT
    model = tv.models.get_model(args.model, weights=weights).eval().to(device)
    prep = weights.transforms()
    img = Image.open(args.image).convert("RGB")
    x = prep(img).unsqueeze(0).to(device)

    # ---- reference family (shared by method side and measure side) ----
    fam_all = default_reference_family()
    chosen = [r.strip() for r in args.references.split(",")]
    references = {k: fam_all[k] for k in chosen if k in fam_all}
    if not references:
        raise SystemExit(f"no valid references in {chosen!r}; "
                         f"available: {list(fam_all)}")

    # ---- 1) run the method over the whole family + SS6 selection ----
    expl = RefLIME(model, device=device, grid=(args.grid, args.grid),
                   n_samples=args.n_samples, val_frac=args.val_frac,
                   c=args.c, batch_size=args.batch_size, seed=args.seed)
    res = expl.explain(x, target=args.target, references=references)
    lib = expl._lib

    print(f"\ntarget class index: {res.target}")
    print(f"{'method_ref':>10} {'m_hat':>10} {'sigma_obs':>10} "
          f"{'beta_min':>10} {'SNR(gamma)':>11} {'#active':>8}")
    for name, pr in res.per_reference.items():
        flag = "  <== selected" if name == res.best_reference else ""
        print(f"{name:>10} {pr.m_hat:>10.5f} {pr.sigma_obs:>10.5f} "
              f"{pr.beta_min:>10.5f} {pr.snr:>11.3f} {pr.n_active:>8d}{flag}")
        np.save(os.path.join(args.out_dir, f"attr_{name}.npy"), pr.attr)

    # ---- 2) measure: full cross product (method attr) x (measure baseline) ----
    print(f"\nrunning ins/del over {len(res.per_reference)} attribution maps "
          f"x {len(references)} measure baselines ...")
    eval_block = {}
    for m_name, pr in res.per_reference.items():
        per_baseline = {}
        for b_name, rho_b in references.items():
            md = insertion_deletion_auc(
                model, x, pr.attr, rho_b, res.target, lib, device,
                steps=args.steps, repeat=args.measure_repeat)
            per_baseline[b_name] = md

        # aggregate across measure baselines: two modes
        agg_all = {
            "insertion_auc": _agg(per_baseline, "insertion_auc"),
            "deletion_auc": _agg(per_baseline, "deletion_auc"),
        }
        agg_excl = {
            "insertion_auc": _agg(per_baseline, "insertion_auc", exclude=m_name),
            "deletion_auc": _agg(per_baseline, "deletion_auc", exclude=m_name),
        }
        eval_block[m_name] = {
            "per_baseline": per_baseline,
            "agg_all": agg_all,
            "agg_exclude_self": agg_excl,
        }

        sel = "  <== selected" if m_name == res.best_reference else ""
        print(f"\n[method ref: {m_name}]{sel}")
        print(f"  all baselines     : "
              f"ins {agg_all['insertion_auc']['mean']:.3f}"
              f" +/- {agg_all['insertion_auc']['std']:.3f} | "
              f"del {agg_all['deletion_auc']['mean']:.3f}"
              f" +/- {agg_all['deletion_auc']['std']:.3f}")
        print(f"  exclude-self ({m_name:>5}): "
              f"ins {agg_excl['insertion_auc']['mean']:.3f}"
              f" +/- {agg_excl['insertion_auc']['std']:.3f} | "
              f"del {agg_excl['deletion_auc']['mean']:.3f}"
              f" +/- {agg_excl['deletion_auc']['std']:.3f}")

    # ---- 2b) within-image agreement: selection signal vs faithfulness ----
    # Build aligned vectors across method references, using the exclude-self
    # (SS8.2-clean) aggregates as the faithfulness numbers.
    ref_names = list(res.per_reference.keys())
    m_hat_v = np.array([res.per_reference[n].m_hat for n in ref_names])
    snr_v = np.array([res.per_reference[n].snr for n in ref_names])
    ins_v = np.array([eval_block[n]["agg_exclude_self"]["insertion_auc"]["mean"]
                      for n in ref_names])
    del_v = np.array([eval_block[n]["agg_exclude_self"]["deletion_auc"]["mean"]
                      for n in ref_names])

    best = res.best_reference
    # rank of the selected ref: 1 = best (highest insertion / lowest deletion)
    ins_rank = int(np.sum(ins_v > ins_v[ref_names.index(best)]) + 1)
    del_rank = int(np.sum(del_v < del_v[ref_names.index(best)]) + 1)
    n_ref = len(ref_names)
    ins_winner = ref_names[int(np.argmax(ins_v))]
    del_winner = ref_names[int(np.argmin(del_v))]

    spearman = {
        "m_hat_vs_insertion": _spearman(m_hat_v, ins_v),   # conjecture: < 0
        "m_hat_vs_deletion": _spearman(m_hat_v, del_v),    # conjecture: > 0
        "snr_vs_insertion": _spearman(snr_v, ins_v),       # conjecture: > 0
        "snr_vs_deletion": _spearman(snr_v, del_v),        # conjecture: < 0
    }
    agreement = {
        "selected_reference": best,
        "n_references": n_ref,
        "selected_insertion_rank": ins_rank,
        "selected_deletion_rank": del_rank,
        "insertion_winner": ins_winner,
        "deletion_winner": del_winner,
        "selected_won_insertion": bool(ins_winner == best),
        "selected_won_deletion": bool(del_winner == best),
        "spearman_exclude_self": spearman,
    }

    print("\n" + "-" * 60)
    print("within-image agreement (selection signal vs faithfulness)")
    print(f"  selected ref: {best}")
    print(f"  insertion: rank {ins_rank}/{n_ref} "
          f"(winner: {ins_winner}{' == selected' if ins_winner == best else ''})")
    print(f"  deletion : rank {del_rank}/{n_ref} "
          f"(winner: {del_winner}{' == selected' if del_winner == best else ''})")
    print(f"  Spearman rho (across {n_ref} refs, exclude-self AUCs):")
    print(f"    m_hat vs insertion: {spearman['m_hat_vs_insertion']:+.3f}"
          f"   (SS8.2 expects < 0)")
    print(f"    m_hat vs deletion : {spearman['m_hat_vs_deletion']:+.3f}"
          f"   (SS8.2 expects > 0)")
    print(f"    gamma vs insertion: {spearman['snr_vs_insertion']:+.3f}"
          f"   (expects > 0)")
    print(f"    gamma vs deletion : {spearman['snr_vs_deletion']:+.3f}"
          f"   (expects < 0)")
    print("  note: ~5 refs -> descriptive only; pool across images for inference")
    print("-" * 60)

    # ---- 3) headline: the SS6-selected explanation ----
    head_all = eval_block[best]["agg_all"]
    head_excl = eval_block[best]["agg_exclude_self"]
    print("\n" + "=" * 60)
    print(f">>> SELECTED explanation (max SNR): {best}")
    print(f"    ins/del, ALL baselines       : "
          f"ins {head_all['insertion_auc']['mean']:.3f}"
          f" +/- {head_all['insertion_auc']['std']:.3f} | "
          f"del {head_all['deletion_auc']['mean']:.3f}"
          f" +/- {head_all['deletion_auc']['std']:.3f}")
    print(f"    ins/del, EXCLUDING {best:>5} base : "
          f"ins {head_excl['insertion_auc']['mean']:.3f}"
          f" +/- {head_excl['insertion_auc']['std']:.3f} | "
          f"del {head_excl['deletion_auc']['mean']:.3f}"
          f" +/- {head_excl['deletion_auc']['std']:.3f}")
    print("=" * 60)

    # ---- 4) dump everything ----
    summary = {
        "model": args.model,
        "target": res.target,
        "best_reference": best,
        "references": list(references),
        "method": {
            name: dict(m_hat=pr.m_hat, sigma_obs=pr.sigma_obs,
                       floor=pr.floor, beta_min=pr.beta_min, snr=pr.snr,
                       n_active=pr.n_active)
            for name, pr in res.per_reference.items()
        },
        "eval": eval_block,
        "agreement": agreement,
        "config": {
            "grid": args.grid, "n_samples": args.n_samples,
            "val_frac": args.val_frac, "c": args.c, "steps": args.steps,
            "measure_repeat": args.measure_repeat, "seed": args.seed,
        },
    }
    with open(os.path.join(args.out_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved attributions + eval_summary.json to {args.out_dir}/")


if __name__ == "__main__":
    main()