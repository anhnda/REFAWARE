"""
Real-image reference selection (paper §6 + §7.1).

CLI: load a torchvision pretrained classifier, run reference-aware K=1
explanation on one image under a family of references {black, gray, mean, blur,
inpaint}, estimate the held-out residual energy m_{>1,rho} for each, and select
the reference maximizing the explanation SNR  gamma = beta_min / (sigma_obs + c sqrt(m)).

This is *secondary* evidence (the paper says so): the decisive falsifiable test
is synthetic_verify.py. Here we additionally report insertion/deletion AUC so you
can eyeball whether lower m_hat tracks better faithfulness -- but note Theorem 1
does NOT prove that link (it is recoverability, not faithfulness; see §8.2).

Usage:
    python real_select.py --image cat.jpg --model resnet50 --grid 12 \
        --n-samples 2000 --device cuda --out-dir out/

    # library use:
    from lime import RefLIME, default_reference_family
    expl = RefLIME(model, device="cuda")
    res  = expl.explain(x)            # x: (1,3,H,W) normalized
    print(res.best_reference)

Per your preference, this script does NOT auto-run any torch work on import; it
only does so inside main() when you invoke it.
"""
from __future__ import annotations
import argparse, os, json
import numpy as np


def build_argparser():
    p = argparse.ArgumentParser(description="Reference-aware LIME selection on an image.")
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
                   help="comma-separated subset of the reference family")
    p.add_argument("--out-dir", default="out")
    p.add_argument("--insdel", action="store_true",
                   help="also compute insertion/deletion AUC per reference")
    return p


# --------------------------------------------------------------------------- #
#  Insertion/Deletion faithfulness (optional, secondary).
#  Uses the SAME selected reference as the masking baseline for consistency
#  (the §8.2 metric has its own reference; we align it to the fit reference).
# --------------------------------------------------------------------------- #
def insertion_deletion_auc(model, x, attr, rho, target, lib, device,
                           steps=50, batch=1):
    import torch
    import torch.nn.functional as F
    # rank cells by attribution
    n_cells = lib.n_cells
    cell_scores = np.zeros(n_cells)
    ids = lib.cell_ids.cpu().numpy()
    for c in range(n_cells):
        cell_scores[c] = attr[ids == c].mean()
    order = np.argsort(-cell_scores)  # high attribution first

    def curve(insertion: bool):
        ys = []
        keep_cells = np.zeros(n_cells) if insertion else np.ones(n_cells)
        chunk = max(1, n_cells // steps)
        with torch.no_grad():
            for s in range(0, n_cells + 1, chunk):
                z = torch.tensor(keep_cells, dtype=torch.float32).unsqueeze(0)
                keep = lib.to_pixel_keep(z)
                comp = rho(x, keep)
                p = F.softmax(model(comp.to(device)), dim=1)[0, target].item()
                ys.append(p)
                nxt = order[s:s + chunk]
                if insertion:
                    keep_cells[nxt] = 1.0
                else:
                    keep_cells[nxt] = 0.0
        return float(np.trapezoid(ys) / len(ys))
    return {"insertion_auc": curve(True), "deletion_auc": curve(False)}


def main():
    args = build_argparser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # imports inside main() so importing this module never touches torch
    import torch
    import torchvision as tv
    from torchvision import transforms
    from PIL import Image

    from lime import RefLIME, default_reference_family

    device = args.device
    weights_enum = tv.models.get_model_weights(args.model)
    weights = weights_enum.DEFAULT
    model = tv.models.get_model(args.model, weights=weights).eval().to(device)

    prep = weights.transforms()  # correct resize/crop/normalize for the model
    img = Image.open(args.image).convert("RGB")
    x = prep(img).unsqueeze(0).to(device)

    fam_all = default_reference_family()
    chosen = [r.strip() for r in args.references.split(",")]
    references = {k: fam_all[k] for k in chosen if k in fam_all}

    expl = RefLIME(model, device=device, grid=(args.grid, args.grid),
                   n_samples=args.n_samples, val_frac=args.val_frac,
                   c=args.c, batch_size=args.batch_size, seed=args.seed)
    res = expl.explain(x, target=args.target, references=references)

    # report
    summary = {"target": res.target, "best_reference": res.best_reference,
               "per_reference": {}}
    print(f"\ntarget class index: {res.target}")
    print(f"{'reference':>10} {'m_hat':>10} {'sigma_obs':>10} "
          f"{'floor':>10} {'beta_min':>10} {'SNR(gamma)':>11} {'#active':>8}")
    for name, pr in res.per_reference.items():
        print(f"{name:>10} {pr.m_hat:>10.5f} {pr.sigma_obs:>10.5f} "
              f"{pr.floor:>10.5f} {pr.beta_min:>10.5f} {pr.snr:>11.3f} "
              f"{pr.n_active:>8d}")
        np.save(os.path.join(args.out_dir, f"attr_{name}.npy"), pr.attr)
        entry = dict(m_hat=pr.m_hat, sigma_obs=pr.sigma_obs, floor=pr.floor,
                     beta_min=pr.beta_min, snr=pr.snr, n_active=pr.n_active)

        if args.insdel:
            id_metrics = insertion_deletion_auc(
                model, x, pr.attr, references[name], res.target,
                expl._lib, device)
            entry.update(id_metrics)
            print(f"           ins/del AUC: {id_metrics['insertion_auc']:.3f} "
                  f"/ {id_metrics['deletion_auc']:.3f}")
        summary["per_reference"][name] = entry

    print(f"\n>>> selected reference (max SNR): {res.best_reference}")
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved attributions + summary.json to {args.out_dir}/")


if __name__ == "__main__":
    main()
