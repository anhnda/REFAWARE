# Reference-Aware Perturbation Explanation (K=1)

Standalone implementation + verification of the reference-SNR recovery law from
"A Reference-Aware Theory of Perturbation Explanations" (§4, §6, §7, §8.1).

## Files
- `_core.py`         Torch-free: centered orthonormal design, Lasso (sklearn if
                     present, else numpy CD), VECTORIZED Lemma-1 leakage.
- `lime.py`          `RefLIME` + reference family (black/gray/mean/blur/inpaint)
                     + `MaskLibrary`. Faithful to Theorem 1: uniform mu masks,
                     centered chi basis, Lasso, held-out m_hat proxy. Library use
                     accepts ANY callable model (B,C,H,W)->logits.
- `synthetic_verify.py`  Falsifiable §8.1 tests. Decisive one is `lemma1`.
- `real_select.py`   CLI: torchvision ResNet-50/ViT, reference selection on an
                     image via the SNR criterion; optional insertion/deletion AUC.

## Install
    pip install numpy matplotlib scikit-learn torch torchvision pillow
(only numpy+matplotlib+scikit-learn needed for synthetic_verify.py)

## Verify the theory (no model needed)
    python synthetic_verify.py lemma1     # fast, vectorized; the clean proof
    python synthetic_verify.py floor      # Experiment 1 (slower)
    python synthetic_verify.py collapse   # Experiment 2 (slowest)
    python synthetic_verify.py all

`lemma1` should print ratio eta / sqrt(m log p / N) ~constant (~1.26) across a
16x range of m and 4x of N -> confirms the sqrt-law; that constant IS the c used
everywhere else.

## Real image
    python real_select.py --image cat.jpg --model resnet50 --device cuda \
        --grid 12 --n-samples 2000 --insdel --out-dir out/

## Vectorization status
- Lemma-1 leakage: fully vectorized over B trials (einsum, no Python loop).
- Lasso fit: sequential coordinate descent CANNOT be vectorized over coordinates
  (update j depends on residual from j-1). Speedup is via sklearn's compiled CD
  (default) and by batching the many INDEPENDENT fits across trials/beta/m.
- The image masking loop batches masks (`--batch-size`); the bottleneck is GPU
  forward passes = n_samples * n_references, linear in both.

## Caveats (honest)
- Theorem 1 is about RECOVERABILITY, not faithfulness. insertion/deletion AUC is
  reported only as secondary, non-decisive evidence (paper §8.2).
- The off-manifold => large m_hat claim (§7.1) is a conjecture, not proven.
- Standard LIME's distance kernel is DROPPED here on purpose; the floor is proven
  for uniform mu masks. Re-adding a kernel changes the design law.
