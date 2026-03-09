"""Week 2 — Task Vector Conflict Analysis (weight-space).

For each parameter group in the transformer (embed, layer0.attn, layer0.mlp, …):

  cos(τ_rev, τ_add)   — cosine similarity between the two task vectors.
                         Negative values mean the updates directly oppose each
                         other in weight space, which causes cancellation when
                         you sum them during merging.

  sign_conflict %     — fraction of scalar entries where sign(τ_rev) ≠ sign(τ_add).
                         This is the core diagnostic from TIES-Merging: a high
                         conflict rate means naive summation inverts many weights.

  ‖τ_rev‖, ‖τ_add‖    — L2 norm of each task vector per group.  Large norms
                         indicate the layers that changed most during fine-tuning.

No forward passes are needed — only the three saved checkpoints.

Usage:
  cd transformer-algebra
  python week2_task_vector_conflict.py
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, '/Users/sonnguyen/comp560-nanoGPT')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_CKPT        = 'out/base/ckpt.pt'
REVERSE_FT_CKPT  = 'out/reverse_ft/ckpt.pt'
ADDITION_FT_CKPT = 'out/addition_ft/ckpt.pt'


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------

def load_params(ckpt_path):
    """Return a flat state-dict {name: float32 tensor} stripped of _orig_mod. prefix."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    return {k.removeprefix('_orig_mod.'): v.float()
            for k, v in ckpt['model'].items()}


# ---------------------------------------------------------------------------
# Per-group statistics
# ---------------------------------------------------------------------------

def cosine_sim(a, b):
    a, b = a.flatten(), b.flatten()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def sign_conflict(a, b):
    """Fraction of non-zero entries where sign(a) ≠ sign(b)."""
    a, b = a.flatten(), b.flatten()
    mask = (a != 0) & (b != 0)
    if mask.sum() == 0:
        return 0.0
    return (torch.sign(a[mask]) != torch.sign(b[mask])).float().mean().item()


def group_key(param_name):
    """Map a full parameter name to a short display-group label."""
    if param_name.startswith('transformer.h.'):
        parts  = param_name.split('.')   # ['transformer','h','0','attn',...]
        layer  = parts[2]                # '0' … '3'
        submod = parts[3]                # 'attn' | 'mlp' | 'ln_1' | 'ln_2'
        return f'layer{layer}.{submod}'
    if 'wte' in param_name or 'wpe' in param_name:
        return 'embed'
    if 'ln_f' in param_name:
        return 'ln_final'
    return 'lm_head'


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run(theta_base, theta_rev, theta_add):
    tau_rev = {k: theta_rev[k] - theta_base[k] for k in theta_base}
    tau_add = {k: theta_add[k] - theta_base[k] for k in theta_base}

    # Collect parameter names for each display group
    groups = {}
    for k in theta_base:
        groups.setdefault(group_key(k), []).append(k)

    print()
    print('=' * 74)
    print('Task Vector Conflict Analysis')
    print('=' * 74)
    print(f"  {'Group':<28}  {'cos(τ_rev,τ_add)':>16}  {'sign_conflict':>13}  {'‖τ_rev‖':>8}  {'‖τ_add‖':>8}")
    print('-' * 74)

    all_cos, all_sc = [], []

    for group in sorted(groups):
        cos_vals, sc_vals, norms_rev, norms_add = [], [], [], []
        for k in groups[group]:
            tr, ta = tau_rev[k], tau_add[k]
            if tr.numel() > 1:          # skip scalar (bias) params
                cos_vals.append(cosine_sim(tr, ta))
                sc_vals.append(sign_conflict(tr, ta))
            norms_rev.append(tr.norm().item())
            norms_add.append(ta.norm().item())

        cos_mean = sum(cos_vals) / len(cos_vals) if cos_vals else float('nan')
        sc_mean  = sum(sc_vals)  / len(sc_vals)  if sc_vals  else float('nan')
        nr_mean  = sum(norms_rev) / len(norms_rev)
        na_mean  = sum(norms_add) / len(norms_add)

        print(f"  {group:<28}  {cos_mean:>16.4f}  {sc_mean:>13.2%}  {nr_mean:>8.4f}  {na_mean:>8.4f}")
        all_cos.extend(cos_vals)
        all_sc.extend(sc_vals)

    print('-' * 74)
    if all_cos:
        print(f"  {'OVERALL':<28}  {sum(all_cos)/len(all_cos):>16.4f}  {sum(all_sc)/len(all_sc):>13.2%}")

    print()
    print('Interpretation:')
    print('  cos < 0        → τ_rev and τ_add actively oppose each other in that group')
    print('  sign_conflict  → fraction of scalar weights pushed in opposite directions')
    print('  Large ‖τ‖      → fine-tuning moved those weights the most')


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print('Loading checkpoints ...')
    theta_base = load_params(BASE_CKPT)
    theta_rev  = load_params(REVERSE_FT_CKPT)
    theta_add  = load_params(ADDITION_FT_CKPT)

    run(theta_base, theta_rev, theta_add)


if __name__ == '__main__':
    main()
