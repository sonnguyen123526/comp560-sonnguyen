import argparse
import os
import random
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

sys.path.insert(0, '/Users/sonnguyen/comp560-nanoGPT')
from model import GPT, GPTConfig

from ties import (
    ADDITION_FT_CKPT,
    ADDITION_META_PATH,
    BASE_CKPT,
    COMPOSITE_TEST_FILE,
    MIXED_META_PATH,
    REVERSE_FT_CKPT,
    REVERSE_META_PATH,
    evaluate_addition,
    evaluate_composite,
    evaluate_reverse,
    generate_composite_test,
    load_meta,
    load_params,
)


def apply_dare(tau, drop_rate, seed, rescale=True):
    """Apply DARE to a task vector.

    DARE randomly drops a fraction of coordinates and rescales survivors by
    1/(1-drop_rate) so expected magnitude is preserved.
    """
    if drop_rate <= 0.0:
        return {k: v.clone() for k, v in tau.items()}

    if drop_rate >= 1.0:
        return {k: torch.zeros_like(v) for k, v in tau.items()}

    g = torch.Generator(device='cpu')
    g.manual_seed(seed)

    keep_prob = 1.0 - drop_rate
    scale = (1.0 / keep_prob) if rescale else 1.0

    out = {}
    for k, v in tau.items():
        mask = (torch.rand(v.shape, generator=g) < keep_prob).to(v.dtype)
        out[k] = v * mask * scale
    return out


def build_dare_model(drop_rate, lam, seed=42, rescale=True):
    base_ckpt, theta_base = load_params(BASE_CKPT)
    _, theta_rev = load_params(REVERSE_FT_CKPT)
    _, theta_add = load_params(ADDITION_FT_CKPT)

    tau_rev = {k: theta_rev[k] - theta_base[k] for k in theta_base}
    tau_add = {k: theta_add[k] - theta_base[k] for k in theta_base}

    tau_rev_dare = apply_dare(tau_rev, drop_rate=drop_rate, seed=seed, rescale=rescale)
    # Offset seed so rev/add masks are different but reproducible.
    tau_add_dare = apply_dare(tau_add, drop_rate=drop_rate, seed=seed + 100_003, rescale=rescale)

    merged = {
        k: theta_base[k] + lam * (tau_rev_dare[k] + tau_add_dare[k])
        for k in theta_base
    }

    model = GPT(GPTConfig(**base_ckpt['model_args']))
    model.load_state_dict(merged)
    model.eval()
    return model


def run_single(drop_rate, lam, seed, rescale, device, n):
    rev_meta = load_meta(REVERSE_META_PATH)
    add_meta = load_meta(ADDITION_META_PATH)

    model = build_dare_model(drop_rate, lam, seed=seed, rescale=rescale).to(device)
    rev_acc = evaluate_reverse(model, rev_meta, n, device)
    add_acc = evaluate_addition(model, add_meta, n, device)

    print()
    print('=' * 44)
    print('DARE Single Run')
    print('=' * 44)
    print(f'drop_rate={drop_rate:.2f} lam={lam:.2f} seed={seed} rescale={rescale}')
    print(f'rev: {rev_acc*100:.1f}%')
    print(f'add: {add_acc*100:.1f}%')
    print()


def run_sweep(device, n, seed, rescale):
    rev_meta = load_meta(REVERSE_META_PATH)
    add_meta = load_meta(ADDITION_META_PATH)

    drop_rates = [0.2, 0.4, 0.6, 0.8]
    lambdas = [0.2, 0.4, 0.6, 0.8, 1.0]

    print()
    print('=' * 64)
    print('DARE Results')
    print('=' * 64)
    print(f"{'drop_rate':>10}  {'lam':>5}  {'rev':>8}  {'add':>8}")
    print('-' * 64)

    for drop_rate in drop_rates:
        for lam in lambdas:
            model = build_dare_model(drop_rate, lam, seed=seed, rescale=rescale).to(device)
            rev_acc = evaluate_reverse(model, rev_meta, n, device)
            add_acc = evaluate_addition(model, add_meta, n, device)
            print(f"{drop_rate:>10.1f}  {lam:>5.1f}  {rev_acc*100:>7.1f}%  {add_acc*100:>7.1f}%")

    print()


def run_composite(device, n, seed, drop_rate, lam, rescale, composite_file):
    if not os.path.exists(composite_file):
        generate_composite_test(composite_file, n, seed)

    meta = load_meta(MIXED_META_PATH)
    model = build_dare_model(drop_rate, lam, seed=seed, rescale=rescale).to(device)
    acc = evaluate_composite(model, meta, n, device, composite_file)

    print()
    print('=' * 44)
    print('DARE Composite Eval')
    print('=' * 44)
    print(f'drop_rate={drop_rate:.2f} lam={lam:.2f} seed={seed} rescale={rescale}')
    print(f'composite: {acc*100:.1f}%')
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drop_rate', type=float, default=0.2)
    parser.add_argument('--lam', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rescale', action='store_true')
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--composite_eval', action='store_true')
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--composite_file', type=str, default=COMPOSITE_TEST_FILE)
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.sweep:
        run_sweep(args.device, args.n, args.seed, args.rescale)
    elif args.composite_eval:
        run_composite(
            args.device,
            args.n,
            args.seed,
            args.drop_rate,
            args.lam,
            args.rescale,
            args.composite_file,
        )
    else:
        run_single(args.drop_rate, args.lam, args.seed, args.rescale, args.device, args.n)


if __name__ == '__main__':
    main()
