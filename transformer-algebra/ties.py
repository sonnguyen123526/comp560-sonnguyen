import argparse
import os
import pickle
import sys

import torch

sys.path.insert(0, '/Users/sonnguyen/comp560-nanoGPT')
from model import GPT, GPTConfig

BASE_CKPT        = 'out/base/ckpt.pt'
REVERSE_FT_CKPT  = 'out/reverse_ft/ckpt.pt'
ADDITION_FT_CKPT = 'out/addition_ft/ckpt.pt'

MIXED_META_PATH    = 'data/mixed/meta.pkl'
REVERSE_META_PATH  = 'data/mixed/meta.pkl'
ADDITION_META_PATH = 'data/mixed/meta.pkl'

REVERSE_TEST_FILE  = 'data/test/reverse_test.txt'
ADDITION_TEST_FILE = 'data/test/addition_test.txt'


def load_params(path):
    ckpt = torch.load(path, map_location='cpu')
    sd = {k.removeprefix('_orig_mod.'): v.float() for k, v in ckpt['model'].items()}
    return ckpt, sd


def trim(tau, density):
    # For each parameter tensor, zero out the bottom (1-density) fraction by magnitude
    trimmed = {}
    for k, t in tau.items():
        if density >= 1.0:
            trimmed[k] = t.clone()
        else:
            flat = t.flatten()
            threshold = torch.quantile(torch.abs(flat), 1.0 - density)
            mask = torch.abs(t) >= threshold
            trimmed[k] = t.clone()
            trimmed[k][~mask] = 0.0
    return trimmed


def elect_sign(tau_rev, tau_add):
    # Compute majority sign: sign(tau_rev + tau_add)
    elected = {}
    for k in tau_rev:
        elected[k] = torch.sign(tau_rev[k] + tau_add[k])
    return elected


def disjoint_merge(tau_rev, tau_add, elected):
    # For each parameter, average the task vectors whose sign matches the elected sign
    merged = {}
    for k in tau_rev:
        tr = tau_rev[k]
        ta = tau_add[k]
        e = elected[k]

        mask_rev = torch.sign(tr) == e
        mask_add = torch.sign(ta) == e

        avg = torch.zeros_like(tr)
        count = torch.zeros_like(tr).float()

        avg[mask_rev] = tr[mask_rev] + avg[mask_rev]
        count[mask_rev] += 1.0
        avg[mask_add] = ta[mask_add] + avg[mask_add]
        count[mask_add] += 1.0

        safe_count = torch.where(count > 0, count, torch.ones_like(count))
        merged[k] = torch.where(count > 0, avg / safe_count, torch.zeros_like(avg))

    return merged


def build_ties_model(density, lam):
    base_ckpt, theta_base = load_params(BASE_CKPT)
    _,          theta_rev  = load_params(REVERSE_FT_CKPT)
    _,          theta_add  = load_params(ADDITION_FT_CKPT)

    tau_rev = {k: theta_rev[k] - theta_base[k] for k in theta_base}
    tau_add = {k: theta_add[k] - theta_base[k] for k in theta_base}

    tau_rev_trimmed = trim(tau_rev, density)
    tau_add_trimmed = trim(tau_add, density)

    elected = elect_sign(tau_rev_trimmed, tau_add_trimmed)

    tau_ties = disjoint_merge(tau_rev_trimmed, tau_add_trimmed, elected)

    merged = {k: theta_base[k] + lam * tau_ties[k] for k in theta_base}

    model = GPT(GPTConfig(**base_ckpt['model_args']))
    model.load_state_dict(merged)
    model.eval()
    return model


def build_naive_model(lam_rev, lam_add):
    base_ckpt, theta_base = load_params(BASE_CKPT)
    _,          theta_rev  = load_params(REVERSE_FT_CKPT)
    _,          theta_add  = load_params(ADDITION_FT_CKPT)

    tau_rev = {k: theta_rev[k] - theta_base[k] for k in theta_base}
    tau_add = {k: theta_add[k] - theta_base[k] for k in theta_base}

    merged = {
        k: theta_base[k] + lam_rev * tau_rev[k] + lam_add * tau_add[k]
        for k in theta_base
    }

    model = GPT(GPTConfig(**base_ckpt['model_args']))
    model.load_state_dict(merged)
    model.eval()
    return model


def load_meta(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_test_examples(path, n):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines[:n]


@torch.no_grad()
def greedy(model, prompt_ids, max_new, device):
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    bs = model.config.block_size
    for _ in range(max_new):
        logits, _ = model(x[:, -bs:])
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        x = torch.cat([x, nxt], dim=1)
    return x[0, len(prompt_ids):].tolist()


def evaluate_reverse(model, meta, n, device):
    stoi, itos = meta['stoi'], meta['itos']
    examples = load_test_examples(REVERSE_TEST_FILE, n)
    correct = 0
    for ex in examples:
        parts  = ex.split(' -> ')
        prompt = parts[0] + ' ->'
        exp    = parts[1].strip()
        digits = prompt.replace(' ->', '').split()
        try:
            ids = [stoi[c] for c in prompt]
        except KeyError:
            continue
        out = greedy(model, ids, len(digits) * 2 + 4, device)
        got = ''.join(itos[i] for i in out).split('\n')[0].strip()
        if ' '.join(got.split()[:len(digits)]) == exp:
            correct += 1
    return correct / len(examples) if len(examples) > 0 else 0.0


def evaluate_addition(model, meta, n, device):
    stoi, itos = meta['stoi'], meta['itos']
    examples = load_test_examples(ADDITION_TEST_FILE, n)
    correct = 0
    for ex in examples:
        prompt = ex.rsplit(' -> ', 1)[0] + ' ->'
        exp    = ''.join(ex.rsplit(' -> ', 1)[1].strip().split())
        try:
            ids = [stoi[c] for c in prompt]
        except KeyError:
            continue
        out = greedy(model, ids, 12, device)
        got = ''.join(''.join(itos[i] for i in out).split('\n')[0].strip().split())
        if got == exp:
            correct += 1
    return correct / len(examples) if len(examples) > 0 else 0.0


def run_sweep(device, n):
    rev_meta = load_meta(REVERSE_META_PATH)
    add_meta = load_meta(ADDITION_META_PATH)

    densities = [0.2, 0.4, 0.6, 0.8, 1.0]
    lambdas   = [0.2, 0.4, 0.6, 0.8, 1.0]

    print()
    print('=' * 60)
    print('TIES-Merging Results')
    print('=' * 60)
    print(f"{'density':>8}  {'lam':>5}  {'rev':>8}  {'add':>8}")
    print('-' * 60)

    baseline_model = build_naive_model(1.0, 1.0).to(device)
    base_rev = evaluate_reverse(baseline_model, rev_meta, n, device)
    base_add = evaluate_addition(baseline_model, add_meta, n, device)
    print(f"{'naive':>8}  {1.0:>5.1f}  {base_rev*100:>7.1f}%  {base_add*100:>7.1f}%")
    print('-' * 60)

    for density in densities:
        for lam in lambdas:
            model   = build_ties_model(density, lam).to(device)
            rev_acc = evaluate_reverse(model,  rev_meta, n, device)
            add_acc = evaluate_addition(model, add_meta, n, device)
            print(f"{density:>8.1f}  {lam:>5.1f}  {rev_acc*100:>7.1f}%  {add_acc*100:>7.1f}%")

    print()


def run_single(density, lam, device, n):
    rev_meta = load_meta(REVERSE_META_PATH)
    add_meta = load_meta(ADDITION_META_PATH)

    model = build_ties_model(density, lam).to(device)
    rev_acc = evaluate_reverse(model, rev_meta, n, device)
    add_acc = evaluate_addition(model, add_meta, n, device)

    print()
    print('=' * 40)
    print('TIES Single Run')
    print('=' * 40)
    print(f'density={density:.2f} lam={lam:.2f}')
    print(f'rev: {rev_acc*100:.1f}%')
    print(f'add: {add_acc*100:.1f}%')
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--density', type=float, default=1.0)
    parser.add_argument('--lam',     type=float, default=1.0)
    parser.add_argument('--sweep',   action='store_true')
    parser.add_argument('--n',       type=int,   default=200)
    parser.add_argument('--device',  type=str,   default='mps')
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if args.sweep:
        run_sweep(args.device, args.n)
    else:
        run_single(args.density, args.lam, args.device, args.n)


if __name__ == '__main__':
    main()
