import argparse
import os
import pickle
import random
import sys

import torch

sys.path.insert(0, '/Users/sonnguyen/comp560-nanoGPT')
from model import GPT, GPTConfig

SEED = 42

BASE_CKPT        = 'out/base/ckpt.pt'
REVERSE_FT_CKPT  = 'out/reverse_ft/ckpt.pt'
ADDITION_FT_CKPT = 'out/addition_ft/ckpt.pt'


def load_params(path):
    ckpt = torch.load(path, map_location='cpu')
    sd = {k.removeprefix('_orig_mod.'): v.float() for k, v in ckpt['model'].items()}
    return ckpt, sd


def build_merged_model(lam_rev, lam_add):
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
    random.seed(SEED)
    stoi, itos = meta['stoi'], meta['itos']
    correct = 0
    for _ in range(n):
        digits = [random.randint(0, 9) for _ in range(random.randint(2, 4))]
        inp    = ' '.join(str(d) for d in digits)
        exp    = ' '.join(str(d) for d in reversed(digits))
        prompt = inp + ' ->'
        try:
            ids = [stoi[c] for c in prompt]
        except KeyError:
            continue
        out  = greedy(model, ids, len(digits) * 2 + 4, device)
        got  = ''.join(itos[i] for i in out).split('\n')[0].strip()
        if ' '.join(got.split()[:len(digits)]) == exp:
            correct += 1
    return correct / n


def evaluate_addition(model, meta, n, device):
    random.seed(SEED + 1)
    stoi, itos = meta['stoi'], meta['itos']
    correct = 0
    for _ in range(n):
        a, b   = random.randint(100, 999), random.randint(100, 999)
        exp    = str(a + b)
        prompt = ' '.join(str(a)) + ' + ' + ' '.join(str(b)) + ' ->'
        try:
            ids = [stoi[c] for c in prompt]
        except KeyError:
            continue
        out = greedy(model, ids, 12, device)
        got = ''.join(''.join(itos[i] for i in out).split('\n')[0].strip().split())
        if got == exp:
            correct += 1
    return correct / n


def run_single(lam_rev, lam_add, device, n):
    rev_meta = load_meta('data/reverse_ft/meta.pkl')
    add_meta = load_meta('data/addition/meta.pkl')

    model   = build_merged_model(lam_rev, lam_add).to(device)
    rev_acc = evaluate_reverse(model,  rev_meta, n, device)
    add_acc = evaluate_addition(model, add_meta, n, device)

    print(f"lambda_rev={lam_rev}  lambda_add={lam_add}")
    print(f"  reverse accuracy  : {rev_acc*100:.2f}%")
    print(f"  addition accuracy : {add_acc*100:.2f}%")


def run_sweep(device, n):
    rev_meta = load_meta('data/reverse_ft/meta.pkl')
    add_meta = load_meta('data/addition/meta.pkl')

    lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # single task: addition only
    print("addition only (lam_rev=0, lam_add varies)")
    print(f"{'lam':>5}  {'add':>8}")
    print("-" * 18)
    for lam in lambdas:
        model   = build_merged_model(0.0, lam).to(device)
        add_acc = evaluate_addition(model, add_meta, n, device)
        print(f"{lam:>5.1f}  {add_acc*100:>7.1f}%")

    print()

    # single task: reverse only
    print("reverse only (lam_add=0, lam_rev varies)")
    print(f"{'lam':>5}  {'rev':>8}")
    print("-" * 18)
    for lam in lambdas:
        model   = build_merged_model(lam, 0.0).to(device)
        rev_acc = evaluate_reverse(model,  rev_meta, n, device)
        print(f"{lam:>5.1f}  {rev_acc*100:>7.1f}%")

    print()

    # both tasks combined
    print("both tasks (lam_rev=lam, lam_add=lam)")
    print(f"{'lam':>5}  {'rev':>8}  {'add':>8}")
    print("-" * 28)
    for lam in lambdas:
        model   = build_merged_model(lam, lam).to(device)
        rev_acc = evaluate_reverse(model,  rev_meta, n, device)
        add_acc = evaluate_addition(model, add_meta, n, device)
        print(f"{lam:>5.1f}  {rev_acc*100:>7.1f}%  {add_acc*100:>7.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lam_rev', type=float, default=1.0)
    parser.add_argument('--lam_add', type=float, default=1.0)
    parser.add_argument('--sweep',   action='store_true')
    parser.add_argument('--n',       type=int,   default=200)
    parser.add_argument('--device',  type=str,   default='mps')
    args = parser.parse_args()

    ckpt = torch.load(ADDITION_FT_CKPT, map_location='cpu')
    if ckpt['iter_num'] == 0:
        print("addition_ft not trained yet. Run:")
        print("  NANOGPT_CONFIG=/Users/sonnguyen/comp560-nanoGPT/configurator.py \\")
        print("  python /Users/sonnguyen/comp560-nanoGPT/train.py config/train_addition_ft.py")
        sys.exit(1)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if args.sweep:
        run_sweep(args.device, args.n)
    else:
        run_single(args.lam_rev, args.lam_add, args.device, args.n)
