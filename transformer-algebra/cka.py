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
MIXED_META_PATH  = 'data/mixed/meta.pkl'
REVERSE_TEST     = 'data/test/reverse_test.txt'
ADDITION_TEST    = 'data/test/addition_test.txt'

N_LAYERS = 4


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd   = {k.removeprefix('_orig_mod.'): v.float()
            for k, v in ckpt['model'].items()}
    model = GPT(GPTConfig(**ckpt['model_args']))
    model.load_state_dict(sd)
    model.eval()
    return model


def build_prompts(n, stoi):
    # load n//2 prompts from each test file; keep only the input side (up to '->')
    prompts = []
    for filepath, stop_at in [(REVERSE_TEST, n // 2), (ADDITION_TEST, n)]:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                prompt = line.split('->')[0] + '->'
                if all(c in stoi for c in prompt):
                    prompts.append(prompt)
                if len(prompts) >= stop_at:
                    break
    return prompts


def collect_hidden_states(model, prompts, stoi, device):
    model = model.to(device)
    hiddens = [[] for _ in range(N_LAYERS)]
    hooks   = []

    for i in range(N_LAYERS):
        def make_hook(idx):
            def hook(module, input, output):
                hiddens[idx].append(output.squeeze(0).detach().cpu())
            return hook
        hooks.append(model.transformer.h[i].register_forward_hook(make_hook(i)))

    with torch.no_grad():
        for prompt in prompts:
            ids = [stoi[c] for c in prompt]
            x   = torch.tensor([ids], dtype=torch.long, device=device)
            model(x[:, :model.config.block_size])

    for h in hooks:
        h.remove()
    model.cpu()

    return [torch.cat(hiddens[i], dim=0) for i in range(N_LAYERS)]


def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    num   = torch.norm(Y.T @ X) ** 2
    denom = torch.norm(X.T @ X) * torch.norm(Y.T @ Y)
    if denom < 1e-10:
        return float('nan')
    return (num / denom).item()


def run(model_base, model_rev, model_add, stoi, device, n_prompts):
    prompts = build_prompts(n_prompts, stoi)
    n_rev   = sum(1 for p in prompts if '+' not in p)
    n_add   = len(prompts) - n_rev
    print(f'  Using {len(prompts)} prompts  ({n_rev} reverse + {n_add} addition)')

    print('  Collecting hidden states — base ...')
    h_base = collect_hidden_states(model_base, prompts, stoi, device)
    print('  Collecting hidden states — reverse_ft ...')
    h_rev  = collect_hidden_states(model_rev,  prompts, stoi, device)
    print('  Collecting hidden states — addition_ft ...')
    h_add  = collect_hidden_states(model_add,  prompts, stoi, device)

    print()
    print('=' * 64)
    print('CKA Representation Similarity  (higher = more similar)')
    print('=' * 64)
    print(f"  {'Layer':<8}  {'CKA(base,rev)':>14}  {'CKA(base,add)':>14}  {'CKA(rev,add)':>14}")
    print('-' * 64)

    for i in range(N_LAYERS):
        c_br = linear_cka(h_base[i], h_rev[i])
        c_ba = linear_cka(h_base[i], h_add[i])
        c_ra = linear_cka(h_rev[i],  h_add[i])
        print(f"  Layer {i:<2}  {c_br:>14.4f}  {c_ba:>14.4f}  {c_ra:>14.4f}")

    print()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_prompts', type=int, default=200)
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'Device: {device}')

    print('Loading models ...')
    model_base = load_model(BASE_CKPT)
    model_rev  = load_model(REVERSE_FT_CKPT)
    model_add  = load_model(ADDITION_FT_CKPT)

    with open(MIXED_META_PATH, 'rb') as f:
        stoi = pickle.load(f)['stoi']

    run(model_base, model_rev, model_add, stoi, device, args.n_prompts)


if __name__ == '__main__':
    main()
