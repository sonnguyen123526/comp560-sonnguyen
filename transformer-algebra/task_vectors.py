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


class TaskVector:
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.

        This can either be done by passing two checkpoint paths (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passing in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                # modified: nanoGPT stores {'model': ..., 'model_args': ...}
                # so we parse the dict and strip torch.compile / DDP prefixes
                pretrained_state_dict = self._load_sd(pretrained_checkpoint)
                finetuned_state_dict  = self._load_sd(finetuned_checkpoint)
                self.vector = {}
                for key in pretrained_state_dict:
                    # original: skip non-float params (int64, uint8)
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    # modified: cast to float32 — required for MPS (Apple Silicon)
                    self.vector[key] = (
                        finetuned_state_dict[key].float() -
                        pretrained_state_dict[key].float()
                    )

    def _load_sd(self, path):
        # modified: parse nanoGPT checkpoint dict; strip _orig_mod. and module. prefixes
        ckpt = torch.load(path, map_location='cpu')
        sd   = {}
        for k, v in ckpt['model'].items():
            key = k.replace('_orig_mod.', '').replace('module.', '')
            sd[key] = v
        return sd

    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        # original paper had this indented inside __add__ (bug) — fixed here
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def __sub__(self, other):
        # not in original paper — added for convenience
        return self.__add__(-other)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0, device='cpu',
                 _cache={}):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            # modified: cache checkpoint on first load to avoid repeated disk reads
            if pretrained_checkpoint not in _cache:
                ckpt = torch.load(pretrained_checkpoint, map_location='cpu')
                sd   = {}
                for k, v in ckpt['model'].items():
                    key = k.replace('_orig_mod.', '').replace('module.', '')
                    sd[key] = v.float()
                _cache[pretrained_checkpoint] = (ckpt['model_args'], sd)

            model_args, base_sd = _cache[pretrained_checkpoint]

            new_state_dict = {}
            for key in base_sd:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    new_state_dict[key] = base_sd[key]
                    continue
                new_state_dict[key] = base_sd[key] + scaling_coef * self.vector[key]

            # modified: construct GPT from saved model_args, then load weights
            # original: pretrained_model.load_state_dict(new_state_dict, strict=False)
            cfg   = GPTConfig(**model_args)
            model = GPT(cfg)
            model.load_state_dict(new_state_dict)
            model.eval()
            model.to(device)
        return model


def load_meta(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


@torch.no_grad()
def greedy_batch(model, prompts_ids, max_new, device):
    pad_id = 0
    bs     = model.config.block_size
    mlen   = max(len(p) for p in prompts_ids)
    padded = [[pad_id] * (mlen - len(p)) + p for p in prompts_ids]
    x = torch.tensor(padded, dtype=torch.long, device=device)
    for _ in range(max_new):
        logits, _ = model(x[:, -bs:])
        x = torch.cat([x, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    return [x[i, mlen:].tolist() for i in range(len(prompts_ids))]


def evaluate_reverse(model, meta, n, device):
    random.seed(SEED)
    stoi, itos = meta['stoi'], meta['itos']
    samples = [[random.randint(0, 9) for _ in range(random.randint(2, 4))] for _ in range(n)]
    ids     = [[stoi[c] for c in (' '.join(str(d) for d in s) + ' ->') if c in stoi] for s in samples]
    outs    = greedy_batch(model, ids, 12, device)
    correct = 0
    for digits, gen in zip(samples, outs):
        exp = ' '.join(str(d) for d in reversed(digits))
        got = ''.join(itos[i] for i in gen).split('\n')[0].strip()
        if ' '.join(got.split()[:len(digits)]) == exp:
            correct += 1
    return correct / n


def evaluate_addition(model, meta, n, device):
    random.seed(SEED + 1)
    stoi, itos = meta['stoi'], meta['itos']
    pairs   = [(random.randint(100, 999), random.randint(100, 999)) for _ in range(n)]
    prompts = [' '.join(str(a)) + ' + ' + ' '.join(str(b)) + ' ->' for a, b in pairs]
    ids     = [[stoi[c] for c in p if c in stoi] for p in prompts]
    outs    = greedy_batch(model, ids, 12, device)
    correct = sum(
        1 for (a, b), gen in zip(pairs, outs)
        if ''.join(itos[i] for i in gen).split('\n')[0].replace(' ', '') == str(a + b)
    )
    return correct / n


def run_sweep(lambdas, n, device):
    rev_meta = load_meta('data/reverse_ft/meta.pkl')
    add_meta = load_meta('data/addition/meta.pkl')

    tau_rev  = TaskVector(BASE_CKPT, REVERSE_FT_CKPT)
    tau_add  = TaskVector(BASE_CKPT, ADDITION_FT_CKPT)
    combined = tau_rev + tau_add

    print("both tasks combined (scaling_coef varies)")
    print(f"\n{'lambda':>8}  {'rev':>8}  {'add':>8}")
    print("-" * 30)
    for lam in lambdas:
        model   = combined.apply_to(BASE_CKPT, scaling_coef=lam, device=device)
        rev_acc = evaluate_reverse(model,  rev_meta, n, device)
        add_acc = evaluate_addition(model, add_meta, n, device)
        print(f"{lam:>8.2f}  {rev_acc*100:>7.1f}%  {add_acc*100:>7.1f}%")


def run_negation(n, device):
    rev_meta = load_meta('data/reverse_ft/meta.pkl')
    add_meta = load_meta('data/addition/meta.pkl')

    tau_add = TaskVector(BASE_CKPT, ADDITION_FT_CKPT)

    # θ_result = θ_rev_ft - λ·τ_add
    # at λ=0: pure reverse model; at λ>0: addition vector subtracted from it
    print("negation: apply -τ_add to reverse_ft model (θ_rev_ft - λ·τ_add)")
    print(f"\n{'lambda':>8}  {'rev':>8}  {'add':>8}")
    print("-" * 30)

    neg_add = -tau_add
    for lam in [0.0, 0.5, 1.0, 1.5, 2.0]:
        model   = neg_add.apply_to(REVERSE_FT_CKPT, scaling_coef=lam, device=device)
        rev_acc = evaluate_reverse(model,  rev_meta, n, device)
        add_acc = evaluate_addition(model, add_meta, n, device)
        print(f"{lam:>8.2f}  {rev_acc*100:>7.1f}%  {add_acc*100:>7.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',   choices=['sweep', 'negate'], default='sweep')
    parser.add_argument('--n',      type=int,  default=200)
    parser.add_argument('--device', type=str,  default='mps')
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    for path in [BASE_CKPT, REVERSE_FT_CKPT, ADDITION_FT_CKPT]:
        if not os.path.exists(path):
            print(f"Missing checkpoint: {path}")
            sys.exit(1)

    lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    if args.mode == 'sweep':
        run_sweep(lambdas, args.n, args.device)
    elif args.mode == 'negate':
        run_negation(args.n, args.device)
