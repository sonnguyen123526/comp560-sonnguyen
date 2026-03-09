# Evaluate the reverse model on the held-out test set (5-digit OOD sequences).

import os, sys, argparse, pickle, torch
import numpy as np

_here    = os.path.abspath(os.path.dirname(__file__))
NANOGPT  = os.path.join(_here, '..', '..', 'comp560-nanoGPT')
if not os.path.isdir(NANOGPT):
    NANOGPT = os.path.join(os.path.expanduser('~'), 'comp560-nanoGPT')
sys.path.insert(0, os.path.abspath(NANOGPT))
from model import GPTConfig, GPT


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = GPTConfig(**ckpt['model_args'])
    model = GPT(cfg)
    sd = ckpt['model']
    sd = {k.removeprefix('_orig_mod.'): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    model.to(device)
    return model

@torch.no_grad()
def greedy_batch(model, all_prompt_ids, max_new, device, batch_size=256):
    pad_id = 0
    results = []
    for start in range(0, len(all_prompt_ids), batch_size):
        batch = all_prompt_ids[start:start + batch_size]
        max_len = max(len(p) for p in batch)
        padded = [[pad_id] * (max_len - len(p)) + p for p in batch]
        x = torch.tensor(padded, dtype=torch.long, device=device)
        prompt_lens = [len(p) for p in batch]
        for _ in range(max_new):
            logits, _ = model(x)
            next_ids = logits[:, -1].argmax(dim=-1, keepdim=True)
            x = torch.cat([x, next_ids], dim=1)
        # slice: skip the left-padding and the original prompt, keep generated
        for i, plen in enumerate(prompt_lens):
            pad_len = max_len - plen
            results.append(x[i, max_len:].tolist())
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n',      type=int, default=None,  help='number of test samples (default: all)')
    ap.add_argument('--device', type=str, default='mps', help='cpu | mps | cuda')
    args = ap.parse_args()

    base     = os.path.dirname(__file__)
    ckpt     = os.path.join(base, 'out', 'reverse', 'ckpt.pt')
    meta_f   = os.path.join(base, 'data', 'reverse', 'meta.pkl')
    test_f   = os.path.join(base, 'data', 'reverse', 'test.txt')

    # vocab
    with open(meta_f, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: ''.join(itos[i] for i in ids)

    model = load_model(ckpt, args.device)

    with open(test_f) as f:
        lines = [l.strip() for l in f if l.strip()]
    if args.n:
        import random; random.seed(42)
        lines = random.sample(lines, min(args.n, len(lines)))

    pairs = [line.split(' -> ') for line in lines]
    prompts    = [inp + ' ->' for inp, _ in pairs]
    expecteds  = [exp for _, exp in pairs]
    all_ids    = [encode(p) for p in prompts]

    # longest expected output length across the batch
    max_new = max(len(e) for e in expecteds) + 4

    print(f"  Evaluating {len(lines):,} samples (batch=256)...")
    all_generated = greedy_batch(model, all_ids, max_new, args.device)

    correct = 0
    wrong_examples = []

    for prompt, expected, gen_ids in zip(prompts, expecteds, all_generated):
        out_str   = decode(gen_ids).split('\n')[0].strip()
        pred_digits = ''.join(out_str.split())[:len(''.join(expected.split()))]
        exp_digits  = ''.join(expected.split())
        if pred_digits == exp_digits:
            correct += 1
        elif len(wrong_examples) < 10:
            wrong_examples.append((prompt, expected, out_str))

    total = len(lines)
    acc   = correct / total
    print(f"\n{'='*55}")
    print(f"  Reverse model — test set accuracy")
    print(f"{'='*55}")
    print(f"  Samples evaluated : {total:,}")
    print(f"  Correct           : {correct:,}")
    print(f"  Accuracy          : {acc:.2%}")
    print(f"{'='*55}")
    if wrong_examples:
        print(f"\n  First wrong predictions:")
        for prompt, exp, got in wrong_examples:
            print(f"    prompt={prompt!r}  expected={exp!r}  got={got!r}")

if __name__ == '__main__':
    main()
