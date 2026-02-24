# Evaluate the addition model on the held-out test set.
#
# Usage:
#   python evaluate_addition.py           # eval all of test.txt
#   python evaluate_addition.py --n 500   # random 500 samples
#   python evaluate_addition.py --device cpu
import os, sys, argparse, pickle, torch
import numpy as np

_here   = os.path.abspath(os.path.dirname(__file__))
NANOGPT = os.path.join(_here, '..', '..', 'comp560-nanoGPT')
if not os.path.isdir(NANOGPT):
    NANOGPT = os.path.join(os.path.expanduser('~'), 'comp560-nanoGPT')
sys.path.insert(0, os.path.abspath(NANOGPT))
from model import GPTConfig, GPT


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = GPTConfig(**ckpt['model_args'])
    model = GPT(cfg)
    sd = {k.removeprefix('_orig_mod.'): v for k, v in ckpt['model'].items()}
    model.load_state_dict(sd)
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def greedy_batch(model, all_prompt_ids, max_new, device, batch_size=256):
    pad_id = 0
    results = []
    for start in range(0, len(all_prompt_ids), batch_size):
        batch   = all_prompt_ids[start:start + batch_size]
        max_len = max(len(p) for p in batch)
        padded  = [[pad_id] * (max_len - len(p)) + p for p in batch]
        x = torch.tensor(padded, dtype=torch.long, device=device)
        prompt_lens = [len(p) for p in batch]
        for _ in range(max_new):
            logits, _ = model(x)
            next_ids = logits[:, -1].argmax(dim=-1, keepdim=True)
            x = torch.cat([x, next_ids], dim=1)
        for i, plen in enumerate(prompt_lens):
            results.append(x[i, max_len:].tolist())
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n',      type=int, default=None)
    ap.add_argument('--device', type=str, default='mps')
    args = ap.parse_args()

    base   = os.path.dirname(__file__)
    ckpt   = os.path.join(base, 'out', 'addition', 'ckpt.pt')
    meta_f = os.path.join(base, 'data', 'addition', 'meta.pkl')
    test_f = os.path.join(base, 'data', 'addition', 'test.txt')

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

    pairs     = [line.split(' -> ') for line in lines]
    prompts   = [lhs + ' ->' for lhs, _ in pairs]
    expecteds = [rhs for _, rhs in pairs]
    all_ids   = [encode(p) for p in prompts]
    max_new   = max(len(e) for e in expecteds) + 4

    print(f"Evaluating {len(lines):,} samples...")
    all_generated = greedy_batch(model, all_ids, max_new, args.device)

    correct = 0
    wrong   = []
    # breakdown by carry count (approximated from result digit count)
    by_len  = {}  # result length -> (correct, total)

    for prompt, expected, gen_ids in zip(prompts, expecteds, all_generated):
        out_str    = decode(gen_ids).split('\n')[0].strip()
        pred_digits = ''.join(out_str.split())[:len(''.join(expected.split()))]
        exp_digits  = ''.join(expected.split())
        rlen = len(exp_digits)
        cc, tt = by_len.get(rlen, (0, 0))
        if pred_digits == exp_digits:
            correct += 1
            by_len[rlen] = (cc + 1, tt + 1)
        else:
            by_len[rlen] = (cc, tt + 1)
            if len(wrong) < 10:
                wrong.append((prompt, expected, out_str))

    total = len(lines)
    print(f"\n{'='*55}")
    print(f"  Addition model — test set accuracy")
    print(f"{'='*55}")
    print(f"  Samples   : {total:,}")
    print(f"  Correct   : {correct:,}")
    print(f"  Accuracy  : {correct/total:.2%}")
    print(f"\n  By result length (proxy for carry count):")
    for rlen in sorted(by_len):
        c, t = by_len[rlen]
        label = {3: "no carry   ", 4: "some carry "}.get(rlen, f"{rlen} digits    ")
        print(f"    {label}: {c}/{t}  ({c/t:.1%})")
    print(f"{'='*55}")
    if wrong:
        print(f"\n  First wrong predictions:")
        for prompt, exp, got in wrong:
            print(f"    {prompt}  expected={exp!r}  got={got!r}")


if __name__ == '__main__':
    main()
