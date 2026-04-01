import argparse
import csv
import os
import pickle
import random
import re
import statistics
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
COMPOSITE_TEST_FILE = 'data/test/composite_test.txt'
COMPOSITE_CSV_FILE = 'composite_benchmark.csv'


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


def load_model_from_ckpt(ckpt_path):
    ckpt, state_dict = load_params(ckpt_path)
    model = GPT(GPTConfig(**ckpt['model_args']))
    model.load_state_dict(state_dict)
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


def generate_composite_test(path, n, seed):
    """Generate composite prompts: 'd1 d2 + addend -> answer'.

    The target requires reverse-then-add behavior:
      d1 d2 + k -> int(f"{d2}{d1}") + k
    """
    random.seed(seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    samples = []
    for i in range(n):
        d1 = random.randint(1, 9)
        d2 = random.randint(1, 9)
        rev_val = int(f'{d2}{d1}')

        # Light difficulty stratification keeps both easy and harder additions.
        if i % 2 == 0:
            addend = random.randint(10, 39)
        else:
            addend = random.randint(40, 99)

        ans = rev_val + addend
        samples.append(f'{d1} {d2} + {addend} -> {ans}')

    with open(path, 'w') as f:
        for line in samples:
            f.write(line + '\n')

    print(f'Generated {len(samples)} composite samples -> {path} (seed={seed})')


def evaluate_composite(model, meta, n, device, composite_file):
    """Evaluate reverse-then-add composition on composite prompts."""
    stoi, itos = meta['stoi'], meta['itos']
    examples = load_test_examples(composite_file, n)
    correct = 0
    used = 0

    for ex in examples:
        lhs, rhs = ex.rsplit(' -> ', 1)
        prompt = lhs + ' ->'
        exp = ''.join(rhs.strip().split())

        try:
            ids = [stoi[c] for c in prompt]
        except KeyError:
            continue

        # Generate slightly more than expected to avoid premature truncation.
        max_new = max(4, len(exp) + 2)
        out = greedy(model, ids, max_new, device)
        got = ''.join(''.join(itos[i] for i in out).split('\n')[0].strip().split())
        used += 1

        if got[:len(exp)] == exp:
            correct += 1

    return correct / used if used > 0 else 0.0


def parse_composite_example(example):
    """Parse a composite example line like '1 2 + 10 -> 31'."""
    lhs, rhs = example.rsplit(' -> ', 1)
    left_digits_str, addend_str = lhs.split(' + ')
    digits = left_digits_str.split()
    d1, d2 = int(digits[0]), int(digits[1])
    addend = int(addend_str)
    expected = int(rhs.strip())
    reversed_val = int(f'{d2}{d1}')
    raw_val = int(f'{d1}{d2}')
    return {
        'lhs': lhs,
        'expected': expected,
        'd1': d1,
        'd2': d2,
        'addend': addend,
        'reversed_val': reversed_val,
        'raw_val': raw_val,
    }


def classify_composite_failure(example_info, predicted_text):
    """Classify composite failure mode for quick qualitative analysis."""
    exp = str(example_info['expected'])
    rev_only = str(example_info['reversed_val'])
    add_raw = str(example_info['raw_val'] + example_info['addend'])

    if predicted_text == exp:
        return 'correct'
    if predicted_text == rev_only:
        return 'reversed_only'
    if predicted_text == add_raw:
        return 'add_raw_input'
    if predicted_text.isdigit() and len(predicted_text) > 0:
        return 'numeric_wrong'
    if len(predicted_text) == 0:
        return 'empty_output'
    if re.search(r'\d', predicted_text):
        return 'mixed_symbols'
    return 'noise_or_format'


def analyze_composite_failures(
    model,
    meta,
    n,
    device,
    composite_file,
    sample_limit=20,
    csv_path=None,
):
    """Collect failure categories and sample rows for qualitative analysis."""
    stoi, itos = meta['stoi'], meta['itos']
    examples = load_test_examples(composite_file, n)

    counts = {
        'correct': 0,
        'reversed_only': 0,
        'add_raw_input': 0,
        'numeric_wrong': 0,
        'empty_output': 0,
        'mixed_symbols': 0,
        'noise_or_format': 0,
    }
    samples = []

    for ex in examples:
        info = parse_composite_example(ex)
        prompt = info['lhs'] + ' ->'
        expected = str(info['expected'])

        try:
            ids = [stoi[c] for c in prompt]
        except KeyError:
            continue

        max_new = max(4, len(expected) + 2)
        out = greedy(model, ids, max_new, device)
        got = ''.join(''.join(itos[i] for i in out).split('\n')[0].strip().split())
        pred = got[:len(expected)]

        label = classify_composite_failure(info, pred)
        counts[label] = counts.get(label, 0) + 1

        if label != 'correct' and len(samples) < sample_limit:
            samples.append({
                'prompt': prompt,
                'expected': expected,
                'predicted': pred,
                'category': label,
            })

    if csv_path:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['prompt', 'expected', 'predicted', 'category'],
            )
            writer.writeheader()
            for row in samples:
                writer.writerow(row)

    return counts, samples


def run_composite_benchmark(device, n, composite_file, density, lam, csv_path=None):
    meta = load_meta(MIXED_META_PATH)

    models = [
        ('base', load_model_from_ckpt(BASE_CKPT)),
        ('reverse_ft', load_model_from_ckpt(REVERSE_FT_CKPT)),
        ('addition_ft', load_model_from_ckpt(ADDITION_FT_CKPT)),
        ('naive(l=1.0)', build_naive_model(1.0, 1.0)),
        ('naive(l=0.2)', build_naive_model(0.2, 0.2)),
        ('naive(l=0.4)', build_naive_model(0.4, 0.4)),
        (f'ties(d={density:.1f},l={lam:.1f})', build_ties_model(density, lam)),
    ]

    print()
    print('=' * 72)
    print('Composite (U o T) Benchmark')
    print('=' * 72)
    print(f"{'model':<24} {'composite_acc':>14}")
    print('-' * 72)

    rows = []
    for name, model in models:
        acc = evaluate_composite(model.to(device), meta, n, device, composite_file)
        print(f'{name:<24} {acc*100:>12.1f}%')
        rows.append({'model': name, 'composite_acc': acc})

    if csv_path:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['model', 'composite_acc'])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f'Wrote composite benchmark CSV -> {csv_path}')

    print()


def run_composite_multiseed(
    device,
    n,
    density,
    lam,
    seed_start,
    num_seeds,
    composite_file,
    csv_path=None,
):
    """Run composite benchmark over multiple seeds and report mean/std per model."""
    model_builders = [
        ('base', lambda: load_model_from_ckpt(BASE_CKPT)),
        ('reverse_ft', lambda: load_model_from_ckpt(REVERSE_FT_CKPT)),
        ('addition_ft', lambda: load_model_from_ckpt(ADDITION_FT_CKPT)),
        ('naive(l=1.0)', lambda: build_naive_model(1.0, 1.0)),
        ('naive(l=0.2)', lambda: build_naive_model(0.2, 0.2)),
        ('naive(l=0.4)', lambda: build_naive_model(0.4, 0.4)),
        (f'ties(d={density:.1f},l={lam:.1f})', lambda: build_ties_model(density, lam)),
    ]

    seeds = [seed_start + i for i in range(num_seeds)]
    all_rows = []
    per_model = {name: [] for name, _ in model_builders}

    print()
    print('=' * 84)
    print('Composite (U o T) Multi-Seed Benchmark')
    print('=' * 84)
    print(f'Seeds: {seeds}')

    for seed in seeds:
        generate_composite_test(composite_file, n, seed)
        meta = load_meta(MIXED_META_PATH)
        print('-' * 84)
        print(f'Seed {seed}')

        for name, builder in model_builders:
            model = builder().to(device)
            acc = evaluate_composite(model, meta, n, device, composite_file)
            per_model[name].append(acc)
            all_rows.append({'seed': seed, 'model': name, 'composite_acc': acc})
            print(f'  {name:<24} {acc*100:>7.1f}%')

    print('-' * 84)
    print(f"{'model':<24} {'mean':>10} {'std':>10}")
    print('-' * 84)

    summary_rows = []
    for name, vals in per_model.items():
        mean_acc = statistics.mean(vals) if vals else 0.0
        std_acc = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        summary_rows.append({'model': name, 'mean': mean_acc, 'std': std_acc})
        print(f'{name:<24} {mean_acc*100:>9.2f}% {std_acc*100:>9.2f}%')

    if csv_path:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['seed', 'model', 'composite_acc', 'mean', 'std'],
            )
            writer.writeheader()
            for row in all_rows:
                writer.writerow({
                    'seed': row['seed'],
                    'model': row['model'],
                    'composite_acc': row['composite_acc'],
                    'mean': '',
                    'std': '',
                })
            for row in summary_rows:
                writer.writerow({
                    'seed': 'summary',
                    'model': row['model'],
                    'composite_acc': '',
                    'mean': row['mean'],
                    'std': row['std'],
                })
        print(f'Wrote multi-seed composite CSV -> {csv_path}')

    print()


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
    parser.add_argument('--composite_eval', action='store_true')
    parser.add_argument('--composite_multiseed', action='store_true')
    parser.add_argument('--composite_failure_analysis', action='store_true')
    parser.add_argument('--composite_generate', action='store_true')
    parser.add_argument('--composite_file', type=str, default=COMPOSITE_TEST_FILE)
    parser.add_argument('--composite_csv', type=str, default=COMPOSITE_CSV_FILE)
    parser.add_argument('--failure_csv', type=str, default='composite_failure_samples.csv')
    parser.add_argument('--failure_samples', type=int, default=20)
    parser.add_argument('--seed_start', type=int, default=42)
    parser.add_argument('--num_seeds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n',       type=int,   default=200)
    parser.add_argument('--device',  type=str,   default='mps')
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if args.composite_generate:
        generate_composite_test(args.composite_file, args.n, args.seed)

    if args.composite_multiseed:
        run_composite_multiseed(
            args.device,
            args.n,
            args.density,
            args.lam,
            args.seed_start,
            args.num_seeds,
            args.composite_file,
            csv_path=args.composite_csv,
        )
    elif args.composite_failure_analysis:
        if not os.path.exists(args.composite_file):
            print(f'Composite file not found at {args.composite_file}; generating it now.')
            generate_composite_test(args.composite_file, args.n, args.seed)
        meta = load_meta(MIXED_META_PATH)
        model = build_ties_model(args.density, args.lam).to(args.device)
        counts, samples = analyze_composite_failures(
            model,
            meta,
            args.n,
            args.device,
            args.composite_file,
            sample_limit=args.failure_samples,
            csv_path=args.failure_csv,
        )
        print()
        print('=' * 72)
        print('Composite Failure Analysis (TIES model)')
        print('=' * 72)
        for k, v in counts.items():
            print(f'{k:<20} {v:>6}')
        print(f'Saved {len(samples)} failure samples -> {args.failure_csv}')
        print()
    elif args.composite_eval:
        if not os.path.exists(args.composite_file):
            print(f'Composite file not found at {args.composite_file}; generating it now.')
            generate_composite_test(args.composite_file, args.n, args.seed)
        run_composite_benchmark(
            args.device,
            args.n,
            args.composite_file,
            args.density,
            args.lam,
            csv_path=args.composite_csv,
        )
    elif args.sweep:
        run_sweep(args.device, args.n)
    else:
        run_single(args.density, args.lam, args.device, args.n)


if __name__ == '__main__':
    main()
