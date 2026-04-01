import argparse
import csv
import os
import random
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from ties import (
    ADDITION_FT_CKPT,
    BASE_CKPT,
    MIXED_META_PATH,
    REVERSE_FT_CKPT,
    build_naive_model,
    build_ties_model,
    classify_composite_failure,
    greedy,
    load_meta,
    load_model_from_ckpt,
)

COMPOSITE_TEST_FILE = 'data/test/stratified_composite_test.txt'


def resolve_device(preferred):
    if preferred != 'auto':
        return preferred
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def safe_percent(correct, total):
    return (100.0 * correct / total) if total > 0 else 0.0


def generate_stratified_data(path, samples_per_level=50, seed=42):
    """Generate a stratified composite dataset with 3 difficulty levels."""
    random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    samples = []

    # Level 1: 1-digit reverse + 1-digit add (reversal is identity).
    for _ in range(samples_per_level):
        d1 = random.randint(1, 9)
        addend = random.randint(1, 9)
        ans = d1 + addend
        samples.append(f'L1 | {d1} + {addend} -> {ans}')

    # Level 2: 2-digit reverse + 2-digit add (in-distribution).
    for _ in range(samples_per_level):
        d1, d2 = random.randint(1, 9), random.randint(1, 9)
        rev_val = int(f'{d2}{d1}')
        addend = random.randint(10, 50)
        ans = rev_val + addend
        samples.append(f'L2 | {d1} {d2} + {addend} -> {ans}')

    # Level 3: 3-digit reverse + 2-digit add (length out-of-distribution).
    for _ in range(samples_per_level):
        d1, d2, d3 = random.randint(1, 9), random.randint(1, 9), random.randint(1, 9)
        rev_val = int(f'{d3}{d2}{d1}')
        addend = random.randint(10, 50)
        ans = rev_val + addend
        samples.append(f'L3 | {d1} {d2} {d3} + {addend} -> {ans}')

    with open(path, 'w') as f:
        for line in samples:
            f.write(line + '\n')

    print(f'Generated {len(samples)} stratified samples -> {path} (seed={seed})')


def evaluate_stratified(model, meta, device, filepath):
    """Evaluate model by level and return accuracy + failure-category stats."""
    stoi, itos = meta['stoi'], meta['itos']

    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    results = {
        'L1': {'correct': 0, 'total': 0},
        'L2': {'correct': 0, 'total': 0},
        'L3': {'correct': 0, 'total': 0},
    }
    failure_counts = {
        'reversed_only': 0,
        'add_raw_input': 0,
        'numeric_wrong': 0,
        'empty_output': 0,
        'mixed_symbols': 0,
        'noise_or_format': 0,
    }

    for line in lines:
        level, equation = line.split(' | ')
        lhs, rhs = equation.rsplit(' -> ', 1)
        prompt = lhs + ' ->'  # keep prompt style consistent with ties.py evaluators
        expected = ''.join(rhs.strip().split())

        try:
            ids = [stoi[c] for c in prompt]
        except KeyError:
            continue

        max_new = max(4, len(expected) + 2)
        out = greedy(model, ids, max_new, device)
        got = ''.join(''.join(itos[i] for i in out).split('\n')[0].strip().split())
        pred = got[:len(expected)]

        info = {
            'expected': int(expected),
            'reversed_val': int(''.join(lhs.split(' + ')[0].split()[::-1])),
            'raw_val': int(''.join(lhs.split(' + ')[0].split())),
            'addend': int(lhs.split(' + ')[1]),
        }

        results[level]['total'] += 1
        if pred == expected:
            results[level]['correct'] += 1
        else:
            label = classify_composite_failure(info, pred)
            if label in failure_counts:
                failure_counts[label] += 1

    return results, failure_counts


def run_experiment(device='auto', samples_per_level=50, seed=42, csv_out=None):
    device = resolve_device(device)

    generate_stratified_data(COMPOSITE_TEST_FILE, samples_per_level=samples_per_level, seed=seed)
    meta = load_meta(MIXED_META_PATH)

    print('\nLoading models...')
    models = {
        'Base Pre-trained': load_model_from_ckpt(BASE_CKPT),
        'Reverse FT Only': load_model_from_ckpt(REVERSE_FT_CKPT),
        'Addition FT Only': load_model_from_ckpt(ADDITION_FT_CKPT),
        'Naive (l=1.0)': build_naive_model(1.0, 1.0),
        'TIES (d=1.0, l=0.2)': build_ties_model(density=1.0, lam=0.2),
    }

    print('\n' + '=' * 86)
    print('FAITH AND FATE: SEQUENTIAL COMPOSITION (U o T) BENCHMARK')
    print('=' * 86)
    print(f'Device: {device}')
    print(f"{'Model Name':<25} | {'Level 1':<10} | {'Level 2':<10} | {'Level 3':<10} | {'Overall':<10}")
    print('-' * 86)

    csv_rows = []

    for name, model in models.items():
        model = model.to(device)
        res, failures = evaluate_stratified(model, meta, device, COMPOSITE_TEST_FILE)

        l1_acc = safe_percent(res['L1']['correct'], res['L1']['total'])
        l2_acc = safe_percent(res['L2']['correct'], res['L2']['total'])
        l3_acc = safe_percent(res['L3']['correct'], res['L3']['total'])

        total_correct = res['L1']['correct'] + res['L2']['correct'] + res['L3']['correct']
        total_seen = res['L1']['total'] + res['L2']['total'] + res['L3']['total']
        overall_acc = safe_percent(total_correct, total_seen)

        print(f"{name:<25} | {l1_acc:>8.1f}% | {l2_acc:>8.1f}% | {l3_acc:>8.1f}% | {overall_acc:>8.1f}%")

        if name == 'TIES (d=1.0, l=0.2)':
            print('  TIES failure categories:', failures)

        csv_rows.append({
            'model': name,
            'level1_acc': l1_acc,
            'level2_acc': l2_acc,
            'level3_acc': l3_acc,
            'overall_acc': overall_acc,
        })

        del model
        if device == 'mps':
            torch.mps.empty_cache()
        elif device == 'cuda':
            torch.cuda.empty_cache()

    print('=' * 86)
    print('\nExperiment complete. Near-zero L2/L3 on TIES supports the interface problem claim.')

    if csv_out:
        with open(csv_out, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['model', 'level1_acc', 'level2_acc', 'level3_acc', 'overall_acc'],
            )
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)
        print(f'Wrote stratified benchmark CSV -> {csv_out}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'mps', 'cuda', 'cpu'])
    parser.add_argument('--samples_per_level', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--csv_out', type=str, default='stratified_composition_results.csv')
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_experiment(
        device=args.device,
        samples_per_level=args.samples_per_level,
        seed=args.seed,
        csv_out=args.csv_out,
    )


if __name__ == '__main__':
    main()
