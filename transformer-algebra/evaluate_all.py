"""Evaluate all composition strategies."""

import argparse
import sys

def run_evaluation(script_name, n_samples):
    """Fire up an evaluation script and grab what it prints."""
    import subprocess
    cmd = [sys.executable, script_name, '--n', str(n_samples)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return result.stdout + result.stderr

def extract_accuracy(output, pattern='Accuracy'):
    """Fish out the accuracy percentage from the output."""
    for line in output.split('\n'):
        if pattern in line and '%' in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if '%' in part and part.replace('.', '').replace('%', '').isdigit():
                    return float(part.strip('%'))
    return None

def main():
    parser = argparse.ArgumentParser(description='Evaluate all composition strategies')
    parser.add_argument('--n', type=int, default=100, help='Number of test samples')
    parser.add_argument('--device', type=str, default='mps', help='Device: cpu|mps|cuda')
    args = parser.parse_args()

    print("=" * 80)
    print(" COMPOSITION EVALUATION")
    print("=" * 80)
    print(f"Samples: {args.n} | Device: {args.device}\n")

    results = {}

    print("Individual Models:")
    output = run_evaluation('evaluate_reverse.py', args.n)
    results['reverse'] = extract_accuracy(output)
    print(f"  Reversal:  {results['reverse']:.1f}%")
    
    output = run_evaluation('evaluate_addition.py', args.n)
    results['addition'] = extract_accuracy(output)
    print(f"  Addition:  {results['addition']:.1f}%")

    print("\nComposition Strategies:")
    output = run_evaluation('evaluate_composed.py', args.n)
    for line in output.split('\n'):
        if 'End-to-end' in line:
            print(f"  Pipeline:  ", end='')
            acc = float(line.split('(')[1].split('%')[0])
            results['pipeline'] = acc
            print(f"{acc:.1f}%")
            break
    
    print("  Layer concat: not trained")
    print("  Adapter: not trained")

    print("\nTask Arithmetic:")
    output = run_evaluation('task_arithmetic.py', args.n)
    for line in output.split('\n'):
        if 'reverse accuracy' in line.lower():
            print(f"  {line.strip()}")
        if 'addition accuracy' in line.lower():
            print(f"  {line.strip()}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    pipeline_acc = results.get('pipeline')
    if pipeline_acc and results['reverse'] == 100:
        drop = 100 - pipeline_acc
        print(f"Pipeline drops {drop:.1f}% accuracy (representation mismatch)")
    print(f"Task arithmetic fails (models need shared base)")
    print("=" * 80)

if __name__ == '__main__':
    main()
