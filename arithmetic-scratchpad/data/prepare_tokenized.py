"""
Generate and tokenize arithmetic datasets for nanoGPT.
Based on "How Far Can Transformers Reason? The Globality Barrier and Inductive Scratchpad"
"""
import os
import pickle
import numpy as np
import random
import argparse

def add_with_scratchpad(a, b):
    """Generate addition with scratchpad showing column-by-column work"""
    max_len = max(len(str(a)), len(str(b)))
    str_a = str(a).zfill(max_len)
    str_b = str(b).zfill(max_len)
    
    steps = []
    carry = 0
    result_digits = []
    
    # Process right to left
    for i in range(max_len - 1, -1, -1):
        digit_a = int(str_a[i])
        digit_b = int(str_b[i])
        total = digit_a + digit_b + carry
        digit_result = total % 10
        new_carry = total // 10
        
        if carry > 0:
            steps.append(f"{digit_a}+{digit_b}+{carry}={digit_result}c{new_carry}")
        elif new_carry > 0:
            steps.append(f"{digit_a}+{digit_b}={digit_result}c{new_carry}")
        else:
            steps.append(f"{digit_a}+{digit_b}={digit_result}")
        
        carry = new_carry
        result_digits.append(str(digit_result))
    
    if carry > 0:
        result_digits.append(str(carry))
    
    result = ''.join(reversed(result_digits))
    scratchpad = ','.join(steps)
    return result, scratchpad

def generate_dataset(output_file, num_samples=50000, max_digits=3, use_scratchpad=False):
    samples = set()
    max_num = 10 ** max_digits - 1
    
    print(f"Generating {num_samples} samples (max {max_digits} digits, range: 1-{max_num})")
    print(f"  Scratchpad: {use_scratchpad}")
    
    while len(samples) < num_samples:
        a = random.randint(1, max_num)
        b = random.randint(1, max_num)
        
        if use_scratchpad:
            result, scratchpad = add_with_scratchpad(a, b)
            sample = f"{a}+{b}->{scratchpad}->{result}\n"
        else:
            result = a + b
            sample = f"{a}+{b}->{result}\n"
        
        samples.add(sample)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        samples_list = list(samples)
        random.shuffle(samples_list)
        f.writelines(samples_list)
    
    print(f"  ✓ {len(samples):,} samples -> {output_file}")
    print(f"  Sample: {list(samples)[0].strip()}")


def tokenize_data(input_file, output_dir):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        return
    
    print(f"Tokenizing: {input_file}")
    
    with open(input_file, 'r') as f:
        data = f.read()
    
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"  Vocab size: {vocab_size} chars: {''.join(chars)}")
    
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    
    data_ids = encode(data)
    n = len(data_ids)
    train_ids = np.array(data_ids[:int(n * 0.9)], dtype=np.uint16)
    val_ids = np.array(data_ids[int(n * 0.9):], dtype=np.uint16)
    
    os.makedirs(output_dir, exist_ok=True)
    train_ids.tofile(os.path.join(output_dir, 'train.bin'))
    val_ids.tofile(os.path.join(output_dir, 'val.bin'))
    
    with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump({'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi}, f)
    
    print(f"  ✓ {len(train_ids):,} train / {len(val_ids):,} val tokens -> {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Generate and tokenize arithmetic datasets')
    parser.add_argument('--test', action='store_true', help='Generate test set with longer numbers')
    parser.add_argument('--max_digits', type=int, default=3, help='Maximum number of digits (default: 3)')
    parser.add_argument('--num_samples', type=int, default=50000, help='Number of samples (default: 50000)')
    parser.add_argument('--tokenize-only', action='store_true', help='Skip generation, only tokenize')
    parser.add_argument('--generate-only', action='store_true', help='Skip tokenization, only generate')
    args = parser.parse_args()
    
    # Generate datasets
    if not args.tokenize_only:
        print("=" * 70)
        if args.test:
            print("GENERATING TEST SETS (Out-of-Distribution)")
            print("=" * 70)
            test_digits = args.max_digits if args.max_digits > 3 else 4
            test_samples = args.num_samples // 10
            
            print("\nWithout scratchpad:")
            generate_dataset('without_scratchpad/test.txt', test_samples, test_digits, False)
            
            print("\nWith scratchpad:")
            generate_dataset('with_scratchpad/test.txt', test_samples, test_digits, True)
        else:
            print("GENERATING TRAINING SETS")
            print("=" * 70)
            
            print("\nWithout scratchpad (baseline):")
            generate_dataset('without_scratchpad/train.txt', args.num_samples, args.max_digits, False)
            
            print("\nWith scratchpad:")
            generate_dataset('with_scratchpad/train.txt', args.num_samples, args.max_digits, True)
        
        print("\n✓ Generation complete!")
    
    # Tokenize datasets
    if not args.generate_only:
        print("\n" + "=" * 70)
        print("TOKENIZING FOR NANOGPT")
        print("=" * 70)
        
        file_suffix = 'test.txt' if args.test else 'train.txt'
        datasets = [
            (f'without_scratchpad/{file_suffix}', 'without_scratchpad'),
            (f'with_scratchpad/{file_suffix}', 'with_scratchpad')
        ]
        
        for input_file, output_dir in datasets:
            print()
            tokenize_data(input_file, output_dir)
        
        print("\n✓ Tokenization complete!")
        print("\nOutput files: train.bin, val.bin, meta.pkl")
    
    if not args.test and not args.generate_only:
        print("\nTrain with:")
        print("  NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \\")
        print("    python -u ../../comp560-nanoGPT/train.py config/without_scratchpad.py")

if __name__ == '__main__':
    main()
