"""
Generate dataset using random spaces method from the paper.

Example:
  Input: 94_+_3__1=
  Output: <START>[01]4[08]1c0r5$xgwg#[00]9[05]3c1r25$xgw#[-1]_[03]_c0r125$xg<EOS>
"""
import random
import string
import os
import argparse

def random_text(length):
    chars = string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

def add_with_random_spaces(a, b, d_amb=10):
    str_a = str(a).zfill(max(len(str(a)), len(str(b))))
    str_b = str(b).zfill(max(len(str(a)), len(str(b))))
    n = len(str_a)
    
    # Input with random underscores
    input_parts = []
    for char in str_a:
        input_parts.append(char)
        if random.random() < 0.3:
            input_parts.append('_')
    
    input_parts.append('+')
    if random.random() < 0.3:
        input_parts.append('_')
    
    for char in str_b:
        input_parts.append(char)
        if random.random() < 0.3:
            input_parts.append('_')
    
    input_parts.append('=')
    input_str = ''.join(input_parts)
    
    # Answer starts with random text
    ans = ['$'] + list(random_text(d_amb + 1))
    
    states = ['<START>']
    carry = 0
    result_digits = []
    
    # Work right to left
    for i in range(n):
        digit_a = int(str_a[n - 1 - i])
        digit_b = int(str_b[n - 1 - i])
        
        total = digit_a + digit_b + carry
        result_digit = total % 10
        carry = total // 10
        
        result_digits.append(str(result_digit))
        
        # Shift answer accumulator
        ans = [str(result_digit)] + ans[:-1]
        ans_str = ''.join(ans)
        
        ptr_a = f"[{i:02d}]"
        ptr_b = f"[{n + i + 1:02d}]"
        states.append(f"{ptr_a}{digit_a}{ptr_b}{digit_b}c{carry}r{ans_str}")
    
    if carry > 0:
        result_digits.append(str(carry))
        ans = [str(carry)] + ans[:-1]
        states.append(f"[-1]_[-1]_c{carry}r{''.join(ans)}")
    
    states.append('<EOS>')
    
    scratchpad = ''.join(states)
    result = ''.join(reversed(result_digits))
    
    return input_str, scratchpad, result

def generate_dataset(output_file, max_digits=3, num_samples=50000):
    samples = set()
    d_amb = max_digits * 4
    
    print(f"Generating {num_samples} samples (max {max_digits} digits, d_amb={d_amb})")
    
    while len(samples) < num_samples:
        n_digits = random.choices(range(1, max_digits + 1), 
                                   weights=range(1, max_digits + 1))[0]
        max_num = 10 ** n_digits - 1
        
        a = random.randint(1, max_num)
        b = random.randint(1, max_num)
        
        input_str, scratchpad, _ = add_with_random_spaces(a, b, d_amb)
        samples.add(f"{input_str}{scratchpad}\n")
    
    samples_list = list(samples)
    random.shuffle(samples_list)
    
    with open(output_file, 'w') as f:
        f.writelines(samples_list)
    
    print(f"  ✓ {len(samples):,} samples -> {output_file}")
    print(f"  Example: {samples_list[0].strip()[:100]}...")

def main():
    parser = argparse.ArgumentParser(description='Generate random spaces dataset')
    parser.add_argument('--test', action='store_true', help='Generate test set')
    parser.add_argument('--max_digits', type=int, default=3, help='Max digits')
    parser.add_argument('--num_samples', type=int, default=50000, help='Number of samples')
    args = parser.parse_args()
    
    os.makedirs('random_spaces', exist_ok=True)
    
    if args.test:
        print("\nGenerating test set (random spaces)...")
        generate_dataset('random_spaces/test.txt', 
                        max_digits=args.max_digits,
                        num_samples=args.num_samples // 10)
    else:
        print("\nGenerating training set (random spaces)...")
        generate_dataset('random_spaces/train.txt', 
                        max_digits=args.max_digits,
                        num_samples=args.num_samples)

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()
