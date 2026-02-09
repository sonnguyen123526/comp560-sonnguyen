"""
Generate dataset using cyclic shifts method from the paper.

Example:
  Input: fs$46+ih$98=
  Output: $kckn|0#6fs$4+8ih$9=4$kck|1#46fs$+98ih$=44$kc|1#144$kc<EOS>
"""
import random
import string
import os
import argparse

def random_text(length):
    chars = string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

def shift_right(s):
    """'abc' -> 'cab'"""
    return s[-1] + s[:-1] if s else s

def add_with_shifts(a, b, d_amb=10):
    str_a = str(a).zfill(max(len(str(a)), len(str(b))))
    str_b = str(b).zfill(max(len(str(a)), len(str(b))))
    n = len(str_a)
    
    # Add random prefix + $ marker
    x = random_text(d_amb - n) + '$' + str_a
    y = random_text(d_amb - n) + '$' + str_b
    ans = '$' + random_text(d_amb + 1)
    
    states = [f"{x}+{y}={ans}|0"]
    
    carry = 0
    result_digits = []
    
    for i in range(n):
        digit_a = int(x[-1]) if x[-1].isdigit() else 0
        digit_b = int(y[-1]) if y[-1].isdigit() else 0
        
        total = digit_a + digit_b + carry
        result_digit = total % 10
        carry = total // 10
        
        result_digits.append(str(result_digit))
        ans = str(result_digit) + ans[:-1]
        
        x = shift_right(x)
        y = shift_right(y)
        
        states.append(f"#{x}+{y}={ans}|{carry}")
    
    if carry > 0:
        result_digits.append(str(carry))
    
    final_answer = ''.join(reversed(result_digits))
    states.append(f"#{final_answer}")
    states.append('<EOS>')
    
    return ''.join(states), final_answer

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
        
        full_text, _ = add_with_shifts(a, b, d_amb)
        samples.add(f"{full_text}\n")
    
    samples_list = list(samples)
    random.shuffle(samples_list)
    
    with open(output_file, 'w') as f:
        f.writelines(samples_list)
    
    print(f"  ✓ {len(samples):,} samples -> {output_file}")
    print(f"  Example: {samples_list[0].strip()[:100]}...")

def main():
    parser = argparse.ArgumentParser(description='Generate cyclic shifts dataset')
    parser.add_argument('--test', action='store_true', help='Generate test set')
    parser.add_argument('--max_digits', type=int, default=3, help='Max digits')
    parser.add_argument('--num_samples', type=int, default=50000, help='Number of samples')
    args = parser.parse_args()
    
    os.makedirs('cyclic_shifts', exist_ok=True)
    
    if args.test:
        print("\nGenerating test set (cyclic shifts)...")
        generate_dataset('cyclic_shifts/test.txt',
                        max_digits=args.max_digits,
                        num_samples=args.num_samples // 10)
    else:
        print("\nGenerating training set (cyclic shifts)...")
        generate_dataset('cyclic_shifts/train.txt',
                        max_digits=args.max_digits,
                        num_samples=args.num_samples)

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()
