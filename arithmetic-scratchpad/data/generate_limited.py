"""
Generate LIMITED 2-digit arithmetic (10-50 only)
Bridge between single-digit and full 2-digit
"""
import random

def add_with_scratchpad(a, b):
    """Generate addition with scratchpad"""
    str_a = str(a)
    str_b = str(b)
    max_len = max(len(str_a), len(str_b))
    str_a = str_a.zfill(max_len)
    str_b = str_b.zfill(max_len)
    
    steps = []
    carry = 0
    result_digits = []
    
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

def generate_limited_dataset(output_file, with_scratchpad=False):
    """Generate 2-digit addition but only 10-50 range"""
    samples = set()
    
    # Generate all combinations of 10-50
    for a in range(10, 51):
        for b in range(10, 51):
            if with_scratchpad:
                result, scratchpad = add_with_scratchpad(a, b)
                samples.add(f"{a}+{b}->{scratchpad}->{result}\n")
            else:
                result = a + b
                samples.add(f"{a}+{b}->{result}\n")
    
    # Shuffle
    samples_list = list(samples)
    random.shuffle(samples_list)
    
    # Repeat to get more training data
    samples_list = samples_list * 30  # ~1600 unique * 30 = 48K samples
    
    with open(output_file, 'w') as f:
        f.writelines(samples_list)
    
    print(f"Generated {len(samples_list)} samples")
    print(f"Range: 10-50 (limited 2-digit)")
    print(f"Sample: {samples_list[0].strip()}")

if __name__ == '__main__':
    print("Generating LIMITED 2-digit datasets (10-50 range)...")
    print("\nWithout scratchpad:")
    generate_limited_dataset('without_scratchpad/train.txt', with_scratchpad=False)
    
    print("\nWith scratchpad:")
    generate_limited_dataset('with_scratchpad/train.txt', with_scratchpad=True)
    
    print("\n✓ Done! Now run: python prepare_tokenized.py")
