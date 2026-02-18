# Generate string reversal dataset
# Format: hello->olleh

import random
import string
import os

def generate_random_string(min_len=3, max_len=10):
    # just make a random string of lowercase letters
    length = random.randint(min_len, max_len)
    return ''.join(random.choices(string.ascii_lowercase, k=length))

def generate_reversal_problem(min_len=3, max_len=10):
    s = generate_random_string(min_len, max_len)
    reversed_s = s[::-1]  # python slice trick to reverse
    return f"{s}->{reversed_s}"

def generate_dataset(num_samples=50000, min_len=3, max_len=10, output_file='train.txt'):
    problems = set()
    
    print(f"Generating {num_samples} reversal problems (length {min_len}-{max_len})...")
    
    while len(problems) < num_samples:
        problem = generate_reversal_problem(min_len, max_len)
        problems.add(problem)
        
        if len(problems) % 10000 == 0:
            print(f"  {len(problems)} so far...")
    
    problems = list(problems)
    random.shuffle(problems)
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(problems))
    
    print(f"Done! {len(problems)} problems saved to {output_file}")
    print(f"Total chars: {sum(len(p) for p in problems):,}")
    print(f"\nExamples:")
    for i in range(5):
        print(f"  {problems[i]}")

if __name__ == '__main__':
    os.makedirs('reverse', exist_ok=True)
    
    random.seed(42)
    generate_dataset(
        num_samples=50000,
        min_len=3,
        max_len=10,
        output_file='reverse/train.txt'
    )
    
    print("\nNext: python prepare_tokenized.py reverse")
