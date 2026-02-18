# Generate addition problems for training
# Simple format: 123+456=579

import random
import os

def generate_addition_problem(min_digits=1, max_digits=3):
    # pick random number of digits for each number
    digits1 = random.randint(min_digits, max_digits)
    digits2 = random.randint(min_digits, max_digits)
    
    # make the actual numbers
    num1 = random.randint(10**(digits1-1), 10**digits1 - 1)
    num2 = random.randint(10**(digits2-1), 10**digits2 - 1)
    
    result = num1 + num2
    
    return f"{num1}+{num2}={result}"

def generate_dataset(num_samples=50000, min_digits=1, max_digits=3, output_file='train.txt'):
    problems = set()  # using set to avoid duplicates
    
    print(f"Generating {num_samples} addition problems ({min_digits}-{max_digits} digits)...")
    
    while len(problems) < num_samples:
        problem = generate_addition_problem(min_digits, max_digits)
        problems.add(problem)
        
        if len(problems) % 10000 == 0:
            print(f"  Generated {len(problems)}...")
    
    problems = list(problems)
    random.shuffle(problems)
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(problems))
    
    print(f"Done! Wrote {len(problems)} problems to {output_file}")
    print(f"Total chars: {sum(len(p) for p in problems):,}")
    print(f"\nSample:")
    for i in range(5):
        print(f"  {problems[i]}")

if __name__ == '__main__':
    os.makedirs('addition', exist_ok=True)
    
    random.seed(42)
    generate_dataset(
        num_samples=50000,
        min_digits=1,
        max_digits=3,
        output_file='addition/train.txt'
    )
    
    print("\nNext: python prepare_tokenized.py addition")
