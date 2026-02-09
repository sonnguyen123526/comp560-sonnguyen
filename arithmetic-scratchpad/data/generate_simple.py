"""
Generate VERY simple arithmetic dataset - single digit addition only
This should be easy enough for the small model to learn
"""
import random

def generate_simple_dataset(output_file, with_scratchpad=False):
    """Generate all possible single-digit additions"""
    samples = []
    
    # All combinations of single digits (0-9)
    for a in range(10):
        for b in range(10):
            result = a + b
            if with_scratchpad:
                # Show the work even for single digits
                if result >= 10:
                    samples.append(f"{a}+{b}->{a}+{b}={result%10}c{result//10}->{result}\n")
                else:
                    samples.append(f"{a}+{b}->{a}+{b}={result}->{result}\n")
            else:
                samples.append(f"{a}+{b}->{result}\n")
    
    # Shuffle
    random.shuffle(samples)
    
    # Repeat to get more training data
    samples = samples * 500  # 100 examples * 500 = 50,000 samples
    
    with open(output_file, 'w') as f:
        f.writelines(samples)
    
    print(f"Generated {len(samples)} samples")
    print(f"Sample: {samples[0].strip()}")

if __name__ == '__main__':
    print("Generating simple single-digit datasets...")
    print("\nWithout scratchpad:")
    generate_simple_dataset('without_scratchpad/train.txt', with_scratchpad=False)
    
    print("\nWith scratchpad:")
    generate_simple_dataset('with_scratchpad/train.txt', with_scratchpad=True)
    
    print("\n✓ Done! Now run: python prepare_tokenized.py")
