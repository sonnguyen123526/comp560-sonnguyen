"""
Generate and tokenize datasets for transformer composition experiments.

This script creates three datasets to test if we can compose transformers:
1. Reverse: Teach a model to reverse digit sequences (12345->54321)
2. Addition: Teach a model to add two numbers (123+456->579)  
3. Composed: The combined task - reverse THEN add (123+456->321+654->975)

The big question: Can we train models A and B separately, then combine them
to solve A∘B without training a whole new model from scratch?

Think of it like teaching one person to reverse numbers, another to add,
then seeing if they can work together to reverse-then-add!
"""
import os
import pickle
import numpy as np
import random
import argparse

def generate_reverse_dataset(output_file, num_samples=10000, max_len=4):
    """
    Generate digit reversal dataset with spaced format.
    
    Trains on 2-4 digit sequences. The 5-digit sequences are held out
    as a LENGTH GENERALIZATION test — can the model apply the rule to
    sequences longer than it was trained on?
    
    Example:
        Input:  1 2 3 4
        Output: 4 3 2 1
        Format: 1 2 3 4 -> 4 3 2 1
    """
    samples = set()  # Use set to avoid duplicates
    
    print(f"Generating reverse dataset (train: 2-4 digits, test: 5 digits)...")
    print(f"  • {num_samples:,} samples")
    print(f"  • Numbers 2–{max_len} digits long (5-digit held out for test)")
    print(f"  • Using spaced format for better learning")
    
    while len(samples) < num_samples:
        # Generate a random number with varying length (2 to max_len, default 4)
        length = random.randint(2, max_len)
        num = ''.join([str(random.randint(0, 9)) for _ in range(length)])
        
        # Reverse it (this is what the model will learn)
        reversed_num = num[::-1]
        
        # Add spaces between digits to make the pattern more explicit
        num_spaced = ' '.join(num)
        reversed_spaced = ' '.join(reversed_num)
        
        sample = f"{num_spaced} -> {reversed_spaced}\n"
        samples.add(sample)
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        samples_list = list(samples)
        random.shuffle(samples_list)
        f.writelines(samples_list)
    
    print(f"  ✓ Saved {len(samples):,} samples to {output_file}")
    print(f"  Example: {list(samples)[0].strip()}")

def generate_addition_dataset(output_file, num_samples=10000, max_digits=3):
    """
    Generate addition dataset with spaced format.
    
    Spacing helps the model learn digit-by-digit operations,
    making it easier to understand carry operations and alignment.
    
    Example:
        Input:  1 2 3 + 4 5 6
        Output: 5 7 9
        Format: 1 2 3 + 4 5 6 -> 5 7 9
    
    This is Task B - the second step in our composition.
    The model learns basic arithmetic, with clear digit boundaries!
    """
    samples = set()
    max_num = 10 ** max_digits - 1
    min_num = 10 ** (max_digits - 1)
    
    print(f"Generating addition dataset...")
    print(f"  • {num_samples:,} samples")
    print(f"  • Numbers with {max_digits} digits ({min_num} to {max_num})")
    print(f"  • Using spaced format for better learning")
    
    while len(samples) < num_samples:
        a = random.randint(min_num, max_num)
        b = random.randint(min_num, max_num)
        result = a + b
        
        # Add spaces between digits for clarity
        # Example: "123+456->579" becomes "1 2 3 + 4 5 6 -> 5 7 9"
        a_spaced = ' '.join(str(a))
        b_spaced = ' '.join(str(b))
        result_spaced = ' '.join(str(result))
        
        sample = f"{a_spaced} + {b_spaced} -> {result_spaced}\n"
        samples.add(sample)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        samples_list = list(samples)
        random.shuffle(samples_list)
        f.writelines(samples_list)
    
    print(f"  ✓ {len(samples):,} samples -> {output_file}")
    print(f"  Sample: {list(samples)[0].strip()}")
    print(f"  Note: Using spaced format (e.g., '1 2 3 + 4 5 6 -> 5 7 9')")

def generate_composed_dataset(output_file, num_samples=10000, max_digits=3):
    """
    Generate reverse-then-add dataset (the composed task!) with spaced format.
    
    Using spaces makes the reversal pattern more explicit in the intermediate step.
    
    Example:
        Input:  1 2 3 + 4 5 6
        Step 1: Reverse both → 3 2 1 + 6 5 4
        Step 2: Add them → 9 7 5
        Format: 1 2 3 + 4 5 6 -> 3 2 1 + 6 5 4 -> 9 7 5
    
    This is what we're trying to achieve through composition!
    The question: Can we combine Model A (reverse) and Model B (addition)
    to solve this, instead of training a whole new model?
    
    The spaced format makes it easier to see the digit-by-digit reversal!
    """
    samples = set()
    max_num = 10 ** max_digits - 1
    min_num = 10 ** (max_digits - 1)
    
    print(f"Generating composed dataset...")
    print(f"  • {num_samples:,} samples")
    print(f"  • Using spaced format to show reversal clearly")
    print(f"  • Testing if A∘B can match end-to-end training!")
    
    while len(samples) < num_samples:
        a = random.randint(min_num, max_num)
        b = random.randint(min_num, max_num)
        
        # Reverse both numbers
        a_rev = int(str(a)[::-1])
        b_rev = int(str(b)[::-1])
        
        # Add reversed numbers
        result = a_rev + b_rev
        
        # Create spaced versions for better learning
        # Input: space out each number separately
        a_spaced = ' '.join(str(a))
        b_spaced = ' '.join(str(b))
        
        # Intermediate: space out the reversed numbers
        a_rev_spaced = ' '.join(str(a_rev))
        b_rev_spaced = ' '.join(str(b_rev))
        
        # Output: space out the final result
        result_spaced = ' '.join(str(result))
        
        # Format shows the composition with clear spacing
        # Example: "1 2 3 + 4 5 6 -> 3 2 1 + 6 5 4 -> 9 7 5"
        sample = f"{a_spaced} + {b_spaced} -> {a_rev_spaced} + {b_rev_spaced} -> {result_spaced}\n"
        samples.add(sample)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        samples_list = list(samples)
        random.shuffle(samples_list)
        f.writelines(samples_list)
    
    print(f"  ✓ {len(samples):,} samples -> {output_file}")
    print(f"  Sample: {list(samples)[0].strip()}")
    print(f"  Note: Spaced format makes reversal pattern explicit")

def tokenize_data(input_file, output_dir):
    """
    Tokenize text data for nanoGPT.
    
    This converts our human-readable text (like "123+456->579")
    into numbers that the neural network can process.
    Each character (0-9, +, -, >) gets assigned a unique ID.
    
    The tokenized data is saved as .bin files for fast loading during training.
    """
    if not os.path.exists(input_file):
        print(f"  ✗ Error: {input_file} not found")
        return
    
    print(f"Tokenizing: {input_file}")
    
    # Read all the text data
    with open(input_file, 'r') as f:
        data = f.read()
    
    # Build vocabulary: all unique characters in the dataset
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"  • Vocabulary: {vocab_size} unique characters")
    print(f"    Characters: {''.join(chars)}")
    
    # Create mapping: character -> number and number -> character
    stoi = {ch: i for i, ch in enumerate(chars)}  # string to int
    itos = {i: ch for i, ch in enumerate(chars)}  # int to string
    encode = lambda s: [stoi[c] for c in s]
    
    # Convert all text to numbers
    data_ids = encode(data)
    n = len(data_ids)
    
    # Split into training (90%) and validation (10%)
    train_ids = np.array(data_ids[:int(n * 0.9)], dtype=np.uint16)
    val_ids = np.array(data_ids[int(n * 0.9):], dtype=np.uint16)
    
    # Save everything
    os.makedirs(output_dir, exist_ok=True)
    train_ids.tofile(os.path.join(output_dir, 'train.bin'))
    val_ids.tofile(os.path.join(output_dir, 'val.bin'))
    
    # Save vocabulary for later use
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,  # for decoding model outputs
        'stoi': stoi,  # for encoding inputs
    }
    with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"  ✓ Saved: {len(train_ids):,} train tokens, {len(val_ids):,} validation tokens")

def main():
    parser = argparse.ArgumentParser(description='Generate composition datasets')
    parser.add_argument('--num_samples', type=int, default=10000,
                      help='Number of samples per dataset')
    parser.add_argument('--max_digits', type=int, default=3,
                      help='Maximum number of digits for addition')
    parser.add_argument('--max_len', type=int, default=4,
                      help='Maximum length for reversal')
    parser.add_argument('--generate-only', action='store_true',
                      help='Only generate, skip tokenization')
    parser.add_argument('--tokenize-only', action='store_true',
                      help='Only tokenize existing files')
    
    args = parser.parse_args()
    
    random.seed(42)
    
    datasets = [
        ('reverse', 'data/reverse', lambda f: generate_reverse_dataset(
            f, args.num_samples, args.max_len)),
        ('addition', 'data/addition', lambda f: generate_addition_dataset(
            f, args.num_samples, args.max_digits)),
        ('composed', 'data/composed', lambda f: generate_composed_dataset(
            f, args.num_samples, args.max_digits)),
    ]
    
    if not args.tokenize_only:
        print("=" * 70)
        print("📊 GENERATING DATASETS")
        print("=" * 70)
        print("\nCreating three datasets to test if transformers can compose...\n")
        for name, dirname, generator in datasets:
            print()
            generator(f'{dirname}/train.txt')
    
    if not args.generate_only:
        print("\n" + "=" * 70)
        print("🔢 TOKENIZING DATASETS")
        print("=" * 70)
        print("\nConverting text to numbers for neural network training...\n")
        for name, dirname, _ in datasets:
            print()
            tokenize_data(f'{dirname}/train.txt', dirname)
    
    print("\n" + "=" * 70)
    print("✅ ALL DONE!")
    print("=" * 70)
    print("\n🎯 What you just created:")
    print("  • REVERSE dataset:  Teaches model to reverse numbers")
    print("  • ADDITION dataset: Teaches model to add numbers")
    print("  • COMPOSED dataset: Tests end-to-end reverse-then-add")
    print("\n🚀 Next steps:")
    print("  1. Train reverse model:    config/train_reverse.py")
    print("  2. Train addition model:   config/train_addition.py")
    print("  3. Train composed baseline: config/train_composed.py")
    print("  4. Run composition experiments!")
    print("\n💡 Quick start: ./run_experiments.sh")

if __name__ == '__main__':
    main()
