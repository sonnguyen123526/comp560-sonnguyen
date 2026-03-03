# Transformer Algebra: Composing Learned Operations

Can we train transformers on simple tasks separately and compose them to solve harder problems? Instead of training one big model from scratch for task T∘U, what if we trained model A on T, model B on U, and then combined them?

**Example:** Train one model to reverse digits (123 → 321) and another to add numbers (100+200 → 300). Then compose them to solve reverse-then-add (123+456 → 321+654 → 975) without ever training on that combined task.

We're testing if transformers can work like modular functions that you can mix and match.

## Why This Matters

Training large models is expensive and time-consuming. If we can build specialized models for simple tasks and compose them, we might get:
- **Better sample efficiency** - learn complex tasks with less data
- **Modularity** - reuse trained components across different problems
- **Interpretability** - understand what each piece does

The big question: can composition match (or beat) training from scratch?

## Composition Approaches

We're testing three ways to combine models:

**1. Pipeline (Zero-shot)**  
Just run model A, then feed its output to model B. No training needed, but the models might not "speak the same language" internally.

**2. Layer Concatenation + Fine-tuning**  
Take the first few layers from model A, the last few from model B, stick them together, and fine-tune. Uses both models' learned features but needs some training.

**3. Adapter Network**  
Freeze both models completely and train a small "translator" network between them. Minimal new parameters, but the adapter might limit what information gets through.

## Project Structure

```
transformer-algebra/
├── README.md
├── config/                              # Training configs
│   ├── train_reverse.py                 # Digit reversal task
│   ├── train_addition.py                # Simple addition
│   ├── train_addition_scratchpad.py     # Addition with intermediate steps
│   ├── train_addition_ft.py             # Fine-tuning config
│   └── train_composed.py                # End-to-end baseline
├── data/                                # Generated datasets
│   ├── prepare_tokenized.py             # Main dataset generator
│   ├── prepare_addition_scratchpad.py   # Scratchpad data generator
│   ├── generate_addition_test.py        # Test set generator
│   ├── generate_composed_test.py        # Composed test generator
│   ├── reverse/                         # Reversal task data (train/val)
│   ├── addition/                        # Addition task data
│   ├── addition_scratchpad/             # Addition with steps shown
│   ├── composed/                        # Composed task baseline
│   └── reverse_ft/                      # Fine-tuning data
├── out/                                 # Trained model checkpoints
├── compose.py                           # Composition implementations
├── task_vectors.py                      # Task vector operations
├── task_arithmetic.py                   # Task arithmetic methods
├── evaluate_reverse.py                  # Reversal model evaluation
├── evaluate_addition.py                 # Addition model evaluation
├── evaluate_addition_scratchpad.py      # Scratchpad evaluation
├── evaluate_composed.py                 # Composed model evaluation
└── wandb/                               # Training logs
```

## Quick Start

### 1. Generate the Datasets

```bash
cd data
python prepare_tokenized.py
```

This creates three datasets with 10,000 examples each:
- **reverse/** - digit reversal like `12345 → 54321`
- **addition/** - simple addition like `123+456 → 579`  
- **composed/** - reverse then add like `123+456 → 321+654 → 975`

Want more data or longer numbers? Use `--num_samples 50000` or `--max_digits 5`

### 2. Train the Models

Train reversal:
```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train.py config/train_reverse.py
```

Train addition:
```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train.py config/train_addition.py
```

Both use identical architectures (4 layers, 4 heads, 128-dim) so we're comparing apples to apples. Takes about 5-10 minutes on a GPU, hits 90%+ accuracy.

### 3. Train the Baseline

Train an end-to-end model on the composed task so we have something to compare against:
```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train.py config/train_composed.py
```

### 4. Evaluate Models

Test each model on its task:
```bash
python evaluate_reverse.py              # Test reversal model
python evaluate_addition.py             # Test addition model
python evaluate_addition_scratchpad.py  # Test scratchpad version
python evaluate_composed.py             # Test end-to-end baseline
```

## Dataset Design

We use spacing between digits to make patterns clearer:

**Reversal:** `1 2 3 4 5 → 5 4 3 2 1`  
**Addition:** `1 2 3 + 4 5 6 → 5 7 9`  
**Composed:** `1 2 3 + 4 5 6 → 3 2 1 + 6 5 4 → 9 7 5`

Spacing helps because:
- Each digit is a separate token - easier to learn position-by-position operations
- Patterns become more obvious - you can literally see the reversal happening
- Errors are easier to debug - you know exactly which digit went wrong
- Similar to "scratchpad" methods that show work step-by-step

## Model Architecture

All models use the same architecture (~200K parameters total):
- 4 layers with 4 attention heads each
- 128-dimensional embeddings  
- Context window of 32 tokens
- 10% dropout

We keep it small deliberately - faster training, clearer results, less overfitting on small datasets.

## Usage Examples

Quick test of individual models:
```bash
# Test the reversal model
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/sample.py config/train_reverse.py --start="12345->"

# Test the addition model
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/sample.py config/train_addition.py --start="123+456->"
```

Compose models in Python:
```python
from compose import load_model, create_composition

model_A = load_model('out/reverse/ckpt.pt')
model_B = load_model('out/addition/ckpt.pt')
composed = create_composition(model_A, model_B, strategy='concat')
# Fine-tune if needed...
```

## Acknowledgments

This project builds on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy and draws inspiration from research on compositionality and modularity in neural networks. Created as part of COMP 560.

MIT License - see LICENSE file.