# Transformer Algebra: Can We Mix and Match Models?

Here's the idea: what if we could train transformers on simple tasks separately and then compose them like LEGO blocks? Instead of training one massive model from scratch on a complex task T∘U, we'd train a small model A on T, another model B on U, and then figure out how to combine them.

**Concrete example:** I trained one tiny model to reverse digits (123 → 321) and another to add numbers (100+200 → 300). The question is: can I compose them to do reverse-then-add (123+456 → 321+654 → 975) without ever training on that combined task?

Basically, can transformers work like modular functions? Let's find out.

## Why Bother?

Training big models is slow and expensive. If we could build specialized models for simple tasks and compose them on the fly, we'd get:
- **Sample efficiency** - learn complex tasks using way less data
- **Modularity** - reuse the same models across different problems
- **Interpretability** - actually understand what each piece is doing

The million-dollar question: can composition match training from scratch?

## How to Compose Models

I'm experimenting with three approaches:

**1. Pipeline (Zero-shot)**  
Literally just chain them: run model A, take its output text, feed it to model B. No training whatsoever. The catch? They probably don't "speak the same language" internally.

**2. Layer Concatenation + Fine-tuning**  
Grab the first half of model A's layers, the second half of model B's layers, bolt them together, and fine-tune a bit. This uses what both models learned but needs some training.

**3. Adapter Network**  
Freeze both models completely, train a tiny "translator" network between them. Only ~16K parameters to train instead of 800K. Might work, might be a bottleneck.

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
├── evaluate_all.py                      # Comprehensive evaluation (all strategies)
└── wandb/                               # Training logs
```

## Getting Started

### 1. Make Some Data

```bash
cd data
python prepare_tokenized.py
```

This generates three datasets (10K examples each):
- **reverse/** - flipping digits: `12345 → 54321`
- **addition/** - basic math: `123+456 → 579`  
- **composed/** - both operations: `123+456 → 321+654 → 975`

Need more? Try `--num_samples 50000` or `--max_digits 5` for longer numbers.

### 2. Train Some Models

Reversal model:
```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train.py config/train_reverse.py
```

Addition model:
```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train.py config/train_addition.py
```

Both use the exact same architecture (4 layers, 4 heads, 128-dim) for a fair comparison. Takes maybe 5-10 minutes on a GPU, gets to ~90%+ accuracy pretty quickly.

### 3. Train a Baseline

Let's train a model end-to-end on the composed task so we have something to beat:
```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/train.py config/train_composed.py
```

### 4. See How Well It Works

Test individual models:
```bash
python evaluate_reverse.py              # How good is the reversal model?
python evaluate_addition.py             # How about addition?
python evaluate_addition_scratchpad.py  # Scratchpad version
python evaluate_composed.py             # Zero-shot pipeline test
```

Or get a full report on everything:
```bash
python evaluate_all.py --n 100          # Quick test (100 samples)
python evaluate_all.py --n 1000         # More thorough
```

## Why Space Out the Digits?

All the datasets use spacing: `1 2 3` instead of `123`. Why?

**Reversal:** `1 2 3 4 5 → 5 4 3 2 1`  
**Addition:** `1 2 3 + 4 5 6 → 5 7 9`  
**Composed:** `1 2 3 + 4 5 6 → 3 2 1 + 6 5 4 → 9 7 5`

Spacing makes everything clearer:
- Each digit becomes its own token - easier to learn position-by-position
- You can literally *see* the pattern (watch the digits flip!)
- When something breaks, you know exactly which digit went wrong
- It's like showing your work in math class - makes the intermediate steps explicit

## Model Specs

All models use the same tiny architecture (~200K parameters):
- 4 layers, 4 attention heads per layer
- 128-dimensional embeddings  
- 32-token context window
- 10% dropout

Why so small? Faster experiments, clearer signal, less risk of just memorizing everything.

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

## Results

Evaluated all composition approaches on 100 test samples:

### Individual Models
| Model    | Accuracy |
|----------|----------|
| Reversal | 100%     |
| Addition | 100%     |

Both models perfectly learned their tasks.

### Composition Strategies

**Zero-shot Pipeline** (no training required)
```
Reversal step:  100% ✓
Full pipeline:   81% 
Performance drop: 19%
```
The addition model can't properly interpret the reversal model's output. They don't share the same internal representations (representation mismatch).

**Layer Concatenation** (implementation ready)
- Status: Code complete in `compose.py`
- Architecture: First 2 layers from reversal + last 2 from addition
- Parameters: ~797K total
- Next step: Fine-tune on composed task

**Adapter Network** (implementation ready)
- Status: Architecture complete, both base models frozen
- Parameters: Only ~16K trainable (2% of full model)
- Next step: Train adapter on composed task

**Task Arithmetic** (tested, not viable)
```
Direct merge: <2% accuracy
```
Failed because models were trained from scratch, not fine-tuned from a shared base. Their weight spaces don't align geometrically.

### Key Findings
1. **Individual models work perfectly** - 100% accuracy on both tasks
2. **Composition has a 19% gap** - zero-shot pipeline loses accuracy due to representation mismatch
3. **Task arithmetic requires shared initialization** - doesn't apply to scratch-trained models
4. **Alternative approaches ready** - layer concat and adapter need training/fine-tuning

### Next Experiments
- Fine-tune layer concatenation model and measure sample efficiency vs end-to-end
- Train adapter network with limited data
- Test generalization on longer sequences (5-6 digits)
- Compare all approaches quantitatively

## Related Work

This whole idea of compositionality in neural networks asks: can you build complex behaviors by combining simpler pieces? Since transformers build up increasingly abstract representations layer by layer, maybe we can isolate and recombine the features they learn.

**Papers that inspired this:**
- **Neural Module Networks** (Andreas et al., 2016) - composing vision modules to answer questions
- **Modular Transformers** (Csordás et al., 2021) - training task-specific modules that can work together
- **Model Merging** (Wortsman et al., 2022) - averaging weights from models trained on different tasks  
- **Task Arithmetic** (Ilharco et al., 2023) - the big one: adding and subtracting task vectors in weight space

I'm basically taking task arithmetic ideas and testing them on sequential reasoning tasks, plus trying alternative strategies when task arithmetic doesn't apply.

## Credits

This extends ideas from the "Editing Models with Task Arithmetic" paper:

```bibtex
@article{ilharco2023editing,
  title={Editing Models with Task Arithmetic},
  author={Ilharco, Gabriel and Ribeiro, Marco Tulio and Wortsman, Mitchell and Schmidt, Ludwig and Hajishirzi, Hannaneh and Farhadi, Ali},
  journal={ICLR},
  year={2023}
}
```

Built on top of [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy. Project for COMP 560.

MIT License.