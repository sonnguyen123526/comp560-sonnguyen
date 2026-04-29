# COMP 560 - Experiments with Small Transformers

Can small GPT models learn to reason step-by-step? Can we combine them like LEGO blocks? This repo explores arithmetic reasoning, model composition, and chain-of-thought learning using [nanoGPT](https://github.com/karpathy/nanoGPT).

## Projects

### 1. Transformer Algebra ([transformer-algebra/](transformer-algebra/))

What if we could train models on simple tasks separately and mix them like LEGO blocks? The goal is to compose reverse and addition models to solve reverse-then-add without training end-to-end.

This folder now has two tracks:
- Task-vector/model-merging experiments (TIES/DARE + baselines) with a shared base and unified vocabulary.
- Composition strategies (pipeline, layer concatenation, adapter) trained from scratch.

**The verdict:** Composition remains hard. Single-task accuracy can be high, but composite accuracy stays low in the benchmark. See [transformer-algebra/README.md](transformer-algebra/README.md) for current results and artifacts.

See [transformer-algebra/README.md](transformer-algebra/README.md) for implementation details.

### 2. Arithmetic with Scratchpad ([arithmetic-scratchpad/](arithmetic-scratchpad/))

Do transformers learn better when you show them the work? Testing baseline (`123+456→579`) vs. scratchpad (`123+456→3+6=9,2+5=7,1+4=5→579`) on 50k addition problems. Both models use the same 4-layer architecture.

Based on "How Far Can Transformers Reason?" (Anil et al., 2024).

### 3. Arithmetic Length Generalization ([arithmetic-length-generalization/](arithmetic-length-generalization/))

Can models trained on 2-3 digit addition generalize to 10+ digits? Testing two scratchpad formats from the paper:
- **Random spaces:** Position pointers with underscores
- **Cyclic shifts:** Rotation with explicit state tracking

The paper shows these enable length generalization. Our implementation is in progress.

### 4. Insert Spaces ([insert-spaces/](insert-spaces/))

Early experiment: can GPT learn to insert spaces between letters? ("hello" → "h e l l o")

Trained a 4-layer model (0.79M params) on 3,000 words. It failed—generated wrong characters entirely instead of just adding spaces (val loss: 1.17). Lesson: architecture too small, or this needs a seq2seq approach instead of pure language modeling.

### 5. Vietnamese-English Translation ([translation/](translation/))

Trained a 6-layer GPT to translate Vietnamese numbers to English ("một" → "one", "mười hai" → "twelve"). Achieved ~85-90% accuracy on 50k training pairs.

**Why it worked:** Bigger model (6 layers vs 4), dropout, warmup, and clean data format.

## Repository Structure

```
comp560-sonnguyen/
├── README.md                         # This file
├── transformer-algebra/              # Model composition experiments
│   ├── config/                       # Shared-base training configs (base + fine-tune)
│   ├── data/                         # Dataset generation + tokenization
│   ├── composition_strategies/       # Pipeline/concat/adapter experiments
│   │   ├── config/                   # Scratch model configs
│   │   ├── compose.py
│   │   └── evaluate_*.py
│   ├── ties.py                       # TIES merge + composite eval
│   ├── dare.py                       # DARE merge + composite eval
│   ├── task_arithmetic.py           # Task arithmetic methods
│   ├── task_vector_conflict.py      # Conflict diagnostics
│   ├── cka.py                       # Representation similarity
│   ├── evaluate_stratified_composition.py
│   ├── out/                         # Model checkpoints
│   └── wandb/                       # Training logs
├── arithmetic-scratchpad/            # Scratchpad reasoning experiment
│   ├── config/                       # Training configs (with/without scratchpad)
│   ├── data/                         # Dataset generation + tokenization
│   ├── out/                          # Model checkpoints
│   └── wandb/                        # Training logs
├── arithmetic-length-generalization/ # Length generalization experiment
│   ├── config/                       # Training configs (random_spaces, cyclic_shifts)
│   ├── data/                         # Dataset generation + tokenization
│   ├── out/                          # Model checkpoints
│   └── wandb/                        # Training logs
├── insert-spaces/                    # Character spacing experiment
│   ├── config/
│   ├── data/
│   ├── seq2seq_model_testing/       # Alternative seq2seq approach
│   ├── out/
│   └── wandb/
├── translation/                      # Vietnamese-English number translation
│   ├── config/
│   ├── data/
│   └── out/
├── assets/                           # Plots and figures
└── wandb/                            # Shared training logs
```

## Quick Start

All training uses the same pattern. Set your nanoGPT path once:
```bash
export NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py
```

### Train Composition Models

```bash
cd transformer-algebra
python data/prepare_tokenized.py

# Train individual models
python ../../comp560-nanoGPT/train.py composition_strategies/config/train_reverse.py
python ../../comp560-nanoGPT/train.py composition_strategies/config/train_addition.py

# Evaluate all strategies
python composition_strategies/evaluate_all.py --n 1000
```

### Train Shared-Base (TIES/DARE) Models

```bash
cd transformer-algebra
python data/prepare_mixed.py

python ../../comp560-nanoGPT/train.py config/train_base.py

mkdir -p out/reverse_ft out/addition_ft
cp out/base/ckpt.pt out/reverse_ft/ckpt.pt
cp out/base/ckpt.pt out/addition_ft/ckpt.pt

python ../../comp560-nanoGPT/train.py config/train_reverse_ft.py
python ../../comp560-nanoGPT/train.py config/train_addition_ft_base.py

# Example sweeps
python ties.py --sweep --n 200 --device mps
python dare.py --sweep --n 200 --device mps
```

### Train Scratchpad Models

```bash
cd arithmetic-scratchpad/data && python prepare_tokenized.py && cd ..

# Compare with/without scratchpad
python -u ../../comp560-nanoGPT/train.py config/without_scratchpad.py
python -u ../../comp560-nanoGPT/train.py config/with_scratchpad.py
```

### Sample from any model

```bash
python -u ../../comp560-nanoGPT/sample.py config/YOUR_CONFIG.py --num_samples=10
```

## Results

### Transformer Algebra: The Composition Challenge

Task-vector merges (TIES/DARE) can preserve strong reverse/addition accuracy, but composite reverse-then-add remains near-zero in the benchmark. The scratch composition strategies (pipeline/concat/adapter) live under composition_strategies; see [transformer-algebra/README.md](transformer-algebra/README.md) for the latest numbers and result files.

### Other Results

- **Translation:** ~85-90% accuracy on Vietnamese → English numbers. The 6-layer model learns diacritics naturally.
- **Insert Spaces:** Complete failure. Model outputs wrong characters instead of just spacing them.

## Lessons Learned

**Size matters, but smartly.** The 32MB translation model succeeds while the 9.6MB spacing model fails. But it's not just size—architecture type (seq2seq vs language modeling) and training setup (dropout, warmup) matter just as much.

**Data format is crucial.** Clean formats (`input → output`) beat verbose ones (`Input: X Output: Y`). Spacing digits (`1 2 3` vs `123`) makes patterns clearer. Scratchpads help with multi-step reasoning, but only when designed well.

**Composition is genuinely hard.** You can't just snap together independently trained models. They live in different representation spaces. This is a fundamental challenge, not a tuning problem.

**Trust outputs, not just loss.** The insert-spaces model hit loss 1.17 but still failed the task completely. Always check actual predictions.

## Requirements

- Python 3.8+, PyTorch, NumPy
- [nanoGPT](https://github.com/karpathy/nanoGPT) (set `NANOGPT_CONFIG` to point to configurator.py)

## License

MIT
