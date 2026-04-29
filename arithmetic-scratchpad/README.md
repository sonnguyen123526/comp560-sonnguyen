# Arithmetic with Scratchpad

Can transformers learn multi-step reasoning better if we show them the intermediate steps? This experiment tests that idea using simple addition. Based on "How Far Can Transformers Reason? The Globality Barrier and Inductive Scratchpad" (Anil et al., 2024).

## The Idea

When adding numbers, humans work column-by-column from right to left. Can we teach a transformer to do the same by showing it a "scratchpad" of intermediate steps?

**Without scratchpad (baseline):**
```
123+456->579
```
The model has to compute everything in one shot.

**With scratchpad:**
```
123+456->3+6=9,2+5=7,1+4=5->579
```
The model sees the step-by-step work, potentially making it easier to learn the algorithm.

## Project Structure

```
arithmetic-scratchpad/
├── README.md
├── config/
│   ├── without_scratchpad.py      # Training config for baseline
│   └── with_scratchpad.py         # Training config for scratchpad version
├── data/
│   ├── prepare_tokenized.py       # Main script: generate + tokenize datasets
│   ├── generate_simple.py         # Legacy generator
│   ├── generate_limited.py        # Legacy generator
│   ├── without_scratchpad/        # Tokenized dataset (train.bin/val.bin/meta.pkl)
│   └── with_scratchpad/           # Tokenized dataset (train.bin/val.bin/meta.pkl)
├── without_scratchpad/            # Raw train.txt (+ tokenized copies)
├── with_scratchpad/               # Raw train.txt (+ tokenized copies)
├── out/                           # Model checkpoints saved here
├── wandb/                         # Training logs
```

## Usage

### 1. Generate datasets

```bash
cd data
python prepare_tokenized.py
```

This generates both datasets (with and without scratchpad) and tokenizes them. By default:
- 50,000 samples per dataset
- 2-3 digit numbers
- 90/10 train/val split

Options:
```bash
# Custom parameters
python prepare_tokenized.py --num_samples 100000 --max_digits 4

# Only generate (skip tokenization)
python prepare_tokenized.py --generate-only

# Only tokenize existing files
python prepare_tokenized.py --tokenize-only

# Generate test set with longer numbers
python prepare_tokenized.py --test --max_digits 5
```

### 2. Train models

Baseline (no scratchpad):
```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python -u ../../comp560-nanoGPT/train.py config/without_scratchpad.py
```

With scratchpad:
```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python -u ../../comp560-nanoGPT/train.py config/with_scratchpad.py
```

Both models use identical architecture (4 layers, 4 heads, 128d embeddings) so any performance difference is due to the scratchpad format.

### 3. Sample from trained models

```bash
# Baseline
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python -u ../../comp560-nanoGPT/sample.py config/without_scratchpad.py

# Scratchpad  
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python -u ../../comp560-nanoGPT/sample.py config/with_scratchpad.py
```

## Format Details

### Baseline Format
Just input and output:
```
1+2->3
12+34->46
99+1->100
```

### Scratchpad Format
Shows the column-by-column work:
```
12+34->2+4=6,1+3=4->46
```

With carries:
```
99+99->9+9=8c1,9+9+1=9c1,1->198
```
- `9+9=8c1` means 9+9=18, write 8, carry 1
- `9+9+1=9c1` means 9+9+carry=19, write 9, carry 1  
- Final carry `1` becomes the leading digit

The algorithm works right-to-left, matching how humans actually do addition.

## Expected Results

The scratchpad model should:
- Converge faster (lower validation loss)
- Get higher exact match accuracy
- Generalize better to longer numbers

Why? Because the scratchpad encodes the algorithm structure, making it easier for the model to learn the underlying computation rather than just memorizing input-output pairs.

## What to Analyze

- Compare validation loss curves
- Measure exact match accuracy on test sets
- Test generalization to 4-5 digit numbers (out of distribution)
- Look at error patterns: where does the baseline fail? Carry errors? Specific digit positions?

## Reference

```bibtex
@article{anil2024transformers,
  title={How Far Can Transformers Reason? The Globality Barrier and Inductive Scratchpad},
  author={Anil, Cem and Wu, Yuhuai and Andreassen, Anders and Lewkowycz, Aitor and 
          Misra, Vedant and Ramasesh, Vinay and Slone, Ambrose and Gur-Ari, Guy and 
          Dyer, Ethan and Neyshabur, Behnam},
  journal={arXiv preprint arXiv:2406.06467},
  year={2024}
}
```

Related work:
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) (Wei et al., 2022)
- [Show Your Work: Scratchpads for LMs](https://arxiv.org/abs/2112.00114) (Nye et al., 2021)
