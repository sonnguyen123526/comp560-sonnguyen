# Arithmetic with Scratchpad

This folder studies a simple but important question: does making intermediate arithmetic steps explicit help a small transformer learn addition better?

The experiment compares two dataset formats:

1. A direct input-output baseline, for example `123+456->579`
2. A scratchpad format that exposes the column-wise computation, for example `123+456->3+6=9,2+5=7,1+4=5->579`

The implementation is intentionally small and self-contained. It contains dataset generation, tokenization, and two nanoGPT training configs.

## Contents

```text
arithmetic-scratchpad/
├── README.md
├── config/
│   ├── without_scratchpad.py
│   └── with_scratchpad.py
├── data/
│   ├── generate_simple.py
│   ├── generate_limited.py
│   ├── prepare_tokenized.py
│   ├── without_scratchpad/
│   └── with_scratchpad/
├── without_scratchpad/
└── with_scratchpad/
```

The `without_scratchpad/` and `with_scratchpad/` directories are generated local artifacts that hold the raw text and tokenized files used by training.

## What Is Here

### `config/without_scratchpad.py`

Baseline training config.

- dataset: `without_scratchpad`
- model: 4-layer GPT
- heads: 4
- embedding size: 128
- batch size: 12
- block size: 64
- max iters: 2000
- device: CPU

### `config/with_scratchpad.py`

Scratchpad training config.

- dataset: `with_scratchpad`
- same model size as the baseline for a fair comparison
- block size: 128 to fit the longer target sequence
- max iters: 2000
- device: CPU

### `data/generate_simple.py`

Generates a very small single-digit addition dataset.

Key behavior:

- enumerates all `0-9 + 0-9` pairs
- writes `without_scratchpad/train.txt` and `with_scratchpad/train.txt`
- repeats the 100 unique examples many times to create a 50,000-sample training set

This script is useful as a minimal sanity-check generator.

### `data/generate_limited.py`

Generates a limited 2-digit addition dataset over the range `10-50`.

Key behavior:

- writes `without_scratchpad/train.txt` and `with_scratchpad/train.txt`
- produces a larger, still bounded curriculum than `generate_simple.py`
- keeps the same scratchpad logic as the main dataset

### `data/prepare_tokenized.py`

Main dataset preparation script.

It can:

- generate training data
- generate OOD test data with `--test`
- skip generation with `--tokenize-only`
- skip tokenization with `--generate-only`

By default it produces:

- `without_scratchpad/train.txt`
- `with_scratchpad/train.txt`
- `without_scratchpad/train.bin`, `without_scratchpad/val.bin`, `without_scratchpad/meta.pkl`
- `with_scratchpad/train.bin`, `with_scratchpad/val.bin`, `with_scratchpad/meta.pkl`

When invoked with `--test`, it writes test files instead of train files and uses longer numbers for out-of-distribution evaluation.

## Typical Workflow

### 1. Generate and tokenize data

From `arithmetic-scratchpad/data/`:

```bash
python prepare_tokenized.py
```

Useful options:

```bash
python prepare_tokenized.py --num_samples 100000 --max_digits 4
python prepare_tokenized.py --generate-only
python prepare_tokenized.py --tokenize-only
python prepare_tokenized.py --test --max_digits 5
```

### 2. Train the models

Baseline:

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python -u ../../comp560-nanoGPT/train.py config/without_scratchpad.py
```

Scratchpad:

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python -u ../../comp560-nanoGPT/train.py config/with_scratchpad.py
```

### 3. Sample from a trained model

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python -u ../../comp560-nanoGPT/sample.py config/without_scratchpad.py
```

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python -u ../../comp560-nanoGPT/sample.py config/with_scratchpad.py
```

## Data Format

### Baseline

The baseline keeps the task as direct sequence prediction:

```text
1+2->3
12+34->46
99+1->100
```

### Scratchpad

The scratchpad format surfaces the arithmetic steps explicitly:

```text
12+34->2+4=6,1+3=4->46
```

Carries are represented inline:

```text
99+99->9+9=8c1,9+9+1=9c1,1->198
```

This representation is right-to-left, matching the standard addition algorithm.

## Why This Folder Exists

This experiment isolates a single hypothesis: if the model sees the computation trace, does it learn addition more reliably than if it is asked to infer the entire algorithm from input-output pairs alone?

That makes this folder useful for comparing:

- direct mapping versus explicit computation
- short-horizon fit versus longer-number generalization
- raw accuracy versus actual sample quality

## Notes

- The generated `without_scratchpad/` and `with_scratchpad/` directories are local artifacts and can be recreated from `data/`.
- The model sizes are intentionally matched between conditions, so differences should be attributable to the scratchpad format rather than capacity.
- `generate_simple.py` and `generate_limited.py` are legacy or alternative data generators; `prepare_tokenized.py` is the main entry point.
- There is no separate evaluation harness in this folder. Generalization is typically assessed by generating longer test sets with `prepare_tokenized.py --test` and sampling from the trained checkpoints.

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
