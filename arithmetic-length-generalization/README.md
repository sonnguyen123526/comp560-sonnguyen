# Arithmetic Length Generalization

This folder studies whether a small transformer trained on short addition problems can generalize to much longer inputs when the computation is made explicit in the training data.

The experiment follows the scratchpad ideas from *How Far Can Transformers Reason? The Globality Barrier and Inductive Scratchpad* and implements two dataset formats from that line of work:

1. Random spaces with pointer-style position markers
2. Cyclic shifts with stepwise state tracking

The scope of this folder is intentionally narrow: it contains data generators, tokenization, and training configs for these two formats.

## Contents

```text
arithmetic-length-generalization/
├── README.md
├── config/
│   ├── random_spaces.py
│   └── cyclic_shifts.py
├── data/
│   ├── generate_random_space.py
│   ├── generate_using_shifts.py
│   ├── prepare_tokenized.py
│   ├── random_spaces/
│   └── cyclic_shifts/
```

## What Is Here

### `config/random_spaces.py`

Training configuration for the random-spaces dataset.

- `dataset = 'random_spaces'`
- 4-layer GPT
- 4 attention heads
- 128-dimensional embeddings
- CPU execution by default
- `max_iters = 2000`

### `config/cyclic_shifts.py`

Training configuration for the cyclic-shifts dataset.

- `dataset = 'cyclic_shifts'`
- same model size and training budget as the random-spaces run
- CPU execution by default

### `data/generate_random_space.py`

Generates addition examples with random underscores and an explicit scratchpad.

Key behavior:

- Creates `random_spaces/train.txt`
- Supports `--test` to write `random_spaces/test.txt`
- Supports `--max_digits` and `--num_samples`
- Uses right-to-left addition with pointer markers and carry tracking

### `data/generate_using_shifts.py`

Generates the cyclic-shifts representation from the paper.

Key behavior:

- Creates `cyclic_shifts/train.txt`
- Supports `--test` to write `cyclic_shifts/test.txt`
- Supports `--max_digits` and `--num_samples`
- Encodes the running computation as a sequence of rotated states

### `data/prepare_tokenized.py`

Tokenizes the generated text datasets into nanoGPT-ready binary files.

It reads:

- `random_spaces/train.txt`
- `cyclic_shifts/train.txt`

and writes:

- `random_spaces/train.bin`, `random_spaces/val.bin`, `random_spaces/meta.pkl`
- `cyclic_shifts/train.bin`, `cyclic_shifts/val.bin`, `cyclic_shifts/meta.pkl`

## How To Use It

### 1. Generate the datasets

From `arithmetic-length-generalization/data/`:

```bash
python generate_random_space.py
python generate_using_shifts.py
```

Optional test sets:

```bash
python generate_random_space.py --test --max_digits 5
python generate_using_shifts.py --test --max_digits 5
```

### 2. Tokenize for training

```bash
python prepare_tokenized.py
```

### 3. Train with nanoGPT

Random spaces:

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python -u ../../comp560-nanoGPT/train.py config/random_spaces.py
```

Cyclic shifts:

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python -u ../../comp560-nanoGPT/train.py config/cyclic_shifts.py
```

## Data Format

### Random spaces

The input is a short addition problem with underscores inserted as noise. The target is a scratchpad that exposes:

- digit pointers
- per-step carries
- the running answer state
- start and end sentinels

The goal is to encourage the model to learn an algorithm, not just a string mapping.

### Cyclic shifts

The input uses randomized prefixes and a `$` marker. The target tracks the computation as the numbers rotate through a sequence of states.

This representation is more structured than plain addition and is designed to make long-range generalization easier.

## Why This Folder Exists

The core question is not whether a transformer can fit short addition. It is whether an explicit algorithmic representation can help the model extrapolate beyond the length regime it saw during training.

In practice, this folder is useful for comparing:

- direct sequence learning versus explicit computation traces
- pointer-like representations versus state-rotation representations
- short-length fit versus longer-length generalization

## Relationship To `arithmetic-scratchpad/`

The `arithmetic-scratchpad/` project uses a simpler scratchpad format and is easier to train, but it is not aimed at the same extrapolation problem.

This folder is the more structured, length-generalization-oriented version of that idea.

## Notes

- The generated binary datasets are local artifacts and can be recreated from the scripts in `data/`.
- The training configs are intentionally small and CPU-friendly.
- There is no separate evaluation harness in this folder; generalization is tested by generating longer test examples with the `--test` flag and sampling from the trained model.

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
