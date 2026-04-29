# Insert Spaces

This folder explores a simple character-level transformation task: take an unspaced word like `hello` and produce `h e l l o`.

The project is useful as a small negative control. It tests whether a tiny GPT-style model can learn a formatting transformation that is trivial for humans but awkward for a pure autoregressive language model.

There are two parts:

1. A GPT-based baseline built around the `data/basic/` dataset and the `config/basic.py` training config
2. An exploratory `seq2seq_model_testing/` subfolder that tries encoder-decoder alternatives

## Contents

```text
insert-spaces/
├── README.MD
├── config/
│   └── basic.py
├── data/
│   └── basic/
│       ├── prepare.py
│       ├── train.txt
│       ├── val.txt
│       ├── train.bin
│       ├── val.bin
│       └── meta.pkl
└── seq2seq_model_testing/
    ├── README.md
    ├── simple_seq2seq_model.py
    ├── train_simple_seq2seq.py
    ├── sample_simple_seq2seq.py
    ├── seq2seq_model.py
    ├── train_seq2seq.py
    └── sample_seq2seq.py
```

## What Is Here

### `config/basic.py`

Training config for the GPT baseline.

- 4-layer GPT
- 4 attention heads
- 128-dimensional embeddings
- batch size 12
- block size 64
- 2,000 training iterations
- CPU execution by default

### `data/basic/prepare.py`

Main dataset preparation script for the GPT baseline.

It generates a character-level dataset in the format:

```text
Input: hello Output: h e l l o
```

Key behavior:

- creates about 3,000 unique words
- uses lowercase letters only
- builds a roughly 1M-character dataset
- writes `train.txt`, `val.txt`, `train.bin`, `val.bin`, and `meta.pkl`

The file `prepare.py.bak` is a backup copy and is not part of the main workflow.

## How To Use It

### 1. Prepare the dataset

From `insert-spaces/data/basic/`:

```bash
python prepare.py
```

This writes the raw text and tokenized files into the same directory.

### 2. Train the GPT baseline

From `insert-spaces/`:

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python -u ../../comp560-nanoGPT/train.py config/basic.py
```

### 3. Sample from the trained model

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python -u ../../comp560-nanoGPT/sample.py config/basic.py \
  --num_samples=3 --max_new_tokens=100 --seed=2
```

## Data Format

The dataset uses a clear prompt style:

```text
Input: hello Output: h e l l o
```

The intent is to make the transformation explicit while still framing it as a character-level prediction task.

## Seq2Seq Experiments

The `seq2seq_model_testing/` folder investigates whether a more task-appropriate encoder-decoder architecture handles the problem better than GPT.

It contains two variants:

1. `simple_seq2seq_model.py` and `train_simple_seq2seq.py` for a minimal GRU-based encoder-decoder
2. `seq2seq_model.py` and `train_seq2seq.py` for a GRU model with attention

The bundled README in that subfolder documents the architecture choices and notes that the results were still exploratory.

## Why This Folder Exists

This is a useful diagnostic task because success or failure is informative:

- If GPT solves it easily, then the task is too easy to distinguish architecture choices.
- If GPT struggles, the failure is a reminder that not every transformation is a good fit for a decoder-only language model.
- If seq2seq helps, that supports the idea that the architecture matters as much as the data.

## Notes

- The `data/basic/` directory is the source of truth for the GPT baseline.
- `train.bin`, `val.bin`, and `meta.pkl` are generated artifacts and can be recreated by rerunning `prepare.py`.
- The seq2seq subfolder is experimental and separate from the GPT baseline.
- This folder does not currently include a unified evaluation harness; sampling and manual inspection are the primary checks.

## Reference

Relevant sequence-to-sequence references:

- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
