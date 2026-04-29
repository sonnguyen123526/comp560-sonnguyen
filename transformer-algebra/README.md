# Transformer Algebra

This folder studies whether independently trained small transformer models can be composed in weight space or in execution space to solve a harder task than either model was trained on directly.

The running example is reverse-then-add:

- Reverse: `1 2 3 -> 3 2 1`
- Addition: `1 2 3 + 4 5 6 -> 5 7 9`
- Composed: `1 2 3 + 4 5 6 -> 3 2 1 + 6 5 4 -> 9 7 5`

The codebase is organized around two experimental tracks:

1. Shared-base task-vector experiments, including TIES, DARE, and task arithmetic.
2. Composition strategies trained from scratch, including pipeline, layer concatenation, and adapter-style composition.

## Contents

```text
transformer-algebra/
├── README.md
├── config/
│   ├── train_base.py
│   ├── train_reverse_ft.py
│   └── train_addition_ft_base.py
├── data/
│   ├── prepare_mixed.py
│   ├── prepare_tokenized.py
│   ├── prepare_addition_scratchpad.py
│   ├── generate_addition_test.py
│   ├── generate_composed_test.py
│   ├── check_test_leakage.py
│   ├── mixed/
│   ├── reverse_shared/
│   ├── addition_shared/
│   ├── reverse/
│   ├── addition/
│   ├── composed/
│   ├── addition_scratchpad/
│   └── test/
├── composition_strategies/
│   ├── config/
│   │   ├── train_reverse.py
│   │   ├── train_addition.py
│   │   ├── train_composed.py
│   │   ├── train_addition_ft.py
│   │   └── train_addition_scratchpad.py
│   ├── compose.py
│   ├── create_val_txt.py
│   ├── evaluate_all.py
│   ├── evaluate_composition.py
│   ├── evaluate_composed.py
│   ├── evaluate_reverse.py
│   ├── evaluate_addition.py
│   ├── evaluate_addition_scratchpad.py
│   ├── task_vectors.py
│   ├── train_concat.py
│   └── train_adapter.py
├── ties.py
├── dare.py
├── task_arithmetic.py
├── task_vector_conflict.py
├── cka.py
├── evaluate_stratified_composition.py
├── check_duplicates.py
└── check_test_leakage.py
```

## What Is Here

### Shared-base task-vector track

This track uses a shared vocabulary and a shared base model, then fine-tunes to produce task-specific checkpoints for reverse and addition.

Primary files:

- `data/prepare_mixed.py` builds `data/mixed/`, `data/reverse_shared/`, `data/addition_shared/`, and held-out test sets in `data/test/`
- `config/train_base.py` trains the shared base model from scratch
- `config/train_reverse_ft.py` fine-tunes the reverse checkpoint
- `config/train_addition_ft_base.py` fine-tunes the addition checkpoint
- `ties.py` evaluates TIES-style merging and related composite benchmarks
- `dare.py` evaluates DARE-style merging and related composite benchmarks
- `task_arithmetic.py` runs task-vector arithmetic baselines
- `task_vector_conflict.py` measures weight-space conflict statistics
- `cka.py` computes representation similarity
- `evaluate_stratified_composition.py` runs a difficulty-stratified composite benchmark

### Composition-strategy track

This track trains reverse, addition, and composed models from scratch, then evaluates different ways of chaining or combining them.

Primary files:

- `data/prepare_tokenized.py` builds the scratch datasets
- `data/generate_addition_test.py` and `data/generate_composed_test.py` create held-out evaluation sets
- `composition_strategies/config/train_reverse.py`, `train_addition.py`, and `train_composed.py` train the scratch models
- `composition_strategies/evaluate_composed.py` evaluates the pipeline baseline
- `composition_strategies/train_concat.py` and `composition_strategies/evaluate_composition.py` cover layer concatenation
- `composition_strategies/train_adapter.py` and `composition_strategies/evaluate_composition.py` cover the adapter approach
- `composition_strategies/evaluate_all.py` is the convenience driver for the full set of composition experiments
- `composition_strategies/evaluate_reverse.py`, `evaluate_addition.py`, and `evaluate_addition_scratchpad.py` are task-specific evaluation helpers
- `composition_strategies/task_vectors.py` contains supporting utilities for the composition experiments

## Dependencies

- Python 3.10 or newer
- `torch`
- `numpy`
- An external nanoGPT checkout

Most scripts expect nanoGPT to be available locally and many use `NANOGPT_CONFIG` to point at `configurator.py`.

Example:

```bash
export NANOGPT_CONFIG=/Users/sonnguyen/comp560-nanoGPT/configurator.py
```

Some scripts also hard-code the original nanoGPT path in the command examples. If your checkout lives elsewhere, replace that path consistently.

## Typical Workflow

### 1. Build datasets

Shared-base track:

```bash
python data/prepare_mixed.py
```

Scratch track:

```bash
python data/prepare_tokenized.py
python data/generate_addition_test.py
python data/generate_composed_test.py
```

Optional scratchpad data:

```bash
python data/prepare_addition_scratchpad.py
```

### 2. Train models

Shared-base model:

```bash
NANOGPT_CONFIG=/Users/sonnguyen/comp560-nanoGPT/configurator.py \
  python /Users/sonnguyen/comp560-nanoGPT/train.py config/train_base.py
```

Then copy the base checkpoint before fine-tuning:

```bash
mkdir -p out/reverse_ft out/addition_ft
cp out/base/ckpt.pt out/reverse_ft/ckpt.pt
cp out/base/ckpt.pt out/addition_ft/ckpt.pt
```

Fine-tuning:

```bash
NANOGPT_CONFIG=/Users/sonnguyen/comp560-nanoGPT/configurator.py \
  python /Users/sonnguyen/comp560-nanoGPT/train.py config/train_reverse_ft.py

NANOGPT_CONFIG=/Users/sonnguyen/comp560-nanoGPT/configurator.py \
  python /Users/sonnguyen/comp560-nanoGPT/train.py config/train_addition_ft_base.py
```

Scratch models:

```bash
NANOGPT_CONFIG=/Users/sonnguyen/comp560-nanoGPT/configurator.py \
  python /Users/sonnguyen/comp560-nanoGPT/train.py composition_strategies/config/train_reverse.py

NANOGPT_CONFIG=/Users/sonnguyen/comp560-nanoGPT/configurator.py \
  python /Users/sonnguyen/comp560-nanoGPT/train.py composition_strategies/config/train_addition.py

NANOGPT_CONFIG=/Users/sonnguyen/comp560-nanoGPT/configurator.py \
  python /Users/sonnguyen/comp560-nanoGPT/train.py composition_strategies/config/train_composed.py
```

### 3. Evaluate

Task-vector methods:

```bash
python ties.py --sweep --n 200 --device mps
python dare.py --sweep --n 200 --device mps
python task_arithmetic.py --mode sweep --n 200 --device mps
python evaluate_stratified_composition.py --samples_per_level 50 --device auto
```

Composition strategies:

```bash
python composition_strategies/evaluate_composed.py --strategy pipeline --n 200 --device mps
python composition_strategies/train_concat.py --epochs 50 --device mps
python composition_strategies/evaluate_composition.py --strategy concat --n 200 --device mps
python composition_strategies/train_adapter.py --epochs 100 --device mps
python composition_strategies/evaluate_composition.py --strategy adapter --n 200 --device mps
python composition_strategies/evaluate_all.py --n 200 --device mps
```

## Data Notes

- Digits are space-separated to make token positions explicit.
- The shared-base track uses a single vocabulary across `data/mixed/`, `data/reverse_shared/`, and `data/addition_shared/`.
- The scratch track uses separate datasets under `data/reverse/`, `data/addition/`, and `data/composed/`.
- `data/addition_scratchpad/` stores the scratchpad variant of the addition dataset.
- A legacy/generated copy of the scratchpad data also appears under `data/data/addition_scratchpad/` in some runs; treat it as an artifact, not a canonical source directory.
- `data/test/` contains held-out reverse and addition tests.

## Results Summary

The checked-in result files point to a consistent pattern:

- TIES preserves the source tasks well. In [ties_results_table.md](ties_results_table.md), the strongest runs reach 93-94% reverse accuracy and up to 100% addition accuracy, while the naive merge is 0% on both tasks.
- DARE is similarly strong on the source tasks. In [dare_results_table_no_rescale.csv](dare_results_table_no_rescale.csv), the best run reaches 95.0% reverse and 99.5% addition at `drop_rate=0.8`, `lam=0.2`.
- Composition remains the hard part. In [composite_benchmark_d08_l02.csv](composite_benchmark_d08_l02.csv), composite accuracy is 0-1% across the base, naive, and TIES variants.
- The stratified benchmark shows the same pattern. In [stratified_composition_results.csv](stratified_composition_results.csv), the level-2 and level-3 accuracies are 0.0% for the main models, and the overall scores stay low.

Artifacts such as `ties_results_table.md`, `dare_results_table.csv`, `composite_benchmark_*.csv`, and `stratified_composition_results*.csv` are generated outputs. They are useful for analysis, but they are not required source files.

## Known Gotchas

- Several scripts assume the nanoGPT checkout is at the original local path used during development.
- Many evaluation scripts default to `mps`; switch to `cpu` or `cuda` as needed.
- For the shared-base track, copy `out/base/ckpt.pt` into both fine-tuning directories before running the fine-tune configs.
- `check_duplicates.py` and `check_test_leakage.py` are validation utilities, not training scripts.

## Related Work

- Task arithmetic for model editing in weight space
- TIES for resolving merge interference
- DARE for sparsified task-vector merging
- Compositional reasoning and modularity in small transformers

## Citation

This project is inspired by:

```bibtex
@article{ilharco2023editing,
  title={Editing Models with Task Arithmetic},
  author={Ilharco, Gabriel and Ribeiro, Marco Tulio and Wortsman, Mitchell and Schmidt, Ludwig and Hajishirzi, Hannaneh and Farhadi, Ali},
  journal={ICLR},
  year={2023}
}

@article{yadav2023ties,
  title={TIES-Merging: Resolving Interference When Merging Models},
  author={Yadav, Prateek and Tam, Derek and Choshen, Leshem and Raffel, Colin and Bansal, Mohit},
  journal={arXiv preprint arXiv:2306.01708},
  year={2023},
  url={https://arxiv.org/abs/2306.01708}
}

@article{yu2023supermario,
  title={Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch},
  author={Yu, Le and Yu, Bowen and Yu, Haiyang and Huang, Fei and Li, Yongbin},
  journal={arXiv preprint arXiv:2311.03099},
  year={2023},
  url={https://arxiv.org/abs/2311.03099}
}
```

Built on top of nanoGPT by Andrej Karpathy. Project for COMP 560.
