# Transformer Algebra

This folder explores whether small transformer models trained on simple tasks can be composed to solve a harder task. The primary example is reverse then add:

- Reverse: "1 2 3" -> "3 2 1"
- Addition: "1 2 3 + 4 5 6" -> "5 7 9"
- Composed: "1 2 3 + 4 5 6" -> "3 2 1 + 6 5 4" -> "9 7 5"

There are two parallel tracks:

1) Task-vector and model-merging experiments (TIES, DARE, naive merge) built on a shared base model and a shared vocabulary.
2) Composition strategies (pipeline, layer concatenation, adapter) trained from scratch on separate datasets.

## What is in here

High-level structure (files listed here exist in the repo):

```
transformer-algebra/
├── config/                         # Shared-base training configs (task-vector track)
│   ├── train_base.py
│   ├── train_reverse_ft.py
│   └── train_addition_ft_base.py
├── data/
│   ├── prepare_mixed.py            # Shared vocab datasets + held-out tests
│   ├── prepare_tokenized.py        # Independent reverse/addition/composed datasets
│   ├── prepare_addition_scratchpad.py
│   ├── generate_addition_test.py
│   ├── generate_composed_test.py
│   ├── mixed/ reverse_shared/ addition_shared/ test/
│   ├── reverse/ addition/ composed/
│   └── addition_scratchpad/
├── composition_strategies/         # Pipeline, concat, adapter experiments
│   ├── config/                     # nanoGPT configs for scratch models
│   ├── compose.py
│   ├── train_concat.py
│   ├── train_adapter.py
│   ├── evaluate_composed.py
│   ├── evaluate_composition.py
│   ├── evaluate_reverse.py
│   ├── evaluate_addition.py
│   └── task_vectors.py
├── ties.py                         # TIES merge + composite eval
├── dare.py                         # DARE merge + composite eval
├── task_arithmetic.py              # Task arithmetic on shared-base checkpoints
├── task_vector_conflict.py         # Cosine/sign conflict diagnostics
├── cka.py                          # CKA representation similarity
├── evaluate_stratified_composition.py
├── check_duplicates.py
└── check_test_leakage.py
```

## Dependencies and setup

- Python 3.10+ with `torch`, `numpy`, `pickle`, `argparse`.
- nanoGPT is required. Some scripts hard-code an absolute path, while others
  auto-discover `~/comp560-nanoGPT`.

If your nanoGPT checkout is elsewhere:

1) Put nanoGPT at `~/comp560-nanoGPT` (works with the auto-discovery scripts), and
2) Update the remaining hard-coded paths to your local path.

Quick fix: search for `comp560-nanoGPT` and replace it with your local path.
On Windows, use a double-backslash in string literals, for example
`C:\\Users\\you\\comp560-nanoGPT`.

## Track A: Shared-base task vectors (TIES, DARE)

This track relies on a shared base model and a single unified vocabulary across all tasks.

### 1) Build the shared datasets

```bash
python data/prepare_mixed.py
```

This creates:

- `data/mixed/` for base training
- `data/reverse_shared/` and `data/addition_shared/` for fine-tuning
- `data/test/` held-out test sets for reverse and addition

### 2) Train base and fine-tuned models

```bash
NANOGPT_CONFIG=/Users/sonnguyen/comp560-nanoGPT/configurator.py \
  python /Users/sonnguyen/comp560-nanoGPT/train.py config/train_base.py

# Copy base checkpoint before fine-tuning
mkdir -p out/reverse_ft out/addition_ft
cp out/base/ckpt.pt out/reverse_ft/ckpt.pt
cp out/base/ckpt.pt out/addition_ft/ckpt.pt

NANOGPT_CONFIG=/Users/sonnguyen/comp560-nanoGPT/configurator.py \
  python /Users/sonnguyen/comp560-nanoGPT/train.py config/train_reverse_ft.py

NANOGPT_CONFIG=/Users/sonnguyen/comp560-nanoGPT/configurator.py \
  python /Users/sonnguyen/comp560-nanoGPT/train.py config/train_addition_ft_base.py
```

### 3) Run task-vector experiments

TIES sweep and single run:

```bash
python ties.py --sweep --n 200 --device mps
python ties.py --density 0.6 --lam 0.4 --n 200 --device mps
```

Composite benchmark and failure analysis:

```bash
python ties.py --composite_eval --n 200 --device mps
python ties.py --composite_failure_analysis --n 200 --device mps
```

DARE sweep and composite eval:

```bash
python dare.py --sweep --n 200 --device mps
python dare.py --composite_eval --drop_rate 0.4 --lam 0.6 --n 200 --device mps
```

Task arithmetic baselines:

```bash
python task_arithmetic.py --mode sweep --n 200 --device mps
python task_arithmetic.py --mode negate --n 200 --device mps
```

Diagnostics:

```bash
python task_vector_conflict.py
python cka.py --n_prompts 200
```

Stratified composite evaluation (3 difficulty levels):

```bash
python evaluate_stratified_composition.py --samples_per_level 50 --device auto
```

## Track B: Composition strategies (pipeline, concat, adapter)

This track trains reverse/addition/composed models from scratch with their own vocabularies, then evaluates different composition strategies.

### 1) Build datasets

```bash
python data/prepare_tokenized.py
python data/generate_addition_test.py
python data/generate_composed_test.py
```

Optional scratchpad data:

```bash
python data/prepare_addition_scratchpad.py
```

### 2) Train scratch models

```bash
NANOGPT_CONFIG=/Users/sonnguyen/comp560-nanoGPT/configurator.py \
  python /Users/sonnguyen/comp560-nanoGPT/train.py composition_strategies/config/train_reverse.py

NANOGPT_CONFIG=/Users/sonnguyen/comp560-nanoGPT/configurator.py \
  python /Users/sonnguyen/comp560-nanoGPT/train.py composition_strategies/config/train_addition.py

NANOGPT_CONFIG=/Users/sonnguyen/comp560-nanoGPT/configurator.py \
  python /Users/sonnguyen/comp560-nanoGPT/train.py composition_strategies/config/train_composed.py
```

### 3) Evaluate the pipeline baseline

```bash
python composition_strategies/evaluate_composed.py --strategy pipeline --n 200 --device mps
python composition_strategies/evaluate_composed.py --strategy baseline --n 200 --device mps
```

### 4) Train and evaluate composition models

Layer concatenation:

```bash
python composition_strategies/train_concat.py --epochs 50 --device mps
python composition_strategies/evaluate_composition.py --strategy concat --n 200 --device mps
```

Adapter network:

```bash
python composition_strategies/train_adapter.py --epochs 100 --device mps
python composition_strategies/evaluate_composition.py --strategy adapter --n 200 --device mps
```

### 5) Quick summary script

```bash
python composition_strategies/evaluate_all.py --n 200 --device mps
```

## Data format notes

- Digits are space-separated to make positions explicit.
- The shared-base track uses a unified vocabulary across all datasets in `data/mixed`, `data/reverse_shared`, and `data/addition_shared`.
- The scratch track uses independent vocabularies in `data/reverse`, `data/addition`, and `data/composed`.

## Results and artifacts

Most evaluation scripts write CSV/TXT summaries into this folder. Common outputs include:

- TIES: `ties_results.txt`, `ties_results_table.md`, `ties_results_table_for_compare.csv`,
  `ties_legacy_vs_canonical.csv`, `ties_sweep_canonical.txt`, `ties_sweep_legacy.txt`,
  `ties_rows_for_compare.txt`.
- DARE: `dare_results.txt`, `dare_results_no_rescale.txt`, `dare_results_table.csv`,
  `dare_results_table_no_rescale.csv`, `dare_rows.txt`, `dare_norescale_rows.txt`,
  `dare_vs_ties_comparison.csv`, `dare_norescale_vs_ties_comparison.csv`.
- Composite benchmarks: `composite_benchmark_*.csv`, `composite_multiseed_*.csv`,
  `composite_failure_samples_*.csv`.
- Stratified composition: `stratified_composition_results.csv`,
  `stratified_composition_results_smoke.csv`.

These files are generated artifacts and can be safely deleted; rerunning the
scripts will recreate them.

## Notable results

- TIES merging at density 0.8, lambda 0.2 reaches 93% reverse and 100% addition, while the naive merge is 0%/0% in the same sweep. See [transformer-algebra/ties_results_table.md](transformer-algebra/ties_results_table.md).
- DARE (no-rescale) is strongest at drop_rate 0.8, lambda 0.2 with 95.0% reverse and 99.5% addition. See [transformer-algebra/dare_results_table_no_rescale.csv](transformer-algebra/dare_results_table_no_rescale.csv).
- Composite evaluation remains near-zero: the benchmark at density 0.8, lambda 0.2 yields 0-1% composite accuracy across base/naive/TIES, and the stratified benchmark shows 0% on L2/L3 for all models. See [transformer-algebra/composite_benchmark_d08_l02.csv](transformer-algebra/composite_benchmark_d08_l02.csv) and [transformer-algebra/stratified_composition_results.csv](transformer-algebra/stratified_composition_results.csv).

## Known gotchas

- Several scripts hard-code the nanoGPT path. Update it or add a symlink if you are not on the original machine.
- Many scripts default to `mps` (Apple Silicon). Use `--device cpu` or `--device cuda` as needed.
- For fine-tuning from the shared base, you must copy the base checkpoint into `out/reverse_ft/` and `out/addition_ft/` before running the fine-tune configs.

## Related work

- TIES-Merging (TIES): model merging with sign-based conflict resolution.
- DARE: dropout-based sparsification of task vectors during merging.
- Task arithmetic: editing/combining models via task vectors in weight space.

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