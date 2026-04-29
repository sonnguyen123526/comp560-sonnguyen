# COMP 560: Small Transformer Experiments

This repository is a collection of controlled experiments with small GPT-style models. The common theme is simple but important: can tiny transformers learn algorithmic structure, and what happens when we change the training signal, the data format, or the way independently trained models are combined?

The code is organized as a set of self-contained projects. Each subdirectory has its own README with setup, scripts, and experiment-specific results.

## Projects

### `transformer-algebra/`

Model-composition experiments built around reverse, addition, and reverse-then-add tasks.

This folder has two complementary tracks:

1. Shared-base task-vector experiments: base training plus fine-tuning, then merge methods such as TIES, DARE, and task arithmetic.
2. Composition-strategy experiments: pipeline, layer concatenation, and adapter-style composition trained from scratch.

Representative findings:

- Single-task performance can be strong after fine-tuning or merging.
- Preserving one task does not imply that the composed task works.
- Composite reverse-then-add accuracy remains the hard case in the benchmark suite.

See [transformer-algebra/README.md](transformer-algebra/README.md) for the full experiment map, scripts, and recorded results.

### `arithmetic-scratchpad/`

Addition with and without an inductive scratchpad.

This project tests whether explicit intermediate steps help a transformer learn column-wise addition:

- Baseline format: `123+456->579`
- Scratchpad format: `123+456->3+6=9,2+5=7,1+4=5->579`

The implementation follows the ideas in *How Far Can Transformers Reason? The Globality Barrier and Inductive Scratchpad*.

See [arithmetic-scratchpad/README.md](arithmetic-scratchpad/README.md).

### `arithmetic-length-generalization/`

Scratchpad-style addition designed to test length generalization.

This project explores two algorithmic representations from the paper:

- Random spaces with explicit position pointers
- Cyclic shifts with stepwise state tracking

The focus here is not only fitting short examples, but generalizing to much longer arithmetic strings at test time.

See [arithmetic-length-generalization/README.md](arithmetic-length-generalization/README.md).

### `insert-spaces/`

A character-level spacing task: turn `hello` into `h e l l o`.

This experiment is useful as a small negative control. The model can fit the data distribution, but the generated outputs do not solve the actual transformation reliably. The subproject also includes a `seq2seq_model_testing/` directory for an alternative architecture path.

See [insert-spaces/README.MD](insert-spaces/README.MD).

### `translation/`

Vietnamese-to-English number translation.

This is the clearest success case in the repository. A larger GPT configuration with dropout and warmup learns most of the mapping and generalizes reasonably well on held-out examples.

See [translation/README.MD](translation/README.MD).

## Repository Layout

```text
comp560-sonnguyen/
├── README.md
├── LICENSE
├── assets/                         # Plots and figures
├── transformer-algebra/            # Composition, merging, and diagnostics
├── arithmetic-scratchpad/          # Addition with/without explicit scratchpads
├── arithmetic-length-generalization/ # Length-generalization experiments
├── insert-spaces/                  # Character spacing task and seq2seq tests
└── translation/                    # Vietnamese-English number translation
```

Each experiment directory follows the same general pattern:

- `config/` for nanoGPT training configs
- `data/` for dataset generation and tokenization scripts
- `out/` for checkpoints, when present locally
- `wandb/` for logs, when used

Generated datasets, checkpoints, and evaluation artifacts are intentionally not treated as source-of-truth files. Most of them are reproducible from the scripts in each project and are ignored by git.

## Setup

These experiments depend on an external nanoGPT checkout. Most scripts expect a local path to `configurator.py`, typically supplied through `NANOGPT_CONFIG`.

Example:

```bash
export NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py
```

From there, follow the README in the relevant subdirectory. The workflow is usually:

1. Generate or tokenize data in `data/`
2. Train with the matching config in `config/`
3. Sample or evaluate using the project-specific scripts

## Notes On Scope

- This repo is a research notebook in code form, not a single unified application.
- The root README is intentionally high level; task-specific details belong in the subproject READMEs.
- `transformer-algebra/` contains the richest set of analysis scripts and generated tables.
- `translation/` and `insert-spaces/` are smaller, self-contained language-modeling experiments.

## License

MIT
