# COMP 560 - Liberating Chain-of-thought Reasoning in Large Language Models

Experiments with small GPT models on character-level tasks and arithmetic reasoning, using Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).

## Projects

### 1. Arithmetic with Scratchpad ([arithmetic-scratchpad/](arithmetic-scratchpad/))

Testing if transformers learn better when shown intermediate reasoning steps during addition.

**The idea:** Compare a baseline model (direct `123+456->579`) vs. a scratchpad model that shows step-by-step work (`123+456->3+6=9,2+5=7,1+4=5->579`).

- Train on 2-3 digit addition with 50k samples
- Both models use identical architecture (4 layers, 4 heads, 128d)
- Based on "How Far Can Transformers Reason?" (Anil et al., 2024)

See [arithmetic-scratchpad/README.md](arithmetic-scratchpad/README.md) for details.

### 2. Arithmetic Length Generalization ([arithmetic-length-generalization/](arithmetic-length-generalization/))

Testing if transformers can generalize from short to long numbers using complex scratchpad formats from the paper.

**Two methods:**
- **Random spaces:** Embeds numbers with underscores and position pointers
- **Cyclic shifts:** Uses cyclic rotation with explicit state tracking

Challenge: Train on 2-3 digits, test on 10+ digits. The paper shows these formats enable length generalization.

See [arithmetic-length-generalization/README.md](arithmetic-length-generalization/README.md) for details.

### 3. Insert Spaces ([insert-spaces/](insert-spaces/))

Early experiment training GPT to insert spaces between letters.

- Task: "hello" → "h e l l o"
- Dataset: 3,000+ words (3-8 letters)
- Model: 4-layer GPT (0.79M params)
- Result: Model struggled (val loss: 1.17)

See [insert-spaces/README.MD](insert-spaces/README.MD) for analysis.

### 4. Vietnamese-English Translation ([translation/](translation/))

Training GPT to translate Vietnamese numbers (0-20) to English.

- Dataset: 50,000 pairs ("một" → "one")
- Model: 6-layer GPT (2M params)
- Result: Successfully translates most numbers (val loss: 0.80)

See [translation/README.MD](translation/README.MD) for analysis.

## Repository Structure

```
comp560-sonnguyen/
├── README.md                         # This file
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
│   └── seq2seq_model_testing/       # Alternative seq2seq approach
├── translation/                      # Vietnamese-English number translation
│   ├── config/
│   └── data/
├── assets/                           # Plots and figures
└── wandb/                            # Shared training logs
```

## Quick Start

### Arithmetic Experiments

```bash
# Generate and tokenize data (combined script)
cd arithmetic-scratchpad/data
python prepare_tokenized.py
cd ../..

# Train baseline (no scratchpad)
cd arithmetic-scratchpad
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python -u ../../comp560-nanoGPT/train.py config/without_scratchpad.py

# Train with scratchpad
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python -u ../../comp560-nanoGPT/train.py config/with_scratchpad.py
```

### Length Generalization

```bash
# Generate datasets
cd arithmetic-length-generalization/data
python generate_random_space.py      # Random spaces method
python generate_using_shifts.py      # Cyclic shifts method
python prepare_tokenized.py          # Tokenize both
cd ../..

# Train models
cd arithmetic-length-generalization
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python -u ../../comp560-nanoGPT/train.py config/random_spaces.py
```

### Sample from trained model

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python -u ../../comp560-nanoGPT/sample.py config/with_scratchpad.py \
  --num_samples=10 --max_new_tokens=200
```

## Key Findings

**Architecture matters:**
- Larger models (6 layers, 192d) outperform smaller ones (4 layers, 128d)
- Dropout (0.1) and warmup help stabilization

**Data format matters:**
- Simple formats work better than verbose ones
- Scratchpad shows promise for multi-step reasoning
- Position-invariant encoding helps length generalization

**Evaluation is crucial:**
- Low loss doesn't guarantee task success
- Must check actual outputs for correctness

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- [nanoGPT](https://github.com/karpathy/nanoGPT)

Set `NANOGPT_CONFIG` environment variable to point to nanoGPT's configurator.

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

## License

MIT License - see [LICENSE](LICENSE) for details
