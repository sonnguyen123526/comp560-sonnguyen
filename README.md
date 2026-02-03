# COMP 560 - NanoGPT Experiments

This repository contains experiments training small GPT models on character-level tasks using Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).

## Projects

### 1. Insert Spaces Between Characters ([insert-spaces/](insert-spaces/))
**Goal:** Train GPT to insert spaces between letters (e.g., "hello" → "h e l l o")

- **Model:** 4-layer GPT (128-dim embeddings, 4 heads, 0.79M params)
- **Dataset:** 3,000+ unique words (3-8 letters, a-z)
- **Result:** Model struggled with task (val loss: 1.17) - see [insert-spaces/README.MD](insert-spaces/README.MD) for full analysis

### 2. Vietnamese-English Number Translation ([translation/](translation/))
**Goal:** Train GPT to translate Vietnamese numbers (0-20) to English

- **Model:** 6-layer GPT (192-dim embeddings, 6 heads, ~2M params)
- **Dataset:** 50,000 translation pairs (e.g., "một" → "one")
- **Result:** Successfully translates most numbers (val loss: 0.80) - see [translation/README.MD](translation/README.MD) for full analysis

## Quick Start

### Training a Model

```bash
# Navigate to project directory (insert-spaces or translation)
cd insert-spaces  # or: cd translation

# Generate dataset
cd data && python prepare.py
cd ..

# Train model
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python -u ../../comp560-nanoGPT/train.py config/basic.py
```

### Sampling from Trained Model

```bash
# Generate samples from trained model
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python -u ../../comp560-nanoGPT/sample.py config/basic.py --num_samples=5 --max_new_tokens=100 --seed=2
```

## Key Findings

**What Worked:**
- Larger architecture (6 layers, 192 dims) performs better than smaller (4 layers, 128 dims)
- Dropout (0.1) and warmup (100 iters) stabilize training
- Simple data format (e.g., `một -> one`) works better than verbose format
- Translation task succeeded where insert-spaces struggled

**What Didn't Work:**
- Small models (4 layers) insufficient for complex character manipulation
- No dropout can hurt generalization
- Loss alone doesn't indicate task success - must check actual outputs

## Repository Structure

```
comp560-sonnguyen/
├── README.md              # This file
├── insert-spaces/         # Character spacing project
│   ├── README.MD         # Detailed analysis
│   ├── config/           # Model hyperparameters
│   ├── data/             # Dataset generation scripts
│   └── out/              # Trained model checkpoints
├── translation/           # Vietnamese-English translation project
│   ├── README.MD         # Detailed analysis
│   ├── config/           # Model hyperparameters
│   ├── data/             # Dataset generation scripts
│   └── out/              # Trained model checkpoints
└── test.py               # WandB integration test
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- [nanoGPT](https://github.com/karpathy/nanoGPT) (configured via `NANOGPT_CONFIG` environment variable)

## License

MIT License - see [LICENSE](LICENSE) for details
