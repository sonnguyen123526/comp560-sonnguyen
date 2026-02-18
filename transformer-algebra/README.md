# Transformer Algebra

Training GPT2 to do addition (`123+456=579`) and string reversal (`hello->olleh`).

The main question: can transformers learn algorithmic tasks through pattern matching?

## Two Strategies

**Strategy 1 - Random Init (baseline)**  
Train from scratch on each task independently.

**Strategy 2 - Forking (experimental)**  
First pretrain on mixed data (text + numbers), then fork and finetune for each task.

Does the pretraining help? That's what we're testing.

## Setup

**Important**: Train on Google Colab with GPU. CPU is too slow (hours vs minutes).

Check GPU first:
```bash
python check_gpu.py
```

### 1. Generate Data

```bash
cd data
python generate_addition.py      # 50k addition problems
python generate_reverse.py       # 50k reversal problems
python generate_base_pretrain.py # 20k mixed samples (for strategy 2)

python prepare_tokenized.py addition
python prepare_tokenized.py reverse
python prepare_tokenized.py base_pretrain
```

### 2. Train Models

**Strategy 1 - From Scratch**
```bash
python train.py config/addition.py  # ~5-10 mins on GPU
python train.py config/reverse.py
```

**Strategy 2 - Fork from Base**
```bash
python train.py config/base_pretrain.py    # pretrain first
python train.py config/addition_forked.py   # then fork
python train.py config/reverse_forked.py
```

### 3. Test Output

```bash
python sample.py config/addition.py --num_samples=10
python sample.py config/reverse.py --num_samples=10
```

## Quick Start on Colab

1. Enable GPU: Runtime → Change runtime type → GPU
2. Run `!python check_gpu.py` to verify
3. Clone nanoGPT: `!git clone https://github.com/karpathy/nanoGPT.git`
4. Generate datasets (commands above)
5. Train and sample

Or just run `colab_setup.py` to automate steps 3-4.

## Model Details

**Architecture**: 6 layers, 6 heads, 192d embeddings (~2M params)  
**Training**: batch_size=64, lr=1e-3, 2000 iters (~10 mins on GPU)  
**Data**: Character-level tokenization

**Datasets**:
- Addition: `123+456=579` (50k samples, 1-3 digit numbers)
- Reversal: `hello->olleh` (50k samples, 3-10 char strings)
- Base pretrain: Mixed text/numbers/arithmetic (20k samples)

## What to Expect

**Addition**: Easy for single digits, harder with carrying (789+456). Three-digit addition is the challenge.

**Reversal**: Short strings are easy, accuracy drops with length. 10-char strings might have errors.

**Comparing strategies**: Check if forked models converge faster or reach better accuracy. Maybe they don't and random init is just as good for these tasks!

## Files

```
├── README.md
├── FORKING_GUIDE.md           # Detailed workflow for strategy 2
├── adding.ipynb               # Interactive notebook
├── colab_setup.py             # Automated setup
├── check_gpu.py               # Verify GPU before training
├── compare_strategies.py      # Compare results
├── config/
│   ├── addition.py            # Random init
│   ├── addition_forked.py     # Forked from base
│   ├── reverse.py             # Random init
│   ├── reverse_forked.py      # Forked from base
│   └── base_pretrain.py       # Base model for forking
└── data/
    ├── generate_addition.py
    ├── generate_reverse.py
    ├── generate_base_pretrain.py
    └── prepare_tokenized.py
```

## Ideas to Explore

- Which task is easier for GPT2?
- Can addition model handle 4-digit numbers?
- Does reversal accuracy drop with longer strings?
- What happens with smaller/larger models?
- Can one model learn both tasks?

The main hypothesis: forked models might converge faster or reach better accuracy. Or maybe random init works just as well for algorithmic tasks - that would be interesting too!

