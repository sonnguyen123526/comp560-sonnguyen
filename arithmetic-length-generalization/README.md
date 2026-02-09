# Arithmetic Length Generalization

Testing if transformers can learn addition on short numbers and generalize to longer ones. Based on "How Far Can Transformers Reason? The Globality Barrier and Inductive Scratchpad" (Anil et al., 2024).

## The Challenge

Can a model trained on 2-3 digit addition solve 10+ digit addition? Normally no, but the paper shows two scratchpad formats that enable length generalization by making the algorithm explicit.

## Two Approaches

### Random Spaces Method

Embeds numbers with random underscores, then shows explicit step-by-step computation with position pointers.

**Input:**
```
94_+_3__1=
```

**Scratchpad output:**
```
<START>[01]4[08]1c0r5$xgwg#[00]9[05]3c1r25$xgw#[-1]_[03]_c0r125$xg<EOS>
```

Reading this:
- `[01]` and `[08]` are pointers to digit positions
- `c0`, `c1` are carry values
- `r5` is the running result (builds right-to-left)
- Random text (`$xgwg`) forces position-invariant learning

Paper result: Train on 10 digits → test on 18-20 digits ✓

### Cyclic Shifts Method

Uses cyclic rotation with explicit state tracking at each step.

**Input:**
```
fs$46+ih$98=
```
Each number gets a random prefix before the `$` marker.

**Scratchpad output:**
```
$kckn|0#6fs$4+8ih$9=4$kck|1#46fs$+98ih$=44$kc|1#144$kc<EOS>
```

Reading this:
- States separated by `#`
- Each state: `x[i]+y[i]=ans[i]|c[i]`
- Numbers rotate right each step (cyclic shift)
- `$` marker moves through the computation

Paper result: Train on 4 digits → test on 26-30 digits ✓

## Project Structure

```
arithmetic-length-generalization/
├── README.md
├── config/
│   ├── random_spaces.py         # Config for random spaces method
│   └── cyclic_shifts.py         # Config for cyclic shifts method
├── data/
│   ├── generate_random_space.py # Generate random spaces dataset
│   ├── generate_using_shifts.py # Generate cyclic shifts dataset
│   ├── prepare_tokenized.py     # Tokenize datasets for training
│   ├── random_spaces/           # Generated data (txt + bin)
│   │   ├── train.txt
│   │   ├── train.bin
│   │   ├── val.bin
│   │   └── meta.pkl
│   └── cyclic_shifts/           # Generated data (txt + bin)
│       ├── train.txt
│       ├── train.bin
│       ├── val.bin
│       └── meta.pkl
├── out/                         # Model checkpoints saved here
└── wandb/                       # Training logs
```

## Usage

### Generate datasets

```bash
cd data

# Random spaces method
python generate_random_space.py

# Cyclic shifts method  
python generate_using_shifts.py

# Tokenize both
python prepare_tokenized.py

cd ..
```

### Train models

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

### Test on longer numbers

After training, generate test data with more digits and run inference to check if the model generalizes.

## Why This Works

The key insight: explicitly encoding algorithmic structure helps models learn the underlying computation rather than memorizing patterns.

- **Position tracking** (pointers/shifts) breaks positional bias
- **Random padding** forces position-invariant representations  
- **Step-by-step states** show the recursive structure
- Model learns the algorithm, not just the training distribution

## Comparison to Simple Scratchpad

The `arithmetic-scratchpad/` folder has a simpler format:
```
25+37->5+7=2c1,2+3+1=6->62
```

That's easier to train but doesn't generalize to longer inputs. These methods trade simplicity for length generalization.

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
