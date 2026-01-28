# Insert Spaces Between Characters

**Goal:** Train GPT to insert spaces between letters (e.g., "hello" → "h e l l o")

## Setup

**Dataset:**
- 3,000+ unique words (3-8 letters, a-z)
- Format: `Input: hello Output: h e l l o`
- ~1M characters total

**Generate data:**
```bash
cd data/basic && python prepare.py
```

**Train model:**
```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python -u ../../comp560-nanoGPT/train.py config/basic.py
```

**Sample from trained model:**
```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python -u ../../comp560-nanoGPT/sample.py config/basic.py --num_samples=3 --max_new_tokens=100 --seed=2
```

---

## Experiment

**Model:** 4-layer GPT (128-dim embeddings, 4 heads)
- Batch size: 12
- Learning rate: 1e-3
- Dropout: 0.0
- **Iterations: 2,000** (no warmup)
- Training time: ~1-2 hours on CPU
- Model parameters: 0.79M

**Training Results:**
| Iteration | Train Loss | Val Loss |
|-----------|-----------|----------|
| 1000 | 1.14 | 1.17 |
| 2000 | 1.14 | 1.17 |

Model size: 9.6 MB

**Sample Outputs (seed=2):**
```
Input: zwlsmdz Output: r w l g m s f    (Expected: z w l s m d z)
Input: zwvs Output: z w j d             (Expected: z w v s)
Input: mykkn Output: m v i k n          (Expected: m y k k n)
Input: pqreky Output: e q r d           (Expected: p q r e k y)
```

---

## Conclusions

### What Didn't Work
- **Model fails at task** - Generates wrong characters instead of just adding spaces
- **Loss plateau insufficient** - Loss of 1.14 suggests model never truly learned the pattern
- **Architecture too small** - 4 layers/128 dims may be inadequate for this task
- **No dropout harmful** - Zero dropout didn't help; model neither memorized nor generalized
- **Format confusion** - Model may be confused by "Input:" and "Output:" format instead of learning the transformation

### What Worked ✓
- **Training stability** - Loss converged and stayed stable (not diverging)
- **Character-level processing** - Model can process character sequences
- **Efficient size** - 9.6 MB model is compact and trains quickly on CPU

### Key Insights
- Loss alone doesn't tell the full story - need to check actual outputs
- This task requires either: (1) larger model, (2) different architecture, or (3) simpler data format
- Current approach treats it as language modeling, may need sequence-to-sequence architecture instead

---

## File Structure

```
insert-spaces/
├── README.MD              # This file
├── config/
│   └── basic.py          # Model hyperparameters (4 layers, 128 embd, etc.)
├── data/
│   └── basic/
│       ├── prepare.py    # Generates dataset with space-insertion examples
│       ├── train.bin     # Encoded training data (~1M chars)
│       ├── val.bin       # Encoded validation data (same as train)
│       └── meta.pkl      # Character vocabulary + encode/decode functions
└── out/
    └── basic/
        └── ckpt.pt       # Trained model checkpoint (9.6 MB)
```
