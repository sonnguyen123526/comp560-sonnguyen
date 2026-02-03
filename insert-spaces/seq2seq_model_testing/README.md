# Seq2Seq Model Testing

This folder contains experiments with sequence-to-sequence models for the insert-spaces task, exploring alternatives to the GPT-based approach.

## Overview

The insert-spaces task (transforming "hello" → "h e l l o") is fundamentally a sequence-to-sequence problem. This folder tests whether a simpler Seq2Seq architecture can outperform the GPT model.

## Models

### 1. Simple Seq2Seq (No Attention)
**File:** `simple_seq2seq_model.py`, `train_simple_seq2seq.py`

A minimal encoder-decoder architecture without attention mechanism:
- **Encoder:** Single-layer GRU (64-dim embeddings, 128-dim hidden)
- **Decoder:** Single-layer GRU with same dimensions
- **Parameters:** ~200K (vs 790K for GPT model, ~4M for attention Seq2Seq)
- **Key Features:**
  - No attention mechanism
  - Simple context passing from encoder to decoder
  - Teacher forcing during training

### 2. Seq2Seq with Attention
**File:** `seq2seq_model.py`, `train_seq2seq.py`

Standard encoder-decoder with Bahdanau attention:
- **Encoder:** Bidirectional GRU (256-dim embeddings, 512-dim hidden)
- **Decoder:** GRU with attention mechanism
- **Parameters:** ~4M
- **Key Features:**
  - Attention mechanism to focus on input positions
  - Bidirectional encoding
  - Teacher forcing with decay

## Training

### Simple Seq2Seq
```bash
cd insert-spaces/seq2seq_model_testing
python train_simple_seq2seq.py
```

**Hyperparameters:**
- Embedding dim: 64
- Hidden dim: 128
- Batch size: 64
- Learning rate: 0.001
- Epochs: 100 (with early stopping)
- Dataset: 50,000 samples

### Attention Seq2Seq
```bash
python train_seq2seq.py
```

**Hyperparameters:**
- Embedding dim: 256
- Hidden dim: 512
- Batch size: 32
- Learning rate: 0.001
- Epochs: 50
- Dataset: 50,000 samples

## Results

| Model | Parameters | Val Loss | Val Accuracy | Overfitting |
|-------|-----------|----------|--------------|-------------|
| GPT (4-layer) | 790K | 1.17 | ~20% | High |
| Simple Seq2Seq | ~200K | TBD | TBD | TBD |
| Attention Seq2Seq | ~4M | TBD | TBD | TBD |

## Model Comparison

### Simple Seq2Seq Advantages
- **Smaller model:** Only ~200K parameters (5% of attention model)
- **Task-appropriate:** Direct sequence mapping without complex transformations
- **Less overfitting risk:** Smaller capacity matches simpler task
- **Faster training:** Fewer parameters to update

### Attention Seq2Seq Advantages
- **Better alignment:** Can learn which input position to focus on
- **Proven architecture:** Standard approach for seq2seq tasks
- **Handles variable lengths:** Attention mechanism adapts to input size

### GPT Disadvantages for This Task
- **Wrong architecture:** Autoregressive decoder not designed for seq2seq
- **Overfitting prone:** Too many parameters for simple character manipulation
- **No explicit alignment:** Can't directly map input to output positions

## Files

```
seq2seq_model_testing/
├── README.md                      # This file
├── simple_seq2seq_model.py       # Simple encoder-decoder (no attention)
├── train_simple_seq2seq.py       # Training script for simple model
├── seq2seq_model.py              # Seq2seq with attention
├── train_seq2seq.py              # Training script with attention
├── sample_simple_seq2seq.py      # Sampling script for simple model
├── sample_seq2seq.py             # Sampling script for attention model
└── out/
    ├── simple_seq2seq/           # Simple model checkpoints
    │   └── best_model.pt
    └── seq2seq/                  # Attention model checkpoints
        └── best_model.pt
```

## Key Features

### Data Handling
- **Shared dataset:** Uses same `data/basic/train.txt` as GPT experiments
- **Special tokens:** `<SOS>`, `<EOS>`, `<PAD>` for sequence handling
- **Vocab:** Character-level (a-z, space, special tokens)

### Training Features
- **Early stopping:** Stops if no improvement for 5 epochs
- **Gradient clipping:** Prevents exploding gradients
- **Teacher forcing:** Helps model learn correct alignments
- **Accuracy tracking:** Reports exact-match accuracy, not just loss

### Evaluation
- **Sample generation:** Tests on specific words (hello, world, test, etc.)
- **Exact-match accuracy:** % of sequences perfectly predicted
- **Overfitting detection:** Monitors train-val loss gap

## Usage Example

```python
# Load trained model
checkpoint = torch.load('out/simple_seq2seq/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Generate prediction
input_word = "hello"
src_indices = dataset.encode(input_word)
src_tensor = torch.tensor([src_indices]).to(device)
output = model.generate(src_tensor, max_len=20, sos_idx=dataset.sos_idx, eos_idx=dataset.eos_idx)
prediction = dataset.decode(output[0].cpu().numpy())
# Expected: "h e l l o"
```

## Next Steps

1. **Compare results** between simple and attention models
2. **Analyze failure cases** - which words are hardest?
3. **Try curriculum learning** - start with short words, increase length
4. **Experiment with dropout** - does regularization help?
5. **Test on longer sequences** - how well does it generalize?

## Requirements

- PyTorch
- NumPy
- Python 3.8+

## References

- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
