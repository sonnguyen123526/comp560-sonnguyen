# ============================================================================
# REVERSAL TASK CONFIG (Task A)
# ============================================================================
# This trains a model to reverse digit sequences: 12345 -> 54321
# It's the first step in our composition experiment!
#
# Example training data:
#   123 -> 321
#   4567 -> 7654
#   98 -> 89
#
# Once trained, this model will be combined with the addition model
# to solve the composed task: reverse-then-add
# ============================================================================

# Where to save the trained model
out_dir = 'out/reverse'

# How often to check progress
eval_interval = 100  # Check validation loss every 100 training steps
eval_iters = 20      # Use 20 batches for validation
log_interval = 1     # Print loss every iteration (we like to watch!)

always_save_checkpoint = False  # Only save when validation improves

# wandb setup (for pretty training graphs and tracking)
wandb_log = True
wandb_project = 'transformer-algebra'
wandb_run_name = 'reverse'  # You'll see this name in wandb dashboard

# Data configuration
dataset = 'reverse'  # Looks for data/reverse/train.bin and val.bin
gradient_accumulation_steps = 1  # Simple: update weights every batch
batch_size = 64      # Process 64 examples at once
block_size = 48      # Max sequence length - spaced format needs more space!
                     # e.g. "1 2 3 4 5 -> 5 4 3 2 1" = 23 chars with spaces

# Model architecture - keeping it SMALL for fast experimentation
# IMPORTANT: Must match addition model for composition to work!
n_layer = 4    # 4 transformer layers (deeper = more capacity, slower)
n_head = 4     # 4 attention heads per layer
n_embd = 128   # 128-dimensional embeddings (wider = more capacity)
dropout = 0.0  # NO dropout! Reversal is a deterministic task - dropout just adds
               # noise and prevents learning. Always use 0.0 for deterministic tasks.

# Training hyperparameters
learning_rate = 3e-3     # Higher LR to converge faster
max_iters = 20000        # Need more iterations for full convergence
lr_decay_iters = 20000   # Decay over all steps
min_lr = 1e-4            # Lowest learning rate (0.0001)
beta2 = 0.99             # Adam optimizer parameter (beta1=0.9 by default)

warmup_iters = 200       # Warmup over first 200 steps

# Hardware configuration
device = 'mps'    # Apple Silicon GPU (fast!). Use 'cpu' if not on Apple Silicon,
                  # or 'cuda' if you have an NVIDIA GPU
compile = False   # PyTorch 2.0 compilation (faster but needs setup)
