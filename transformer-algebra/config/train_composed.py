# ============================================================================
# COMPOSED TASK CONFIG (Baseline / End-to-End)
# ============================================================================
# This trains a model DIRECTLY on the composed task: reverse-then-add
# Format: 123+456 -> 321+654 -> 975
#
# This is our BASELINE - the thing we're trying to beat!
#
# The big question: Can we achieve similar accuracy by COMPOSING two
# smaller models (reverse + addition), instead of training this big
# end-to-end model?
#
# If composition works well, we could:
#   - Save training time (reuse pretrained models)
#   - Build modular systems (mix and match capabilities)
#   - Better understand what models learn (each does one thing)
# ============================================================================

# Where to save the trained model
out_dir = 'out/composed'

# How often to check progress
eval_interval = 100
eval_iters = 20
log_interval = 1

always_save_checkpoint = False

# wandb setup
wandb_log = True
wandb_project = 'transformer-algebra'
wandb_run_name = 'composed_baseline'  # Clear name: this is the baseline!

# Data configuration
dataset = 'composed'  # Uses data/composed/train.bin
gradient_accumulation_steps = 1
batch_size = 64
block_size = 48  # Slightly longer! Composed format has more tokens:
                 # "123+456->321+654->975" vs "123->321"

# Model architecture (same size as individual models for fair comparison)
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.1

# Training hyperparameters
learning_rate = 1e-3
max_iters = 2000  # Might need more for this harder task, but let's try!
lr_decay_iters = 2000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

# Hardware
device = 'cpu'  # Change to 'cuda' for GPU
compile = False
