# String reversal config
# Task: hello->olleh, world->dlrow, etc.
# NOTE: Train on Google Colab GPU, not local CPU (too slow!)

out_dir = 'out/reverse'
eval_interval = 100
eval_iters = 20
log_interval = 1

always_save_checkpoint = False

# wandb tracking
wandb_log = True
wandb_project = 'transformer-algebra'
wandb_run_name = 'reverse'

dataset = 'reverse'
gradient_accumulation_steps = 1
batch_size = 64  # same as addition config
block_size = 32  # reversal strings are also pretty short

# same architecture as addition for fair comparison
n_layer = 6
n_head = 6
n_embd = 192
dropout = 0.1

# same training setup as addition
learning_rate = 1e-3
max_iters = 2000
lr_decay_iters = 2000
min_lr = 1e-4  
beta2 = 0.99

warmup_iters = 100

# GPU REQUIRED - don't run on CPU!
device = 'cuda'
compile = True
