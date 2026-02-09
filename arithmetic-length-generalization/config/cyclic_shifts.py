"""
Config for Method 2: Cyclic shifts
Train on 2-3 digits, test length generalization to 5+ digits
"""

out_dir = 'out/cyclic_shifts'
eval_interval = 100
eval_iters = 20
log_interval = 1

always_save_checkpoint = False

wandb_log = True
wandb_project = 'arithmetic-length-gen'
wandb_run_name = 'cyclic-shifts'

dataset = 'cyclic_shifts'
gradient_accumulation_steps = 1
batch_size = 12
block_size = 256  # Longer for complex scratchpad

# Small GPT model (same as your other experiments)
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

learning_rate = 1e-3
max_iters = 2000
lr_decay_iters = 2000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

device = 'cpu'
compile = False
