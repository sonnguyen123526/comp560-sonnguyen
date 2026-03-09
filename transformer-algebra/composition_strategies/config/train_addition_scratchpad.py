# Train scratchpad addition model.
# Trains on 2-digit numbers; OOD test is 3-digit numbers.
# Scratchpad format: "2 3 + 4 5 -> C:0 8 C:0 6 -> 6 8"

out_dir = 'out/addition_scratchpad'

eval_interval = 100
eval_iters    = 20
log_interval  = 1

always_save_checkpoint = False

wandb_log      = True
wandb_project  = 'transformer-algebra'
wandb_run_name = 'addition-scratchpad'

dataset    = 'addition_scratchpad'
batch_size = 64
block_size = 128   # scratchpad traces are longer than plain addition

gradient_accumulation_steps = 1

n_layer  = 4
n_head   = 4
n_embd   = 128
dropout  = 0.1  # regularization — dataset is small even with all pairs

learning_rate  = 3e-3
max_iters      = 5000
lr_decay_iters = 5000
min_lr         = 1e-4
beta2          = 0.99
warmup_iters   = 200

device  = 'mps'
compile = False
