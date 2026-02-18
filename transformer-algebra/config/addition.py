# Addition task config - teaching GPT to do basic math
# Format: 123+456=579
# NOTE: Train on Google Colab GPU, not local CPU (too slow!)

out_dir = 'out/addition'
eval_interval = 100  # check validation every 100 iters
eval_iters = 20
log_interval = 1

always_save_checkpoint = False  # only save when val improves

# wandb setup
wandb_log = True
wandb_project = 'transformer-algebra'
wandb_run_name = 'addition'

dataset = 'addition'
gradient_accumulation_steps = 1
batch_size = 64  # works well on colab GPU, use 12 if on CPU or getting OOM errors
block_size = 32  # addition problems are short so 32 is enough

# model size - keeping it small
n_layer = 6
n_head = 6
n_embd = 192  # has to be divisible by n_head
dropout = 0.1

# training config
learning_rate = 1e-3
max_iters = 2000  # should be plenty for this simple task
lr_decay_iters = 2000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100  # helps with stable training

# GPU REQUIRED - don't run on CPU!
device = 'cuda'
compile = True  # pytorch 2.0 feature, makes it faster
