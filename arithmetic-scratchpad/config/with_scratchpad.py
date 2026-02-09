# train arithmetic model WITH scratchpad (should perform better)
# Same model size as baseline for fair comparison

out_dir = 'out/with_scratchpad'
eval_interval = 100 # keep frequent because we'll overfit
eval_iters = 20
log_interval = 1

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'arithmetic-scratchpad-double-digit'
wandb_run_name = 'with-scratchpad-double-digit'

dataset = 'with_scratchpad'
gradient_accumulation_steps = 1
batch_size = 12
block_size = 128  # Longer to accommodate scratchpad

# very very small GPT model
n_layer = 4
n_head = 4
n_embd = 128  # need n_embd % n_head == 0
dropout = 0.0

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 2000
lr_decay_iters = 2000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 0 # 100 # not super necessary potentially

device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model
