# Train model to reverse digit sequences: "1 2 3 4" -> "4 3 2 1"
# Trained on 2-4 digit sequences; evaluated on 5-digit (length generalization).

out_dir = 'out/reverse'

eval_interval = 100
eval_iters = 20
log_interval = 1
always_save_checkpoint = False

wandb_log = True
wandb_project = 'transformer-algebra'
wandb_run_name = 'reverse'

dataset = 'reverse'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 48  

n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0  

learning_rate = 3e-3
max_iters = 20000
lr_decay_iters = 20000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 200

device = 'mps' 
compile = False
