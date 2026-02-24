out_dir = 'out/addition'

eval_interval = 100
eval_iters = 20
log_interval = 1

always_save_checkpoint = False

wandb_log = True
wandb_project = 'transformer-algebra'
wandb_run_name = 'addition' 


dataset = 'addition'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 32

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
