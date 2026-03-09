# Train shared base model on mixed reverse+addition data (Week 1 experiment).
#
# This is the foundation for the task arithmetic experiment.  Both the
# reverse_ft and addition_ft models will be fine-tuned starting from this
# checkpoint, giving them a shared representational origin.
#
# Run AFTER: python data/prepare_mixed.py

out_dir = 'out/base'

eval_interval = 250
eval_iters    = 20
log_interval  = 10

always_save_checkpoint = True

wandb_log      = True
wandb_project  = 'transformer-algebra'
wandb_run_name = 'week1-base'

dataset    = 'mixed'
batch_size = 64
block_size = 48

gradient_accumulation_steps = 1

n_layer  = 4
n_head   = 4
n_embd   = 128
dropout  = 0.0

learning_rate  = 3e-3
max_iters      = 20000
lr_decay_iters = 20000
min_lr         = 3e-4
beta2          = 0.99
warmup_iters   = 200

init_from = 'scratch'

device  = 'mps'
compile = False
