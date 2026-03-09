# Fine-tune reverse task from shared base checkpoint (Week 1 experiment).
#
# IMPORTANT: copy base checkpoint before running:
#   mkdir -p out/reverse_ft && cp out/base/ckpt.pt out/reverse_ft/ckpt.pt
#
# Uses data/reverse_shared/ which is tokenized with the SAME unified vocabulary
# as the base model (vocab_size matches the base checkpoint).

out_dir = 'out/reverse_ft'

eval_interval = 100
eval_iters    = 20
log_interval  = 10

always_save_checkpoint = True

wandb_log      = True
wandb_project  = 'transformer-algebra'
wandb_run_name = 'week1-reverse-ft'

dataset    = 'reverse_shared'
batch_size = 64
block_size = 48

gradient_accumulation_steps = 1

n_layer  = 4
n_head   = 4
n_embd   = 128
dropout  = 0.0

# Lower LR for fine-tuning to preserve base representations.
# max_iters must be base_iters(20000) + ft_iters(5000) because nanoGPT
# resumes iter_num from the checkpoint and stops when iter_num >= max_iters.
learning_rate  = 1e-3
max_iters      = 25000
lr_decay_iters = 5000
min_lr         = 1e-4
beta2          = 0.99
warmup_iters   = 50

init_from = 'resume'

device  = 'mps'
compile = False
