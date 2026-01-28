# comp560-sonnguyen

# Training Command
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py  python -u ../../comp560-nanoGPT/train.py config/basic.py

# Sampling Command

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py  python -u ../../comp560-nanoGPT/sample.py config/basic.py --num_samples=1 --max_new_tokens=100 --seed=2
