import numpy as np
import pickle

def decode_lines(data_dir):
    with open(f'{data_dir}/meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    itos = meta['itos']
    tokens = np.fromfile(f'{data_dir}/train.bin', dtype=np.uint16)
    text = ''.join(itos[i] for i in tokens)
    return set(l for l in text.split('\n') if l.strip())

mixed = decode_lines('data/mixed')
rev   = decode_lines('data/reverse_shared')
add   = decode_lines('data/addition_shared')

print('Unique examples per dataset (train split)')
print(f'  mixed:            {len(mixed):,}')
print(f'  reverse_shared:   {len(rev):,}')
print(f'  addition_shared:  {len(add):,}')

cross_rm = mixed & rev
cross_am = mixed & add
cross_ra = rev & add

print()
print('Cross-dataset overlap (train split)')
print(f'  mixed ∩ reverse_shared:          {len(cross_rm):,}')
print(f'  mixed ∩ addition_shared:         {len(cross_am):,}')
print(f'  reverse_shared ∩ addition_shared:{len(cross_ra):,}')

if cross_rm:
    print(f'  sample (mixed∩rev): {list(cross_rm)[:3]}')
if cross_am:
    print(f'  sample (mixed∩add): {list(cross_am)[:3]}')
