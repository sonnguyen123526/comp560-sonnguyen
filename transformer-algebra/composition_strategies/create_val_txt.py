"""
Create val.txt from val.bin
"""
import pickle
import numpy as np

# Load metadata
with open('data/composed/meta.pkl', 'rb') as f:
    meta = pickle.load(f)
itos = meta['itos']

# Load binary data
with open('data/composed/val.bin', 'rb') as f:
    data = np.memmap(f, dtype=np.uint16, mode='r')

# Decode to text
text = ''.join([itos[int(i)] for i in data])

# Save as text file
with open('data/composed/val.txt', 'w') as f:
    f.write(text)

print(f"Created val.txt from val.bin ({len(text)} characters)")
print("First 200 characters:")
print(text[:200])
