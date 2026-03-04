"""
Train adapter network composition model.
Only trains the adapter, both base models stay frozen.
"""
import os
import sys
import time
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from compose import load_model, AdapterComposition

class TextDataset(Dataset):
    def __init__(self, text_file, meta_file, block_size):
        # Load metadata for tokenization
        with open(meta_file, 'rb') as f:
            meta = pickle.load(f)
        self.stoi = meta['stoi']
        
        # Load and tokenize all examples
        with open(text_file, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        
        self.examples = []
        for line in lines:
            # Tokenize the entire line (input + output)
            tokens = [self.stoi[c] for c in line + '\n']
            if len(tokens) <= block_size:
                self.examples.append(tokens)
        
        self.block_size = block_size
        print(f"  Loaded {len(self.examples)} examples from {text_file}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        # Pad if necessary
        if len(tokens) < self.block_size:
            tokens = tokens + [0] * (self.block_size - len(tokens))
        else:
            tokens = tokens[:self.block_size]
        
        tokens = torch.tensor(tokens, dtype=torch.long)
        x = tokens[:-1]
        y = tokens[1:]
        return x, y

def train(model, train_loader, val_loader, args):
    device = args.device
    model = model.to(device)
    
    # Only optimize adapter parameters
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_val_loss = float('inf')
    
    print(f"\nTraining adapter network:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {device}")
    print(f"  Trainable params: {sum(p.numel() for p in model.adapter.parameters()):,}")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, loss = model(x, targets=y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), args.grad_clip)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % args.log_interval == 0:
                print(f"Epoch {epoch+1}/{args.epochs} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, loss = model(x, targets=y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
        
        # Save best adapter
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(args.out_dir, exist_ok=True)
            checkpoint = {
                'adapter': model.adapter.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'adapter_dim': args.adapter_dim
            }
            torch.save(checkpoint, os.path.join(args.out_dir, 'adapter.pt'))
            print(f"  Saved adapter (val_loss={avg_val_loss:.4f})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_a', type=str, default='out/reverse/ckpt.pt')
    parser.add_argument('--model_b', type=str, default='out/addition/ckpt.pt')
    parser.add_argument('--data_dir', type=str, default='data/composed')
    parser.add_argument('--out_dir', type=str, default='out/adapter')
    parser.add_argument('--adapter_dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--log_interval', type=int, default=50)
    args = parser.parse_args()
    
    # Load models
    print("Loading base models...")
    model_A = load_model(args.model_a)
    model_B = load_model(args.model_b)
    
    # Create composition with adapter
    print("\nCreating adapter composition...")
    model = AdapterComposition(model_A, model_B, adapter_dim=args.adapter_dim)
    
    # Load data
    print("\nLoading datasets...")
    block_size = model.model_A.config.block_size
    train_data = TextDataset(
        os.path.join(args.data_dir, 'train.txt'),
        os.path.join(args.data_dir, 'meta.pkl'),
        block_size
    )
    val_data = TextDataset(
        os.path.join(args.data_dir, 'val.txt'),
        os.path.join(args.data_dir, 'meta.pkl'),
        block_size
    )
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    
    # Train
    train(model, train_loader, val_loader, args)
    
    print(f"\nTraining complete! Adapter saved to {args.out_dir}/adapter.pt")

if __name__ == '__main__':
    main()
