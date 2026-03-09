"""
Evaluate trained composition models (layer concat or adapter).
"""
import os
import sys
import random
import argparse
import pickle
import torch
import numpy as np

from compose import load_model, LayerConcatComposition, AdapterComposition

SEED = 42

def load_meta(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def make_samples(n):
    random.seed(SEED)
    samples, seen = [], set()
    while len(samples) < n:
        a = random.randint(100, 999)
        b = random.randint(100, 999)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        a_rev = int(str(a)[::-1])
        b_rev = int(str(b)[::-1])
        samples.append((a, b, str(a_rev + b_rev)))
    return samples

def load_concat_model(model_a_path, model_b_path, checkpoint_path, layers_from_a=2, device='mps'):
    model_A = load_model(model_a_path)
    model_B = load_model(model_b_path)
    model = LayerConcatComposition(model_A, model_B, layers_from_A=layers_from_a)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model = model.to(device)
    return model

def load_adapter_model(model_a_path, model_b_path, adapter_path, adapter_dim=64, device='mps'):
    model_A = load_model(model_a_path)
    model_B = load_model(model_b_path)
    model = AdapterComposition(model_A, model_B, adapter_dim=adapter_dim)
    
    checkpoint = torch.load(adapter_path, map_location='cpu')
    model.adapter.load_state_dict(checkpoint['adapter'])
    model.eval()
    model = model.to(device)
    return model

@torch.no_grad()
def greedy(model, prompt_ids, max_new, device):
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    # Get block size from model config (handle both direct models and compositions)
    if hasattr(model, 'config'):
        bs = model.config.block_size
    elif hasattr(model, 'model_A'):
        bs = model.model_A.config.block_size
    else:
        bs = 48  # fallback
    
    for _ in range(max_new):
        logits, _ = model(x[:, -bs:])
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        x = torch.cat([x, nxt], dim=1)
    return x[0, len(prompt_ids):].tolist()

@torch.no_grad()
def evaluate(model, n_samples, device='mps'):
    meta = load_meta('data/composed/meta.pkl')
    stoi, itos = meta['stoi'], meta['itos']
    
    samples = make_samples(n_samples)
    
    total = correct = 0
    wrong = []
    
    for a, b, expected in samples:
        total += 1
        a_str = ' '.join(str(a))
        b_str = ' '.join(str(b))
        prompt = a_str + ' + ' + b_str + ' ->'
        
        try:
            ids = [stoi[c] for c in prompt]
        except KeyError:
            continue
        
        out_ids = greedy(model, ids, max_new=25, device=device)
        raw = ''.join(itos[i] for i in out_ids)
        
        if '->' in raw:
            result = raw.split('->')[-1].strip().split('\n')[0].strip()
        else:
            result = raw.strip().split('\n')[0].strip()
        pred = ''.join(result.split())
        
        if pred == expected:
            correct += 1
        else:
            wrong.append((prompt, expected, pred))
    
    accuracy = 100.0 * correct / total
    return correct, total, accuracy, wrong

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, required=True, choices=['concat', 'adapter'])
    parser.add_argument('--model_a', type=str, default='out/reverse/ckpt.pt')
    parser.add_argument('--model_b', type=str, default='out/addition/ckpt.pt')
    parser.add_argument('--checkpoint', type=str, help='Path to trained model/adapter')
    parser.add_argument('--n', type=int, default=100, help='Number of test samples')
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--layers_from_a', type=int, default=2)
    parser.add_argument('--adapter_dim', type=int, default=64)
    args = parser.parse_args()
    
    # Set default checkpoint paths if not provided
    if not args.checkpoint:
        if args.strategy == 'concat':
            args.checkpoint = 'out/concat/ckpt.pt'
        else:
            args.checkpoint = 'out/adapter/adapter.pt'
    
    # Load model
    print(f"Loading {args.strategy} model...")
    if args.strategy == 'concat':
        model = load_concat_model(args.model_a, args.model_b, args.checkpoint, 
                                  layers_from_a=args.layers_from_a, device=args.device)
    else:
        model = load_adapter_model(args.model_a, args.model_b, args.checkpoint,
                                   adapter_dim=args.adapter_dim, device=args.device)
    
    # Evaluate
    print(f"Evaluating on {args.n} samples...\n")
    correct, total, accuracy, wrong = evaluate(model, args.n, device=args.device)
    
    print("=" * 60)
    print(f"  {args.strategy.upper()} Composition Results")
    print("=" * 60)
    print(f"  Samples   : {total}")
    print(f"  Correct   : {correct}")
    print(f"  Accuracy  : {accuracy:.2f}%")
    print("=" * 60)
    
    if wrong:
        print(f"\n  First {min(5, len(wrong))} wrong predictions:")
        for prompt, exp, got in wrong[:5]:
            print(f"    {prompt!r}  expected={exp!r}  got={got!r}")

if __name__ == '__main__':
    main()
