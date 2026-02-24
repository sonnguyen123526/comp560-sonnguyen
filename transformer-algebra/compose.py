import os
import sys
import torch
import torch.nn as nn
import pickle

nanogpt_path = os.path.join(os.path.dirname(__file__), '..', '..', 'comp560-nanoGPT')
sys.path.insert(0, nanogpt_path)

from model import GPT, GPTConfig

def load_model(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"loading {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"  {gptconf.n_layer}-layer model, {sum(p.numel() for p in model.parameters()):,} params")
    return model

def load_meta(dataset_name):
    meta_path = f'../data/{dataset_name}/meta.pkl'
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    return meta

class PipelineComposition:
    def __init__(self, model_A, model_B, meta_A, meta_B, device='cpu'):
        self.model_A = model_A.to(device)
        self.model_B = model_B.to(device)
        self.device = device
        
        self.encode_A = lambda s: [meta_A['stoi'][c] for c in s]
        self.decode_A = lambda l: ''.join([meta_A['itos'][i] for i in l])
        self.encode_B = lambda s: [meta_B['stoi'][c] for c in s]
        self.decode_B = lambda l: ''.join([meta_B['itos'][i] for i in l])
        
        self.model_A.eval()
        self.model_B.eval()
    
    def generate(self, prompt, max_tokens=20, temperature=0.1):
        return None  # implemented in evaluation script
    
    def pipeline(self, input_text, verbose=False):
        # Step 1: reverse each number, Step 2: add
        
        if verbose:
            print(f"\nPipeline composition:")
            print(f"  Input: {input_text}")

        return "TODO: Implement pipeline generation"

class LayerConcatComposition(nn.Module):
    def __init__(self, model_A, model_B, layers_from_A=2):
        super().__init__()
        
        config_A = model_A.config
        config_B = model_B.config
        
        assert config_A.n_embd == config_B.n_embd, "embedding dim must match"
        assert config_A.n_layer == config_B.n_layer, "num layers must match"
        self.config = config_A
        self.transformer = nn.ModuleDict(dict(
            wte = model_A.transformer.wte,
            wpe = model_A.transformer.wpe,
            drop = model_A.transformer.drop,
            h = nn.ModuleList([
                *list(model_A.transformer.h[:layers_from_A]),
                *list(model_B.transformer.h[layers_from_A:])
            ]),
            ln_f = model_B.transformer.ln_f,
        ))
        self.lm_head = model_B.lm_head
        
        print(f"  Combined model: first {layers_from_A} layers from A, "
              f"last {config_A.n_layer - layers_from_A} from B")
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Sequence too long: {t} > {self.config.block_size}"
        
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

class AdapterComposition(nn.Module):
    def __init__(self, model_A, model_B, adapter_dim=64):
        super().__init__()
        
        self.model_A = model_A
        self.model_B = model_B
        
        for param in self.model_A.parameters():
            param.requires_grad = False
        for param in self.model_B.parameters():
            param.requires_grad = False
        hidden_dim = model_A.config.n_embd
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, hidden_dim)
        )
        
        print(f"  Adapter: {hidden_dim} -> {adapter_dim} -> {hidden_dim}")
        print(f"  Trainable params: {sum(p.numel() for p in self.adapter.parameters())}")
    
    def forward(self, idx, targets=None):
        with torch.no_grad():
            device = idx.device
            b, t = idx.size()
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            
            tok_emb = self.model_A.transformer.wte(idx)
            pos_emb = self.model_A.transformer.wpe(pos)
            x = self.model_A.transformer.drop(tok_emb + pos_emb)
            
            for block in self.model_A.transformer.h:
                x = block(x)
            
            h_A = self.model_A.transformer.ln_f(x)
        
        h_adapted = self.adapter(h_A)
        x = h_adapted
        for block in self.model_B.transformer.h:
            x = block(x)
        
        x = self.model_B.transformer.ln_f(x)
        logits = self.model_B.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss

def create_composition(model_A, model_B, strategy='pipeline', **kwargs):
    if strategy == 'pipeline':
        return PipelineComposition(model_A, model_B, **kwargs)
    elif strategy == 'concat':
        return LayerConcatComposition(model_A, model_B, **kwargs)
    elif strategy == 'adapter':
        return AdapterComposition(model_A, model_B, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

if __name__ == '__main__':
    model_paths = {
        'reverse': '../out/reverse/ckpt.pt',
        'addition': '../out/addition/ckpt.pt',
    }
    
    for name, path in model_paths.items():
        status = 'found' if os.path.exists(path) else 'not found'
        print(f"  {name}: {status}")
    
    if all(os.path.exists(p) for p in model_paths.values()):
        model_A = load_model(model_paths['reverse'])
        model_B = load_model(model_paths['addition'])

        print("\nlayer concat:")
        concat_model = create_composition(model_A, model_B, strategy='concat', layers_from_A=2)
        print(f"  total params: {sum(p.numel() for p in concat_model.parameters()):,}")

        print("\nadapter:")
        adapter_model = create_composition(model_A, model_B, strategy='adapter', adapter_dim=64)
        print(f"  total params: {sum(p.numel() for p in adapter_model.parameters()):,}")
