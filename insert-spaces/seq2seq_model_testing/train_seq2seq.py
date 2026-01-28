"""
Training script for Seq2Seq model on insert spaces task
"""
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import time
from seq2seq_model import Encoder, Decoder, Seq2Seq

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class InsertSpacesDataset(Dataset):
    """Dataset for insert spaces task"""
    def __init__(self, data_path, max_samples=None):
        with open(data_path, 'r') as f:
            lines = f.readlines()
        
        self.examples = []
        for line in lines:
            line = line.strip()
            if not line or 'Input:' not in line:
                continue
            
            # Parse "Input: word Output: w o r d"
            try:
                parts = line.split(' Output: ')
                if len(parts) != 2:
                    continue
                    
                input_part = parts[0].replace('Input: ', '').strip()
                output_part = parts[1].strip()
                
                if input_part and output_part:
                    self.examples.append((input_part, output_part))
                    
            except Exception:
                continue
        
        if max_samples:
            self.examples = self.examples[:max_samples]
        
        print(f"Loaded {len(self.examples)} examples")
        
        # Build vocabulary
        self.build_vocab()
        
    def build_vocab(self):
        """Build character-level vocabulary"""
        chars = set()
        for input_text, output_text in self.examples:
            chars.update(input_text)
            chars.update(output_text)
        
        # Add special tokens
        self.special_tokens = ['<PAD>', '<SOS>', '<EOS>']
        self.chars = sorted(list(chars))
        self.vocab = self.special_tokens + self.chars
        
        self.char2idx = {ch: idx for idx, ch in enumerate(self.vocab)}
        self.idx2char = {idx: ch for idx, ch in enumerate(self.vocab)}
        
        self.pad_idx = self.char2idx['<PAD>']
        self.sos_idx = self.char2idx['<SOS>']
        self.eos_idx = self.char2idx['<EOS>']
        
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Characters: {self.chars}")
        
    def encode(self, text):
        """Encode text to indices"""
        return [self.char2idx[ch] for ch in text if ch in self.char2idx]
    
    def decode(self, indices):
        """Decode indices to text"""
        return ''.join([self.idx2char[idx] for idx in indices if idx in self.idx2char])
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        input_text, output_text = self.examples[idx]
        
        # Encode input and output
        src = self.encode(input_text)
        trg = [self.sos_idx] + self.encode(output_text) + [self.eos_idx]
        
        return torch.tensor(src), torch.tensor(trg)


def collate_fn(batch, pad_idx):
    """Collate function to pad sequences to same length"""
    src_batch, trg_batch = zip(*batch)
    
    # Pad source sequences
    src_lengths = [len(s) for s in src_batch]
    max_src_len = max(src_lengths)
    src_padded = torch.full((len(batch), max_src_len), pad_idx, dtype=torch.long)
    for i, src in enumerate(src_batch):
        src_padded[i, :len(src)] = src
    
    # Pad target sequences
    trg_lengths = [len(t) for t in trg_batch]
    max_trg_len = max(trg_lengths)
    trg_padded = torch.full((len(batch), max_trg_len), pad_idx, dtype=torch.long)
    for i, trg in enumerate(trg_batch):
        trg_padded[i, :len(trg)] = trg
    
    return src_padded, trg_padded


def train_epoch(model, dataloader, optimizer, criterion, clip, device, pad_idx):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    
    for src, trg in dataloader:
        src = src.to(device)
        trg = trg.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, trg)  # (batch, trg_len, vocab_size)
        
        # Reshape for loss calculation
        output_dim = output.shape[-1]
        output = output[:, 1:].contiguous().view(-1, output_dim)  # Ignore <SOS>
        trg = trg[:, 1:].contiguous().view(-1)  # Ignore <SOS>
        
        # Calculate loss
        loss = criterion(output, trg)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, pad_idx):
    """Evaluate model"""
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for src, trg in dataloader:
            src = src.to(device)
            trg = trg.to(device)
            
            # Forward pass with no teacher forcing
            output = model(src, trg, teacher_forcing_ratio=0)
            
            # Reshape for loss calculation
            output_dim = output.shape[-1]
            output = output[:, 1:].contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def calculate_accuracy(model, dataloader, dataset, device):
    """Calculate exact match accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for src, trg in dataloader:
            src = src.to(device)
            batch_size = src.shape[0]
            
            # Generate predictions
            max_len = trg.shape[1]
            generated = model.generate(src, max_len, dataset.sos_idx, dataset.eos_idx)
            
            # Compare with ground truth (ignore SOS and EOS)
            for i in range(batch_size):
                pred_text = dataset.decode(generated[i].cpu().numpy())
                true_text = dataset.decode(trg[i].cpu().numpy())
                
                # Remove special tokens
                pred_text = pred_text.replace('<SOS>', '').replace('<EOS>', '').replace('<PAD>', '')
                true_text = true_text.replace('<SOS>', '').replace('<EOS>', '').replace('<PAD>', '')
                
                if pred_text.strip() == true_text.strip():
                    correct += 1
                total += 1
    
    return correct / total if total > 0 else 0


def main():
    # Hyperparameters
    EMBED_DIM = 64
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    CLIP = 1.0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset (we'll create a temporary text file from the binary data)
    # First, load the binary data and convert it to text
    data_dir = 'data/basic'
    
    # Load meta information
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    # Load train data
    train_data = np.fromfile(os.path.join(data_dir, 'train.bin'), dtype=np.uint16)
    
    # Decode to text
    itos = meta['itos']
    text_data = ''.join([itos[int(i)] for i in train_data])
    
    # Save to temporary text file
    temp_file = os.path.join(data_dir, 'train.txt')
    with open(temp_file, 'w') as f:
        f.write(text_data)
    
    print("Loading dataset...")
    dataset = InsertSpacesDataset(temp_file, max_samples=10000)
    
    # Split into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=lambda batch: collate_fn(batch, dataset.pad_idx)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, dataset.pad_idx)
    )
    
    # Create model
    print("\nInitializing model...")
    vocab_size = len(dataset.vocab)
    
    encoder = Encoder(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    decoder = Decoder(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    # Count parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'Model has {count_parameters(model):,} trainable parameters')
    
    # Initialize optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_idx)
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, CLIP, device, dataset.pad_idx)
        val_loss = evaluate(model, val_loader, criterion, device, dataset.pad_idx)
        
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        # Calculate accuracy every 5 epochs
        if (epoch + 1) % 5 == 0:
            train_acc = calculate_accuracy(model, train_loader, dataset, device)
            val_acc = calculate_accuracy(model, val_loader, dataset, device)
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc*100:.2f}%')
        else:
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {val_loss:.3f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('out/seq2seq', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, 'out/seq2seq/best_model.pt')
            print(f'\tSaved new best model!')
    
    print("\nTraining complete!")
    
    # Final evaluation
    print("\nFinal evaluation on validation set:")
    val_acc = calculate_accuracy(model, val_loader, dataset, device)
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    
    # Test some examples
    print("\nSample predictions:")
    model.eval()
    test_words = ['hello', 'world', 'test', 'python', 'abc']
    
    for word in test_words:
        src_indices = dataset.encode(word)
        src_tensor = torch.tensor([src_indices]).to(device)
        
        generated = model.generate(src_tensor, len(word) * 2 + 2, dataset.sos_idx, dataset.eos_idx)
        output_text = dataset.decode(generated[0].cpu().numpy())
        output_text = output_text.replace('<SOS>', '').replace('<EOS>', '').replace('<PAD>', '').strip()
        
        expected = ' '.join(word)
        print(f"Input: {word:10s} | Predicted: {output_text:20s} | Expected: {expected}")


if __name__ == '__main__':
    main()
