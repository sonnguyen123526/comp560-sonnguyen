"""
Training script for Simple Seq2Seq model on insert spaces task
Much smaller model to avoid overfitting on this simple task
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from simple_seq2seq_model import SimpleEncoder, SimpleDecoder, SimpleSeq2Seq
from train_seq2seq import InsertSpacesDataset, collate_fn, calculate_accuracy


def train_epoch(model, dataloader, optimizer, criterion, clip, device, pad_idx):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    
    for src, trg in dataloader:
        src = src.to(device)
        trg = trg.to(device)
        
        optimizer.zero_grad()
        output = model(src, trg)
        
        # Reshape for loss
        output_dim = output.shape[-1]
        output = output[:, 1:].contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
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
            
            output = model(src, trg, teacher_forcing_ratio=0)
            
            output_dim = output.shape[-1]
            output = output[:, 1:].contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def main():
    # Hyperparameters - Simple model without attention
    EMBED_DIM = 64
    HIDDEN_DIM = 128
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    CLIP = 1.0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    data_dir = 'data/basic'
    temp_file = os.path.join(data_dir, 'train.txt')
    
    print("Loading dataset...")
    dataset = InsertSpacesDataset(temp_file, max_samples=50000)
    
    # Split data
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
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
    
    # Create simple model
    print("\nInitializing simple model...")
    vocab_size = len(dataset.vocab)
    
    encoder = SimpleEncoder(vocab_size, EMBED_DIM, HIDDEN_DIM)
    decoder = SimpleDecoder(vocab_size, EMBED_DIM, HIDDEN_DIM)
    model = SimpleSeq2Seq(encoder, decoder, device).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Simple model has {total_params:,} trainable parameters')
    print(f'(Original model had ~4M parameters - this is {total_params/4045086*100:.1f}% of that)')
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_idx)
    
    # Training loop
    print("\nStarting training...")
    print("="*70)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, CLIP, device, dataset.pad_idx)
        val_loss = evaluate(model, val_loader, criterion, device, dataset.pad_idx)
        
        end_time = time.time()
        epoch_time = end_time - start_time
        
        # Calculate accuracy every 5 epochs
        if (epoch + 1) % 5 == 0:
            train_acc = calculate_accuracy(model, train_loader, dataset, device)
            val_acc = calculate_accuracy(model, val_loader, dataset, device)
            print(f'Epoch {epoch+1:02}/{NUM_EPOCHS} | Time: {epoch_time:.1f}s')
            print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%')
            print(f'  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%')
            
            gap = val_loss - train_loss
            if gap > 0.05:
                print(f' Gap: {gap:.4f} - OVERFITTING detected!')
            elif gap > 0.02:
                print(f' Gap: {gap:.4f} - slight overfitting')
        else:
            print(f'Epoch {epoch+1:02}/{NUM_EPOCHS} | Time: {epoch_time:.1f}s | '
                  f'Train: {train_loss:.4f} | Val: {val_loss:.4f}')
        
        # Save best model with early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs('out/simple_seq2seq', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, 'out/simple_seq2seq/best_model.pt')
            print(f'  ✓ Saved new best model!')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\n Early stopping triggered (no improvement for {patience} epochs)')
                break
    
    print("\n" + "="*70)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Final evaluation
    print("\nFinal evaluation on validation set:")
    val_acc = calculate_accuracy(model, val_loader, dataset, device)
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    
    # Test some examples
    print("\n" + "="*70)
    print("Sample predictions:")
    print("="*70)
    
    model.eval()
    test_words = ['hello', 'world', 'test', 'python', 'abc', 'simple']
    
    for word in test_words:
        src_indices = dataset.encode(word)
        if not src_indices:
            continue
            
        src_tensor = torch.tensor([src_indices]).to(device)
        generated = model.generate(src_tensor, len(word) * 2 + 2, dataset.sos_idx, dataset.eos_idx)
        output_text = dataset.decode(generated[0].cpu().numpy())
        output_text = output_text.replace('<SOS>', '').replace('<EOS>', '').replace('<PAD>', '').strip()
        
        expected = ' '.join(word)
        status = "✓" if output_text == expected else "✗"
        print(f"{status} {word:10s} -> {output_text:20s} (expected: {expected})")


if __name__ == '__main__':
    main()
