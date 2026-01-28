"""
Sample from trained Seq2Seq model
"""
import os
import pickle
import torch
import numpy as np
from seq2seq_model import Encoder, Decoder, Seq2Seq
from train_seq2seq import InsertSpacesDataset


def load_model(checkpoint_path, vocab_size, device):
    """Load trained model from checkpoint"""
    EMBED_DIM = 64
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3
    
    encoder = Encoder(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    decoder = Decoder(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Train Loss: {checkpoint['train_loss']:.3f}, Val Loss: {checkpoint['val_loss']:.3f}")
    
    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset to get vocabulary
    data_dir = 'data/basic'
    temp_file = os.path.join(data_dir, 'train.txt')
    
    if not os.path.exists(temp_file):
        # Create temp file from binary data
        with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        
        train_data = np.fromfile(os.path.join(data_dir, 'train.bin'), dtype=np.uint16)
        itos = meta['itos']
        text_data = ''.join([itos[int(i)] for i in train_data])
        
        with open(temp_file, 'w') as f:
            f.write(text_data)
    
    print("Loading dataset for vocabulary...")
    dataset = InsertSpacesDataset(temp_file, max_samples=1000)
    
    # Load model
    checkpoint_path = 'out/seq2seq/best_model.pt'
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train_seq2seq.py")
        return
    
    print("\nLoading model...")
    model = load_model(checkpoint_path, len(dataset.vocab), device)
    
    # Test with custom words
    print("\n" + "="*60)
    print("Testing Seq2Seq Model - Insert Spaces Task")
    print("="*60)
    
    test_words = [
        'hello',
        'world',
        'python',
        'test',
        'abc',
        'programming',
        'neural',
        'network',
        'seq2seq',
        'attention'
    ]
    
    print("\nPredictions:")
    print("-" * 60)
    
    correct = 0
    total = 0
    
    for word in test_words:
        # Encode input
        src_indices = dataset.encode(word)
        if not src_indices:  # Skip if word has unknown characters
            print(f"Input: {word:15s} | Skipped (unknown characters)")
            continue
            
        src_tensor = torch.tensor([src_indices]).to(device)
        
        # Generate output
        # Expected length: len(word) + (len(word)-1) spaces + SOS + EOS = 2*len(word) + 1
        max_len = len(word) * 2 + 1
        with torch.no_grad():
            generated = model.generate(src_tensor, max_len, dataset.sos_idx, dataset.eos_idx)
        
        # Decode output
        output_text = dataset.decode(generated[0].cpu().numpy())
        output_text = output_text.replace('<SOS>', '').replace('<EOS>', '').replace('<PAD>', '').strip()
        
        # Expected output
        expected = ' '.join(word)
        
        # Check if correct
        is_correct = output_text == expected
        if is_correct:
            correct += 1
        total += 1
        
        status = "✓" if is_correct else "✗"
        print(f"{status} Input: {word:15s} | Output: {output_text:25s} | Expected: {expected}")
    
    print("-" * 60)
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    
    # Interactive mode
    print("\n" + "="*60)
    print("Interactive Mode (type 'quit' to exit)")
    print("="*60)
    
    while True:
        word = input("\nEnter a word: ").strip().lower()
        
        if word in ['quit', 'exit', 'q']:
            break
        
        if not word:
            continue
        
        # Check if all characters are in vocabulary
        unknown_chars = [ch for ch in word if ch not in dataset.char2idx]
        if unknown_chars:
            print(f"Warning: Unknown characters: {unknown_chars}")
            continue
        
        # Encode and generate
        src_indices = dataset.encode(word)
        src_tensor = torch.tensor([src_indices]).to(device)
        
        max_len = len(word) * 2 + 3
        with torch.no_grad():
            generated = model.generate(src_tensor, max_len, dataset.sos_idx, dataset.eos_idx)
        
        output_text = dataset.decode(generated[0].cpu().numpy())
        output_text = output_text.replace('<SOS>', '').replace('<EOS>', '').replace('<PAD>', '').strip()
        
        expected = ' '.join(word)
        is_correct = output_text == expected
        status = "✓" if is_correct else "✗"
        
        print(f"{status} Output: {output_text}")
        print(f"  Expected: {expected}")


if __name__ == '__main__':
    main()
