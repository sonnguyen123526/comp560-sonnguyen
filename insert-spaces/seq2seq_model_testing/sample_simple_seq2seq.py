"""
Sample from trained Simple Seq2Seq model
"""
import os
import torch
from simple_seq2seq_model import SimpleEncoder, SimpleDecoder, SimpleSeq2Seq
from train_seq2seq import InsertSpacesDataset


def load_model(checkpoint_path, vocab_size, device):
    """Load trained model from checkpoint"""
    EMBED_DIM = 64
    HIDDEN_DIM = 128
    
    encoder = SimpleEncoder(vocab_size, EMBED_DIM, HIDDEN_DIM)
    decoder = SimpleDecoder(vocab_size, EMBED_DIM, HIDDEN_DIM)
    model = SimpleSeq2Seq(encoder, decoder, device).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Train Loss: {checkpoint['train_loss']:.4f}, Val Loss: {checkpoint['val_loss']:.4f}")
    
    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset to get vocabulary
    data_dir = 'data/basic'
    temp_file = os.path.join(data_dir, 'train.txt')
    
    print("\nLoading dataset for vocabulary...")
    dataset = InsertSpacesDataset(temp_file, max_samples=1000)
    
    # Load model
    checkpoint_path = 'out/simple_seq2seq/best_model.pt'
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train_simple_seq2seq.py")
        return
    
    print("\nLoading model...")
    model = load_model(checkpoint_path, len(dataset.vocab), device)
    
    # Test with custom words
    print("\n" + "="*60)
    print("Testing Simple Seq2Seq Model - Insert Spaces Task")
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
        'attention',
        'simple',
        'model'
    ]
    
    print("\nPredictions:")
    print("-" * 60)
    
    correct = 0
    total = 0
    
    for word in test_words:
        # Encode input
        src_indices = dataset.encode(word)
        if not src_indices:
            print(f"Input: {word:15s} | Skipped (unknown characters)")
            continue
            
        src_tensor = torch.tensor([src_indices]).to(device)
        
        # Generate output - expected length for "hello" is "h e l l o" = len*2-1 chars + SOS + EOS
        expected_len = len(word) * 2 + 1  # chars + spaces + SOS + EOS
        with torch.no_grad():
            generated = model.generate(src_tensor, expected_len, dataset.sos_idx, dataset.eos_idx)
        
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
    print(f"Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    
    # Interactive mode
    print("\n" + "="*60)
    print("Interactive Mode (type 'quit' to exit)")
    print("="*60)
    print()
    
    while True:
        word = input("Enter a word: ").strip()
        if word.lower() == 'quit':
            break
        
        if not word:
            continue
        
        # Encode and generate
        src_indices = dataset.encode(word)
        if not src_indices:
            print("✗ Word contains unknown characters")
            continue
        
        src_tensor = torch.tensor([src_indices]).to(device)
        expected_len = len(word) * 2 + 1
        
        with torch.no_grad():
            generated = model.generate(src_tensor, expected_len, dataset.sos_idx, dataset.eos_idx)
        
        output_text = dataset.decode(generated[0].cpu().numpy())
        output_text = output_text.replace('<SOS>', '').replace('<EOS>', '').replace('<PAD>', '').strip()
        
        expected = ' '.join(word)
        is_correct = output_text == expected
        status = "✓" if is_correct else "✗"
        
        print(f"{status} Output: {output_text}")
        print(f"  Expected: {expected}")
        print()


if __name__ == '__main__':
    main()
