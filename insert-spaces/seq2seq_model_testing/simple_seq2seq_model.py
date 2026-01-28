"""
Simple Sequence-to-Sequence Model for Insert Spaces Task
Minimal architecture without attention - just encoder-decoder LSTM
"""
import torch
import torch.nn as nn


class SimpleEncoder(nn.Module):
    """Simple LSTM encoder"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len)
        Returns:
            outputs: (batch_size, seq_len, hidden_dim)
            hidden: tuple of (h_n, c_n)
        """
        embedded = self.embedding(x)
        outputs, hidden = self.lstm(embedded)
        return outputs, hidden


class SimpleDecoder(nn.Module):
    """Simple LSTM decoder"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Decoder LSTM only takes embedded input (no concatenation)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, input, hidden):
        """
        Args:
            input: (batch_size) - current token
            hidden: tuple of (h, c) - previous hidden state
        Returns:
            prediction: (batch_size, vocab_size)
            hidden: tuple of (h, c) - new hidden state
        """
        input = input.unsqueeze(1)  # (batch_size, 1)
        embedded = self.embedding(input)  # (batch_size, 1, embed_dim)
        output, hidden = self.lstm(embedded, hidden)  # (batch_size, 1, hidden_dim)
        prediction = self.fc_out(output.squeeze(1))  # (batch_size, vocab_size)
        return prediction, hidden


class SimpleSeq2Seq(nn.Module):
    """Simple Sequence-to-Sequence model without attention"""
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Args:
            src: (batch_size, src_len)
            trg: (batch_size, trg_len)
            teacher_forcing_ratio: probability of using teacher forcing
        Returns:
            outputs: (batch_size, trg_len, vocab_size)
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.vocab_size
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Encode input
        _, hidden = self.encoder(src)
        
        # Start with SOS token
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t, :] = output
            
            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
            
        return outputs
    
    def generate(self, src, max_len, sos_idx, eos_idx=None):
        """
        Generate output sequence
        
        Args:
            src: (batch_size, src_len)
            max_len: maximum length
            sos_idx: start token index
            eos_idx: end token index (optional)
        Returns:
            generated: (batch_size, gen_len)
        """
        self.eval()
        batch_size = src.shape[0]
        
        with torch.no_grad():
            # Encode
            _, hidden = self.encoder(src)
            
            # Start with SOS
            input = torch.full((batch_size,), sos_idx, dtype=torch.long).to(self.device)
            generated = [input]
            
            finished = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
            
            for _ in range(max_len - 1):
                output, hidden = self.decoder(input, hidden)
                predicted = output.argmax(1)
                
                if eos_idx is not None:
                    finished = finished | (predicted == eos_idx)
                
                input = predicted
                generated.append(predicted)
                
                if eos_idx is not None and finished.all():
                    break
            
            generated = torch.stack(generated, dim=1)
            
        return generated
