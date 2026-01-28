"""
Sequence-to-Sequence Model for Insert Spaces Task
Uses an Encoder-Decoder architecture with attention mechanism
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encodes the input sequence (e.g., 'hello')"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len) - input sequence
        Returns:
            outputs: (batch_size, seq_len, hidden_dim * 2) - encoder outputs
            hidden: tuple of (h_n, c_n) - final hidden states
        """
        embedded = self.dropout(self.embedding(x))  # (batch, seq_len, embed_dim)
        outputs, hidden = self.lstm(embedded)
        return outputs, hidden


class Attention(nn.Module):
    """Bahdanau attention mechanism"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, decoder_hidden, encoder_outputs):
        """
        Args:
            decoder_hidden: (batch_size, hidden_dim) - current decoder hidden state
            encoder_outputs: (batch_size, seq_len, hidden_dim * 2) - all encoder outputs
        Returns:
            attention_weights: (batch_size, seq_len)
        """
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Repeat decoder hidden state src_len times
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Calculate attention scores
        energy = torch.tanh(self.attn(torch.cat((decoder_hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    """Decodes to output sequence (e.g., 'h e l l o')"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(hidden_dim)
        
        # LSTM input is embed_dim + (hidden_dim * 2) from attention context
        self.lstm = nn.LSTM(
            embed_dim + hidden_dim * 2,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc_out = nn.Linear(hidden_dim * 3 + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        """
        Args:
            input: (batch_size) - current token
            hidden: tuple of (h, c) - previous hidden state
            encoder_outputs: (batch_size, src_len, hidden_dim * 2)
        Returns:
            prediction: (batch_size, vocab_size)
            hidden: tuple of (h, c) - new hidden state
            attention_weights: (batch_size, src_len)
        """
        input = input.unsqueeze(1)  # (batch_size, 1)
        embedded = self.dropout(self.embedding(input))  # (batch_size, 1, embed_dim)
        
        # Calculate attention weights using the first layer's hidden state
        attention_weights = self.attention(hidden[0][-1], encoder_outputs)  # (batch_size, src_len)
        
        # Calculate attention-weighted context
        attention_weights_unsqueezed = attention_weights.unsqueeze(1)  # (batch_size, 1, src_len)
        context = torch.bmm(attention_weights_unsqueezed, encoder_outputs)  # (batch_size, 1, hidden_dim * 2)
        
        # Concatenate embedded input and context
        lstm_input = torch.cat((embedded, context), dim=2)  # (batch_size, 1, embed_dim + hidden_dim * 2)
        
        # Pass through LSTM
        output, hidden = self.lstm(lstm_input, hidden)
        
        # Concatenate output, context, and embedded input for prediction
        output = output.squeeze(1)  # (batch_size, hidden_dim)
        context = context.squeeze(1)  # (batch_size, hidden_dim * 2)
        embedded = embedded.squeeze(1)  # (batch_size, embed_dim)
        
        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))
        
        return prediction, hidden, attention_weights


class Seq2Seq(nn.Module):
    """Complete Sequence-to-Sequence model with attention"""
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Args:
            src: (batch_size, src_len) - input sequence
            trg: (batch_size, trg_len) - target sequence
            teacher_forcing_ratio: probability of using teacher forcing
        Returns:
            outputs: (batch_size, trg_len, vocab_size)
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.vocab_size
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Encode input sequence
        encoder_outputs, hidden = self.encoder(src)
        
        # Transform encoder hidden state to decoder hidden state
        # Encoder is bidirectional with 2*hidden_dim, decoder has hidden_dim
        # Take forward and backward hidden states and combine them
        h_n, c_n = hidden
        
        # h_n shape: (num_layers * 2, batch, hidden_dim) for bidirectional
        # We need: (num_layers, batch, hidden_dim) for decoder
        
        # Combine forward and backward states by summing them
        h_decoder = []
        c_decoder = []
        for i in range(self.decoder.num_layers):
            # Take corresponding forward and backward layers
            h_forward = h_n[i * 2]
            h_backward = h_n[i * 2 + 1]
            c_forward = c_n[i * 2]
            c_backward = c_n[i * 2 + 1]
            
            h_decoder.append(h_forward + h_backward)
            c_decoder.append(c_forward + c_backward)
            
        hidden = (torch.stack(h_decoder), torch.stack(c_decoder))
        
        # First input to decoder is <SOS> token (first token of target)
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            # Pass through decoder
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            
            # Store output
            outputs[:, t, :] = output
            
            # Decide whether to use teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            # Get the highest predicted token
            top1 = output.argmax(1)
            
            # Use teacher forcing or predicted token as next input
            input = trg[:, t] if teacher_force else top1
            
        return outputs
    
    def generate(self, src, max_len, sos_idx, eos_idx=None):
        """
        Generate output sequence for given input (inference mode)
        
        Args:
            src: (batch_size, src_len) - input sequence
            max_len: maximum length of generated sequence
            sos_idx: start of sequence token index
            eos_idx: end of sequence token index (optional)
        Returns:
            generated: (batch_size, gen_len) - generated sequence
        """
        self.eval()
        batch_size = src.shape[0]
        
        with torch.no_grad():
            # Encode input
            encoder_outputs, hidden = self.encoder(src)
            
            # Transform encoder hidden state to decoder hidden state
            h_n, c_n = hidden
            h_decoder = []
            c_decoder = []
            for i in range(self.decoder.num_layers):
                h_forward = h_n[i * 2]
                h_backward = h_n[i * 2 + 1]
                c_forward = c_n[i * 2]
                c_backward = c_n[i * 2 + 1]
                
                h_decoder.append(h_forward + h_backward)
                c_decoder.append(c_forward + c_backward)
                
            hidden = (torch.stack(h_decoder), torch.stack(c_decoder))
            
            # Start with SOS token
            input = torch.full((batch_size,), sos_idx, dtype=torch.long).to(self.device)
            generated = [input]
            
            # Track which sequences have finished
            finished = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
            
            for _ in range(max_len - 1):
                output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
                predicted = output.argmax(1)
                
                # Store predictions
                generated.append(predicted)
                
                # Mark finished sequences (stop generating after EOS)
                if eos_idx is not None:
                    finished = finished | (predicted == eos_idx)
                    # Stop if all sequences have finished
                    if finished.all():
                        break
                
                # Use predicted token as next input
                input = predicted
            
            generated = torch.stack(generated, dim=1)  # (batch_size, gen_len)
            
        return generated
