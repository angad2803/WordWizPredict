"""
Create sample model and vocabulary files for testing WordWiz.
This script generates dummy files to test the application structure.
"""

import pickle
import torch
import torch.nn as nn
import os

class NextWordLSTM(nn.Module):
    """Sample LSTM model for next word prediction."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(NextWordLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.dropout(lstm_out)
        output = self.fc(output[:, -1, :])  # Use last timestep
        return output

def create_sample_files():
    """Create sample model and vocabulary files."""
    
    # Create sample vocabulary
    sample_words = [
        '<PAD>', '<UNK>', '<START>', '<END>',
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off',
        'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
        'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just',
        'should', 'now', 'good', 'great', 'nice', 'bad', 'small', 'large',
        'big', 'little', 'long', 'short', 'high', 'low', 'old', 'new', 'young',
        'man', 'woman', 'child', 'people', 'person', 'family', 'friend',
        'house', 'home', 'school', 'work', 'place', 'time', 'day', 'week',
        'month', 'year', 'life', 'world', 'country', 'city', 'weather', 'is',
        'was', 'are', 'were', 'been', 'being', 'have', 'has', 'had', 'do',
        'does', 'did', 'doing', 'go', 'goes', 'went', 'going', 'come', 'came',
        'coming', 'see', 'saw', 'seen', 'looking', 'look', 'know', 'think',
        'believe', 'feel', 'want', 'need', 'like', 'love', 'hate', 'enjoy'
    ]
    
    vocab = {
        'word_to_idx': {word: idx for idx, word in enumerate(sample_words)},
        'idx_to_word': {idx: word for idx, word in enumerate(sample_words)}
    }
    
    vocab_size = len(sample_words)
    
    # Create sample model
    model = NextWordLSTM(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2
    )
    
    # Save vocabulary
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    
    # Save model with metadata
    torch.save({
        'state_dict': model.state_dict(),
        'embedding_dim': 128,
        'hidden_dim': 256,
        'num_layers': 2,
        'vocab_size': vocab_size
    }, 'next_word_model.pt')
    
    print(f"✅ Created sample files:")
    print(f"   - vocab.pkl (vocabulary with {vocab_size} words)")
    print(f"   - next_word_model.pt (sample LSTM model)")
    print(f"")
    print(f"ℹ️  These are dummy files for testing the application structure.")
    print(f"   Replace them with your trained model and vocabulary.")

if __name__ == "__main__":
    create_sample_files()
