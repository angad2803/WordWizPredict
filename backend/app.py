"""
FastAPI backend for next-word prediction using PyTorch model.
"""

import pickle
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and vocabulary
model = None
vocab = None
word_to_idx = None
idx_to_word = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        load_model_and_vocab()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    yield
    # Shutdown
    logger.info("Application shutdown")

app = FastAPI(title="Next Word Prediction API", version="1.0.0", lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
        "http://localhost:3000", 
        "http://localhost:8080", 
        "http://localhost:8082",
        "https://*.vercel.app",
        "https://*.netlify.app",
        "https://*.railway.app",
        "*"  # Allow all origins for deployment (configure specific domains in production)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    text: str
    top_k: int = 5

class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    input_text: str

class NextWordLSTM(torch.nn.Module):
    """
    Simple LSTM model for next word prediction.
    Adjust this class based on your actual model architecture.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(NextWordLSTM, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)
        self.dropout = torch.nn.Dropout(0.3)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.dropout(lstm_out)
        
        # Use the last timestep's output for prediction
        # Shape: (batch_size, sequence_length, hidden_dim) -> (batch_size, hidden_dim)
        if lstm_out.size(1) > 1:
            # If we have multiple timesteps, use the last one
            output = self.fc(output[:, -1, :])
        else:
            # If we have only one timestep, use it
            output = self.fc(output[:, 0, :])
        
        return output

def load_model_and_vocab():
    """Load the trained model and vocabulary."""
    global model, vocab, word_to_idx, idx_to_word
    
    try:
        # Load vocabulary
        vocab_path = Path("vocab.pkl")
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
            
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
            
        # Handle different vocabulary formats
        if isinstance(vocab, dict):
            if 'word_to_idx' in vocab and 'idx_to_word' in vocab:
                word_to_idx = vocab['word_to_idx']
                idx_to_word = vocab['idx_to_word']
            else:
                word_to_idx = vocab
                idx_to_word = {idx: word for word, idx in vocab.items()}
        elif isinstance(vocab, list):
            word_to_idx = {word: idx for idx, word in enumerate(vocab)}
            idx_to_word = {idx: word for idx, word in enumerate(vocab)}
        else:
            raise ValueError("Unsupported vocabulary format")
            
        vocab_size = len(word_to_idx)
        logger.info(f"Loaded vocabulary with {vocab_size} words")
        
        # Load model - try both file names
        model_path = None
        for filename in ["lstm_model.pt", "next_word_model.pt"]:
            path = Path(filename)
            if path.exists():
                model_path = path
                break
                
        if model_path is None:
            raise FileNotFoundError("Model file not found. Expected 'lstm_model.pt' or 'next_word_model.pt'")
            
        logger.info(f"Loading model from: {model_path}")
        
        # Try to load the model state dict
        model_data = torch.load(model_path, map_location='cpu')
        
        # Handle different model save formats
        if isinstance(model_data, dict) and 'state_dict' in model_data:
            # Model saved with additional metadata
            state_dict = model_data['state_dict']
            model = NextWordLSTM(
                vocab_size=vocab_size,
                embedding_dim=model_data.get('embedding_dim', 128),
                hidden_dim=model_data.get('hidden_dim', 256),
                num_layers=model_data.get('num_layers', 2)
            )
            model.load_state_dict(state_dict)
        else:
            # State dict only - infer architecture from tensor shapes
            state_dict = model_data
            
            # Infer dimensions from the state dict
            embedding_dim = None
            hidden_dim = None
            num_layers = 1  # Default, will be adjusted based on LSTM layers found
            
            # Get embedding dimension from embedding.weight
            if 'embedding.weight' in state_dict:
                embedding_dim = state_dict['embedding.weight'].shape[1]
                
            # Get hidden dimension from fc.weight (output layer)
            if 'fc.weight' in state_dict:
                hidden_dim = state_dict['fc.weight'].shape[1]
                
            # Count LSTM layers by looking for weight patterns
            lstm_layers = set()
            for key in state_dict.keys():
                if key.startswith('lstm.weight_ih_l'):
                    layer_num = int(key.split('_l')[1])
                    lstm_layers.add(layer_num)
            if lstm_layers:
                num_layers = max(lstm_layers) + 1
                
            logger.info(f"Inferred model architecture: embedding_dim={embedding_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}")
            
            if embedding_dim is None or hidden_dim is None:
                # Fallback to trying different configurations
                possible_configs = [
                    {"embedding_dim": 100, "hidden_dim": 150, "num_layers": 1},  # Your specific model
                    {"embedding_dim": 128, "hidden_dim": 256, "num_layers": 2},
                    {"embedding_dim": 100, "hidden_dim": 128, "num_layers": 1},
                    {"embedding_dim": 64, "hidden_dim": 64, "num_layers": 1},
                    {"embedding_dim": 50, "hidden_dim": 50, "num_layers": 1},
                ]
                
                model = None
                for config in possible_configs:
                    try:
                        test_model = NextWordLSTM(
                            vocab_size=vocab_size,
                            **config
                        )
                        test_model.load_state_dict(state_dict)
                        model = test_model
                        logger.info(f"Successfully loaded model with config: {config}")
                        break
                    except Exception as e:
                        logger.debug(f"Failed to load with config {config}: {e}")
                        continue
                        
                if model is None:
                    raise ValueError("Could not load model with any known architecture configuration")
            else:
                # Use inferred dimensions
                model = NextWordLSTM(
                    vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers
                )
                model.load_state_dict(state_dict)
        
        model.eval()
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model or vocabulary: {e}")
        raise

def tokenize_text(text: str) -> List[int]:
    """Tokenize input text using the loaded vocabulary."""
    words = text.lower().strip().split()
    tokens = []
    
    for word in words:
        if word in word_to_idx:
            tokens.append(word_to_idx[word])
        else:
            # Handle unknown words - you might want to use <UNK> token
            if '<UNK>' in word_to_idx:
                tokens.append(word_to_idx['<UNK>'])
            elif '<unk>' in word_to_idx:
                tokens.append(word_to_idx['<unk>'])
            else:
                # Skip unknown words if no UNK token
                continue
    
    return tokens

def predict_next_words(text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Predict the next words given input text."""
    if model is None or vocab is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    
    try:
        # Tokenize input
        tokens = tokenize_text(text)
        if not tokens:
            raise HTTPException(status_code=400, detail="No valid tokens found in input text")
        
        # Convert to tensor
        input_tensor = torch.tensor([tokens], dtype=torch.long)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=-1)
            
            # Debug logging
            logger.info(f"Input tokens: {tokens}")
            logger.info(f"Output shape: {outputs.shape}")
            logger.info(f"Raw output sample (first 10): {outputs[0][:10].tolist()}")
            logger.info(f"Probability sum: {probabilities[0].sum().item()}")
            logger.info(f"Max probability: {probabilities[0].max().item()}")
            logger.info(f"Min probability: {probabilities[0].min().item()}")
            
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities[0], top_k)
        
        # Debug top predictions
        logger.info(f"Top {top_k} probabilities: {top_probs.tolist()}")
        logger.info(f"Top {top_k} indices: {top_indices.tolist()}")
        
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            word = idx_to_word.get(idx.item(), "<UNK>")
            predictions.append({
                "word": word,
                "probability": float(prob),
                "confidence": float(prob * 100)
            })
            
        logger.info(f"Final predictions: {predictions}")
        
        return predictions
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Next Word Prediction API is running"}

@app.get("/health")
async def health_check():
    """Health check with model status."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "vocab_loaded": vocab is not None,
        "vocab_size": len(word_to_idx) if word_to_idx else 0
    }

@app.get("/model-info")
async def get_model_info():
    """Get detailed model information."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "architecture": "LSTM",
        "vocab_size": len(word_to_idx) if word_to_idx else 0,
        "embedding_dim": model.embedding.embedding_dim if hasattr(model, 'embedding') else "Unknown",
        "hidden_dim": model.lstm.hidden_size if hasattr(model, 'lstm') else "Unknown",
        "num_layers": model.lstm.num_layers if hasattr(model, 'lstm') else "Unknown",
        "sample_vocabulary": list(word_to_idx.keys())[:20] if word_to_idx else [],
        "model_parameters": sum(p.numel() for p in model.parameters()) if model else 0
    }

@app.post("/analyze", response_model=dict)
async def analyze_text(request: PredictionRequest):
    """Analyze text and return detailed processing information."""
    if model is None or vocab is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    
    try:
        # Tokenize input
        tokens = tokenize_text(request.text)
        if not tokens:
            raise HTTPException(status_code=400, detail="No valid tokens found in input text")
        
        # Get word tokens for display
        word_tokens = request.text.lower().strip().split()
        
        # Convert to tensor
        input_tensor = torch.tensor([tokens], dtype=torch.long)
        
        # Get predictions with intermediate results
        with torch.no_grad():
            # Get embeddings
            embeddings = model.embedding(input_tensor)
            
            # Get LSTM outputs
            lstm_out, (hidden, cell) = model.lstm(embeddings)
            
            # Get final outputs
            outputs = model.fc(lstm_out[:, -1, :])
            probabilities = F.softmax(outputs, dim=-1)
            
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities[0], request.top_k)
        
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            word = idx_to_word.get(idx.item(), "<UNK>")
            predictions.append({
                "word": word,
                "probability": float(prob),
                "confidence": float(prob * 100)
            })
        
        return {
            "input_text": request.text,
            "word_tokens": word_tokens,
            "token_ids": tokens,
            "sequence_length": len(tokens),
            "embedding_shape": list(embeddings.shape),
            "lstm_output_shape": list(lstm_out.shape),
            "predictions": predictions,
            "total_vocabulary_considered": len(word_to_idx)
        }
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/debug")
async def debug_prediction(request: PredictionRequest):
    """Debug endpoint to see raw model outputs."""
    if model is None or vocab is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Tokenize input
        tokens = tokenize_text(request.text)
        if not tokens:
            raise HTTPException(status_code=400, detail="No valid tokens found")
        
        # Convert to tensor
        input_tensor = torch.tensor([tokens], dtype=torch.long)
        
        # Get raw outputs
        with torch.no_grad():
            raw_outputs = model(input_tensor)
            probabilities = F.softmax(raw_outputs, dim=-1)
            
        # Get top-10 for debugging
        top_probs, top_indices = torch.topk(probabilities[0], 10)
        
        debug_info = {
            "input_tokens": tokens,
            "raw_output_shape": list(raw_outputs.shape),
            "raw_output_sample": raw_outputs[0][:10].tolist(),  # First 10 raw values
            "probability_sum": float(probabilities[0].sum()),
            "max_probability": float(probabilities[0].max()),
            "min_probability": float(probabilities[0].min()),
            "top_10_predictions": [
                {
                    "word": idx_to_word.get(idx.item(), "<UNK>"),
                    "probability": float(prob),
                    "confidence": float(prob * 100),
                    "raw_output": float(raw_outputs[0][idx])
                }
                for prob, idx in zip(top_probs, top_indices)
            ]
        }
        
        return debug_info
        
    except Exception as e:
        logger.error(f"Debug error: {e}")
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict next words for given input text."""
    predictions = predict_next_words(request.text, request.top_k)
    
    return PredictionResponse(
        predictions=predictions,
        input_text=request.text
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
