import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import gradio as gr

def load_data(ticker='AAPL', start_date='2020-01-01', end_date='2025-01-01'):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data[['Close']]

def preprocess_data(data, seq_len=60):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data.values)

    seqs, labels = [], []
    for i in range(len(scaled_data) - seq_len):
        seqs.append(scaled_data[i:i + seq_len])
        labels.append(scaled_data[i + seq_len])

    return torch.tensor(seqs, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32), scaler

class StockDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=1, num_heads=4, num_layers=2, dim_feedforward=256, d_model=64):
        super().__init__()
        # Ensure d_model is divisible by num_heads
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        # Project input_dim (1) to d_model for the transformer
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model) # Positional encoding uses d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1) # Final linear layer from d_model to output (1)

    def forward(self, x):
        x = self.input_projection(x) # Project the input
        x = self.positional_encoding(x)
        x = self.transformer(x)
        return self.fc(x[:, -1])

def train_model(model, train_loader, epochs=10, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(seqs)
            loss = criterion(output.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

def predict_next_close(ticker='AAPL'):
    data = load_data(ticker)
    if data.empty:
        return f"Could not load data for {ticker}. Please check the ticker symbol."

    seqs, labels, scaler = preprocess_data(data)

    # Initialize the model with an appropriate d_model (e.g., 64)
    # Ensure d_model is divisible by num_heads (default 4)
    model = TimeSeriesTransformer(d_model=64)
    
    train_loader = DataLoader(StockDataset(seqs, labels), batch_size=32, shuffle=True)
    train_model(model, train_loader, epochs=5)

    model.eval()
    with torch.no_grad():
        last_seq = seqs[-1].unsqueeze(0)
        pred = model(last_seq).item()
        pred = scaler.inverse_transform([[pred]])[0][0]

    return f"Predicted next close for {ticker}: ${pred:.2f}"

gr.Interface(fn=predict_next_close,
             inputs=gr.Textbox(label="Stock Ticker", value="AAPL"),
             outputs=gr.Textbox(label="Predicted Close Price"),
             title="Stock Price Forecast (Transformer)",
             description="Enter a stock ticker (e.g., AAPL, TSLA) to predict its next closing price using a Transformer model."
).launch(share=False)