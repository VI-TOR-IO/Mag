# train_models.py
import os
import numpy as np
from datetime import datetime, timedelta
from moexalgo import Stock
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

TICKERS = ["SBER"]#, "GAZP", "LKOH", "VTBR", "ROSN"]
FORECAST_DAYS_LIST = [1, 3, 7, 30]
SEQ_LENGTH = 60
MODEL_PATH = "models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODEL_PATH, exist_ok=True)

class StockDatasetMultiStep(Dataset):
    def __init__(self, data, seq_length, forecast_length):
        self.data = data
        self.seq_length = seq_length
        self.forecast_length = forecast_length

    def __len__(self):
        return len(self.data) - self.seq_length - self.forecast_length

    def __getitem__(self, index):
        x = self.data[index:index+self.seq_length]
        y = self.data[index+self.seq_length:index+self.seq_length+self.forecast_length, 3]  # прогноз по close
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class LSTMModelMultiStep(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, forecast_length=7):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, forecast_length)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_lstm_multistep(model, train_loader, num_epochs=20, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        y_true = y_batch.detach().cpu().numpy().flatten()
        y_pred = output.detach().cpu().numpy().flatten()
        mae = mean_absolute_error(y_true, y_pred)
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.6f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

def get_prices(ticker_symbol):
    stock = Stock(ticker_symbol)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    df = stock.candles(start=start_date, end=end_date)
    df = df[['open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)
    return df

for ticker in TICKERS:
    print(f"\n=== {ticker} ===")
    df = get_prices(ticker)
    features = df[['open', 'high', 'low', 'close', 'volume']].values
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    for forecast_days in FORECAST_DAYS_LIST:
        print(f"→ Обучение модели {forecast_days}d вперед")
        dataset = StockDatasetMultiStep(features_scaled, SEQ_LENGTH, forecast_days)
        if len(dataset) < 10:
            print(f"Недостаточно данных для {ticker} {forecast_days} дней, пропуск...")
            continue
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

        model = LSTMModelMultiStep(input_size=5, forecast_length=forecast_days).to(device)
        train_lstm_multistep(model, train_loader, num_epochs=20)

        model_file = f"{MODEL_PATH}/{ticker}_model_{forecast_days}d.pth"
        torch.save(model.state_dict(), model_file)
        print(f"Модель сохранена: {model_file}")
