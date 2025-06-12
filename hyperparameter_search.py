import os
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
from moexalgo import Stock

# Настройки GPU/CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Путь к директориям
MODEL_PATH = "models"
SCALER_PATH = "scalers"
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(SCALER_PATH, exist_ok=True)

# --- Подготовка данных ---
def add_technical_indicators(df):
    df["sma_5"] = df["close"].rolling(window=5).mean()
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df.bfill(inplace=True)
    return df

def get_prices(ticker_symbol):
    stock = Stock(ticker_symbol)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=1500)
    df = stock.candles(start=start_date, end=end_date, period='1d')
    df = df[['open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)
    df = add_technical_indicators(df)
    return df

class StockDatasetMultiStep(Dataset):
    def __init__(self, x_data, y_data, seq_length, forecast_length):
        self.x_data = x_data
        self.y_data = y_data
        self.seq_length = seq_length
        self.forecast_length = forecast_length

    def __len__(self):
        return max(0, len(self.x_data) - self.seq_length - self.forecast_length)

    def __getitem__(self, idx):
        x = self.x_data[idx:idx + self.seq_length]
        y = self.y_data[idx + self.seq_length:idx + self.seq_length + self.forecast_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y.squeeze(), dtype=torch.float32)

# --- Модели ---
class LSTMModelMultiStep(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, forecast_length, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, forecast_length)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# --- Тренировка и оценка ---
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, patience=3):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float('inf')
    stop_counter = 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                val_loss += criterion(out, yb).item()
        val_loss /= len(val_loader)
        if val_loss < best_val:
            best_val = val_loss
            stop_counter = 0
        else:
            stop_counter += 1
            if stop_counter >= patience:
                break
    return best_val

def run_search(ticker="SBER", seq_range=(10, 15), hidden_range=(64, 128),
               layers_range=(2, 3), forecast_range=(1, 3), dropout_range=(0.2, 0.3)):
    df = get_prices(ticker)
    x_features = df[['open', 'high', 'low', 'close', 'volume', 'sma_5', 'ema_12']].values
    y_target = df[['close']].values
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    x_scaled = x_scaler.fit_transform(x_features)
    y_scaled = y_scaler.fit_transform(y_target)

    results = []

    for seq_len in seq_range:
        for forecast_len in forecast_range:
            dataset = StockDatasetMultiStep(x_scaled, y_scaled, seq_len, forecast_len)
            if len(dataset) < 10:
                continue
            x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, shuffle=False)
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=False)
            train_ds = StockDatasetMultiStep(x_train, y_train, seq_len, forecast_len)
            val_ds = StockDatasetMultiStep(x_val, y_val, seq_len, forecast_len)
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=32)
            for hidden in hidden_range:
                for layers in layers_range:
                    for dropout in dropout_range:
                        model = LSTMModelMultiStep(input_size=7, hidden_size=hidden,
                                                   num_layers=layers, forecast_length=forecast_len,
                                                   dropout=dropout).to(DEVICE)
                        val_loss = train_model(model, train_loader, val_loader)
                        results.append({
                            'seq_length': seq_len,
                            'hidden_size': hidden,
                            'num_layers': layers,
                            'forecast_length': forecast_len,
                            'dropout': dropout,
                            'val_loss': val_loss
                        })
    # Сортировка результатов по значению функции потерь
    results.sort(key=lambda x: x['val_loss'])
    return results

if __name__ == "__main__":
    res = run_search()
    if res:
        best = res[0]
        print("Лучшие параметры:")
        for k, v in best.items():
            print(f"{k}: {v}")
    else:
        print("Недостаточно данных для поиска оптимальных параметров")
