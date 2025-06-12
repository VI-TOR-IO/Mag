import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from moexalgo import Stock
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split

# Параметры
TICKERS = ["SBER"]
FORECAST_DAYS_LIST = [1, 3, 7, 30]
SEQ_LENGTH = 160
MODEL_PATH = "models"
SCALER_PATH = "scalers"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(SCALER_PATH, exist_ok=True)

# Функция добавления технических индикаторов
def add_technical_indicators(df):
    df["sma_5"] = df["close"].rolling(window=5).mean()
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df.bfill(inplace=True)
    return df

# Класс для создания набора данных для многодневного прогнозирования
class StockDatasetMultiStep(Dataset):
    def __init__(self, x_data, y_data, seq_length, forecast_length):
        self.x_data = x_data
        self.y_data = y_data
        self.seq_length = seq_length
        self.forecast_length = forecast_length

    def __len__(self):
        return len(self.x_data) - self.seq_length - self.forecast_length

    def __getitem__(self, index):
        x = self.x_data[index:index+self.seq_length]
        y = self.y_data[index+self.seq_length:index+self.seq_length+self.forecast_length]
        y = y.squeeze()
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Модель LSTM для многодневного прогнозирования
class LSTMModelMultiStep(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=2, forecast_length=7, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, forecast_length)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# Функция для получения исторических данных по акциям
def get_prices(ticker_symbol):
    stock = Stock(ticker_symbol)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=400)
    df = stock.candles(start=start_date, end=end_date)
    df = df[['open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)
    df = add_technical_indicators(df)
    return df

# Реализация ранней остановки
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        """
        patience: сколько эпох терпеть без улучшения
        delta: минимальное улучшение для того, чтобы считаться улучшением
        """
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Функция тренировки с ранней остановкой и оценкой на тестовых данных
def train_lstm_multistep_with_early_stopping(model, train_loader, val_loader, num_epochs=50, lr=0.001, patience=5):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Приведение меток к форме (batch_size, 1), если они одномерные
            if y_batch.ndim == 1:
                y_batch = y_batch.unsqueeze(1)  # Приводим метки к нужной размерности (batch_size, 1)

            output = model(X_batch)

            # Приводим выход модели к нужной размерности (batch_size, 1)
            if output.shape != y_batch.shape:
                output = output.view(-1, 1)  # Приводим выход модели к размерности (batch_size, 1)

            loss = criterion(output, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Оценка на валидационном наборе
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # Приведение меток к нужной размерности
                if y_batch.ndim == 1:
                    y_batch = y_batch.unsqueeze(1)

                output = model(X_batch)

                # Приведение выходных данных к нужной размерности
                if output.shape != y_batch.shape:
                    output = output.view(-1, 1)

                val_loss += criterion(output, y_batch).item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.6f}, Val Loss: {val_loss:.6f}")

        # Проверка на раннюю остановку
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

# Оценка модели на тестовых данных
def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            y_true.append(y_batch.cpu().numpy())
            y_pred.append(output.cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))

    print(f"Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}")
    return mae, rmse

# Тренировка и сохранение моделей для разных дней прогнозирования
for ticker in TICKERS:
    print(f"\n=== {ticker} ===")
    df = get_prices(ticker)

    x_features = df[['open', 'high', 'low', 'close', 'volume', 'sma_5', 'ema_12']].values
    y_target = df[['close']].values

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    x_scaled = x_scaler.fit_transform(x_features)
    y_scaled = y_scaler.fit_transform(y_target)

    for forecast_days in FORECAST_DAYS_LIST:
        print(f"→ Обучение модели {forecast_days}d вперед")
        dataset = StockDatasetMultiStep(x_scaled, y_scaled, SEQ_LENGTH, forecast_days)
        if len(dataset) < 10:
            print(f"Недостаточно данных для {ticker} {forecast_days} дней, пропуск...")
            continue

        # Разделим данные на обучающую, валидационную и тестовую выборки
        x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, shuffle=False)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=False)

        train_dataset = StockDatasetMultiStep(x_train, y_train, SEQ_LENGTH, forecast_days)
        val_dataset = StockDatasetMultiStep(x_val, y_val, SEQ_LENGTH, forecast_days)
        test_dataset = StockDatasetMultiStep(x_test, y_test, SEQ_LENGTH, forecast_days)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Создание и тренировка модели
        model = LSTMModelMultiStep(input_size=7, forecast_length=forecast_days).to(device)
        train_lstm_multistep_with_early_stopping(model, train_loader, val_loader, num_epochs=50, lr=0.001, patience=5)

        # Сохранение модели и скейлеров
        model_path = f"{MODEL_PATH}/{ticker}_model_{forecast_days}d.pth"
        torch.save(model.state_dict(), model_path)

        joblib.dump(x_scaler, f"{SCALER_PATH}/{ticker}_x_scaler_{forecast_days}d.pkl")
        joblib.dump(y_scaler, f"{SCALER_PATH}/{ticker}_y_scaler_{forecast_days}d.pkl")

        print(f"Модель сохранена: {model_path}")

        # Оценка модели на тестовых данных
        evaluate_model(model, test_loader)
