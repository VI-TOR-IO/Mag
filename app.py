from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
import locale
import numpy as np
import pandas as pd
import sqlite3
import time
from datetime import datetime, timedelta
from moexalgo import Stock
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
import plotly.offline as pyo
from torch.utils.data import Dataset
from flask import session
from sklearn.metrics import mean_absolute_error, mean_squared_error
from werkzeug.security import generate_password_hash, check_password_hash
from flask import request, redirect, url_for, flash, session
import joblib

app = Flask(__name__)
app.secret_key = 'Trader'

TICKERS = ["SBER", "GAZP", "LKOH", "VTBR", "ROSN"]
SEQ_LENGTH = 160
MODEL_PATH = "models"
SCALER_PATH = "scalers"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
locale.setlocale(locale.LC_TIME, 'Russian_Russia.1251')

class StockDatasetMultiStep(Dataset):
    def __init__(self, data, seq_length, forecast_length):
        self.data = data
        self.seq_length = seq_length
        self.forecast_length = forecast_length

    def __len__(self):
        return len(self.data) - self.seq_length - self.forecast_length

    def __getitem__(self, index):
        x = self.data[index:index+self.seq_length]
        y = self.data[index+self.seq_length:index+self.seq_length+self.forecast_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y.squeeze(), dtype=torch.float32)

class LSTMModelMultiStep(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=2, forecast_length=7):
        super(LSTMModelMultiStep, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size * 2, forecast_length)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


def get_prices(ticker_symbol):
    stock = Stock(ticker_symbol)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=400)
    df = stock.candles(start=start_date, end=end_date)
    df['begin'] = pd.to_datetime(df['begin'])
    df = df[['begin', 'open', 'high', 'low', 'close', 'volume']]
    df = add_technical_indicators(df)
    return df


def predict_lstm_multistep(data, model, x_scaler, y_scaler, seq_length, forecast_length):
    model.eval()
    last_seq = data[-seq_length:]
    last_seq_scaled = x_scaler.transform(last_seq)
    input_seq = torch.tensor(last_seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_scaled = model(input_seq).cpu().numpy().flatten()
    
    # Используем y_scaler для преобразования предсказания
    pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    return pred

def backtest_lstm_multistep(data, model, x_scaler, y_scaler, seq_length, forecast_length, real_dates):
    model.eval()
    close_column_index = 3
    i = len(data) - forecast_length - seq_length - 15
    window = data[i - seq_length:i]
    if window.shape[0] < seq_length:
        return [], {'mae': None, 'rmse': None}, [], [], []

    scaled_window = x_scaler.transform(window)
    input_seq = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_scaled = model(input_seq).cpu().numpy().flatten()

    inverse_input = np.zeros((forecast_length, 5))
    inverse_input[:, close_column_index] = pred_scaled
    pred = y_scaler.inverse_transform(inverse_input)[:, close_column_index]

    real = data[i:i + forecast_length, close_column_index]
    if len(real) < forecast_length or len(real) == 0:
        return [], {'mae': None, 'rmse': None}, [], [], []

    dates = [datetime.today() - timedelta(days=forecast_length - offset + 45) for offset in range(forecast_length)]

    mae = mean_absolute_error(real, pred)
    rmse = np.sqrt(mean_squared_error(real, pred))

    predictions = [{'date': d.strftime('%a, %d %B'), 'value': round(p, 2)} for d, p in zip(dates, pred)]
    return predictions, {'mae': round(mae, 4), 'rmse': round(rmse, 4)}, pred, real, dates


def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        conn = get_db_connection()
        try:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            flash('Регистрация прошла успешно! Войдите в аккаунт.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Имя пользователя уже занято.', 'danger')
        finally:
            conn.close()
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password_input = request.form['password']
        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        conn.close()
        if user and check_password_hash(user['password'], password_input):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Вы вошли в систему.', 'success')
            return redirect(url_for('index'))
        else:
            flash('Неверное имя пользователя или пароль.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Вы вышли из аккаунта.', 'info')
    return redirect(url_for('login'))

models_cache = {}
def load_all_models():
    for ticker in TICKERS:
        for days in [1, 3, 7, 30]:
            model_path = f"{MODEL_PATH}/{ticker}_model_{days}d.pth"
            if os.path.exists(model_path):
                model = LSTMModelMultiStep(input_size=7, forecast_length=days).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                models_cache[(ticker, days)] = model

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', tickers=TICKERS)

# --- Профиль пользователя ---
@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']

    conn = get_db_connection()
    has_tinkoff = conn.execute(
        "SELECT 1 FROM transactions WHERE user_id = ? AND broker = 'Тинькофф' LIMIT 1", (user_id,)
    ).fetchone() is not None
    has_sber = conn.execute(
        "SELECT 1 FROM transactions WHERE user_id = ? AND broker = 'Сбер' LIMIT 1", (user_id,)
    ).fetchone() is not None

    # История всех покупок
    transactions = conn.execute(
        "SELECT id, datetime, ticker, count, price, broker FROM transactions WHERE user_id = ? ORDER BY datetime DESC",
        (user_id,)
    ).fetchall()

    # Портфель — агрегируем по тикеру+брокеру
    portfolio_data = {}
    for t in transactions:
        key = (t['ticker'], t['broker'])
        if key not in portfolio_data:
            portfolio_data[key] = {'ticker': t['ticker'], 'broker': t['broker'], 'count': 0, 'sum_price': 0}
        portfolio_data[key]['count'] += t['count']
        portfolio_data[key]['sum_price'] += t['count'] * t['price']
    portfolio = []
    for v in portfolio_data.values():
        if v['count'] > 0:
            avg_price = v['sum_price'] / v['count']
            portfolio.append({'ticker': v['ticker'], 'count': v['count'], 'avg_price': round(avg_price, 2), 'broker': v['broker']})

    conn.close()
    brokers = [
        {"name": "Тинькофф", "icon_url": url_for('static', filename='tinkoff.png'), "connected": has_tinkoff}
    ]
    return render_template(
        'profile.html',
        brokers=brokers,
        portfolio=portfolio,
        transactions=transactions,
        has_tinkoff=has_tinkoff,
        has_sber=has_sber
    )

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    selected_ticker = data.get('ticker')
    forecast_days = int(data.get('days', 1))

    # Получаем данные по выбранной акции
    df = get_prices(selected_ticker)

    features = df[['open', 'high', 'low', 'close', 'volume', 'sma_5', 'ema_12']].values
    real_dates = df['begin'].tolist()
    prices = df['close'].values

    x_scaler_path = f"{SCALER_PATH}/{selected_ticker}_x_scaler_{forecast_days}d.pkl"
    y_scaler_path = f"{SCALER_PATH}/{selected_ticker}_y_scaler_{forecast_days}d.pkl"
    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)

    scaled_features = x_scaler.transform(features)

    MIN_SEQ_LENGTH = 30
    max_possible_seq = len(prices) - forecast_days
    seq_length = min(max_possible_seq, 30) if max_possible_seq >= MIN_SEQ_LENGTH else MIN_SEQ_LENGTH

    if len(prices) < seq_length + forecast_days:
        return jsonify({'error': 'Недостаточно данных для прогноза'})

    model = models_cache.get((selected_ticker, forecast_days))
    if model is None:
        return jsonify({'error': 'Модель не найдена'})

    # Прогноз вперед
    future_predictions = predict_lstm_multistep(features, model, x_scaler, y_scaler, seq_length, forecast_days)
    # Backtest для ошибок
    backtest_predictions, backtest_error, pred_back, real_back, back_dates = backtest_lstm_multistep(
        features, model, x_scaler, y_scaler, seq_length, forecast_days, real_dates
    )

    # График цен
    forecast_x = [datetime.today() + timedelta(days=i - 1) for i in range(forecast_days)]
    # График цен
    trace_real = go.Scatter(
        x=[datetime.today() - timedelta(days=len(prices) - i) for i in range(len(prices))],
        y=df['close'],
        mode='lines',
        name='Исторические данные',
        line=dict(color='cyan')
    )
    trace_pred = go.Scatter(x=forecast_x, y=future_predictions, mode='lines+markers',
                            name='Прогноз (вперёд)', line=dict(color='orange'))
    traces = [trace_real, trace_pred]

    if backtest_predictions and pred_back is not None and real_back is not None and back_dates:
        trace_backtest = go.Scatter(x=back_dates, y=pred_back, mode='lines+markers',
                                    name='Прогноз назад', line=dict(color='magenta'))
        trace_backtest_actual = go.Scatter(x=back_dates, y=real_back, mode='lines+markers',
                                           name='Реальные значения (назад)', line=dict(color='lime'))
        traces += [trace_backtest_actual, trace_backtest]

    N = 30
    if len(prices) > N:
        x_range = [
            (datetime.today() - timedelta(days=N - 1)).strftime('%Y-%m-%d'),
            datetime.today().strftime('%Y-%m-%d')
        ]
    else:
        x_range = None

    layout = go.Layout(
        title=f'{selected_ticker} - Прогноз на {forecast_days} дней',
        template="plotly_dark",
        xaxis=dict(
            title='Дата',
            tickangle=45,
            type='date',
            range=x_range,
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(title='Цена'),
        hovermode='x unified',
        autosize=True,
        width=None,
        height=600
    )
    fig = go.Figure(data=traces, layout=layout)
    plot_div = pyo.plot(fig, output_type='div', config={'responsive': True})

    # График ошибок (рубли и проценты)
    error_plot_div = ""
    if backtest_predictions and pred_back is not None and real_back is not None and back_dates:
        trace_error = go.Scatter(
            x=back_dates,
            y=[p - r for p, r in zip(pred_back, real_back)],
            mode='lines+markers',
            name='Ошибка прогноза (руб.)',
            line=dict(color='red', dash='dot'),
            hovertemplate='Дата: %{x}, Ошибка: %{y:.2f} руб.<extra></extra>'
        )
        trace_percent = go.Scatter(
            x=back_dates,
            y=[(p - r) / r * 100 if r else 0 for p, r in zip(pred_back, real_back)],
            mode='lines+markers',
            name='Ошибка прогноза (%)',
            yaxis='y2',
            line=dict(color='orange', dash='dot'),
            hovertemplate='Дата: %{x}, Ошибка: %{y:.2f}%<extra></extra>'
        )
        layout_error = go.Layout(
            title='Ошибка прогноза (в рублях и %)',
            template='plotly_dark',
            yaxis=dict(title='Ошибка (руб.)'),
            yaxis2=dict(title='Ошибка (%)', overlaying='y', side='right'),
            xaxis=dict(title='Дата', tickangle=45, type='date'),
            legend=dict(x=0.01, y=1.15, orientation='h'),
            hovermode='x unified'
        )
        fig_error = go.Figure(data=[trace_error, trace_percent], layout=layout_error)
        error_plot_div = pyo.plot(fig_error, output_type='div', config={'responsive': True})

    forecast_dates = [(datetime.today() + timedelta(days=i)).strftime('%a, %d %B') for i in range(forecast_days)]
    predictions = [{'date': date, 'value': round(val, 2)} for date, val in zip(forecast_dates, future_predictions)]

    # --- Рекомендация Покупка/Продажа ---
    RECOMMENDATION_THRESHOLD = 2.0  # %
    last_real_price = prices[-1]
    diff_percent = ((future_predictions[-1] - last_real_price) / last_real_price) * 100

    if diff_percent > RECOMMENDATION_THRESHOLD:
        recommendation = "Покупка"
    elif diff_percent < -RECOMMENDATION_THRESHOLD:
        recommendation = "Продажа"
    else:
        recommendation = "Держать"

    plot_html = render_template('partials/plot.html', plot_div=plot_div, error_plot_div=error_plot_div)
    predictions_html = render_template('partials/predictions.html',
                                       predictions=predictions,
                                       backtest_predictions=backtest_predictions,
                                       backtest_error=backtest_error)

    return jsonify({
        'plot_html': plot_html,
        'predictions_html': predictions_html,
        'recommendation': recommendation,
        'diff_percent': round(diff_percent, 2) if diff_percent is not None else None,
        'forecast_price': round(float(future_predictions[-1]), 2),
        'show_recommendation': True
    })

prices_cache = {}
CACHE_TTL = 6000  # 100 минут

def add_technical_indicators(df):
    df["sma_5"] = df["close"].rolling(window=5).mean()
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df.fillna(method='bfill', inplace=True)
    return df

def get_prices_cached(ticker_symbol):
    now = time.time()
    cache_entry = prices_cache.get(ticker_symbol)
    if cache_entry and now - cache_entry['timestamp'] < CACHE_TTL:
        return cache_entry['data']
    df = get_prices(ticker_symbol)
    prices_cache[ticker_symbol] = {'data': df, 'timestamp': now}
    return df

@app.route('/prices', methods=['GET'])
def prices():
    result = []
    for ticker in TICKERS:
        df = get_prices_cached(ticker)
        last_close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2] if len(df) > 1 else last_close
        change = last_close - prev_close
        percent = (change / prev_close) * 100 if prev_close != 0 else 0
        result.append({
            'symbol': ticker,
            'last': round(last_close, 2),
            'change': round(change, 2),
            'percent': round(percent, 2)
        })
    return jsonify(result)

# CREATE: Добавить акцию в портфель
@app.route('/portfolio/add', methods=['POST'])
def add_stock():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    ticker = request.form['ticker'].strip().upper()
    count = int(request.form['count'])
    price = float(request.form['price'])
    broker = request.form['broker']
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    conn = get_db_connection()
    # Записываем покупку в историю
    conn.execute(
        "INSERT INTO transactions (user_id, datetime, ticker, count, price, broker) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, now, ticker, count, price, broker)
    )
    conn.commit()
    conn.close()
    flash('Покупка добавлена в историю.', 'success')
    return redirect(url_for('profile'))

# UPDATE: Изменить данные об акции
@app.route('/transactions/edit/<int:transaction_id>', methods=['POST'])
def edit_transaction(transaction_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    ticker = request.form['ticker'].strip().upper()
    count = int(request.form['count'])
    price = float(request.form['price'])
    broker = request.form['broker']
    conn = get_db_connection()
    conn.execute(
        "UPDATE transactions SET ticker=?, count=?, price=?, broker=? WHERE id=?",
        (ticker, count, price, broker, transaction_id)
    )
    conn.commit()
    conn.close()
    flash('Операция обновлена.', 'success')
    return redirect(url_for('profile'))

# DELETE: Удалить акцию из портфеля
@app.route('/transactions/delete/<int:transaction_id>', methods=['POST'])
def delete_transaction(transaction_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    conn = get_db_connection()
    conn.execute("DELETE FROM transactions WHERE id=?", (transaction_id,))
    conn.commit()
    conn.close()
    flash('Операция удалена из истории.', 'info')
    return redirect(url_for('profile'))

if __name__ == '__main__':
    load_all_models()
    app.run(debug=True)
