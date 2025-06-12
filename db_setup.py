# db_setup.py
import sqlite3
from werkzeug.security import generate_password_hash

conn = sqlite3.connect('users.db')
cursor = conn.cursor()

# Таблица пользователей
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
)
''')

# Таблица портфеля
cursor.execute('''
CREATE TABLE IF NOT EXISTS portfolio (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    count INTEGER NOT NULL,
    avg_price REAL NOT NULL,
    broker TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id)
)
''')

# Таблица транзакций
cursor.execute('''
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    count INTEGER NOT NULL,
    price REAL NOT NULL,
    broker TEXT,
    datetime TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id)
)
''')

# Добавим тестового пользователя (username: admin, password: admin)
password_hash = generate_password_hash('admin')
cursor.execute('''
INSERT OR IGNORE INTO users (username, password)
VALUES (?, ?)
''', ('admin', password_hash))

# Получаем id тестового пользователя
cursor.execute('SELECT id FROM users WHERE username = ?', ('admin',))
user_id = cursor.fetchone()[0]

# Добавим запись в портфель
cursor.execute('''
INSERT OR IGNORE INTO portfolio (user_id, ticker, count, avg_price, broker)
VALUES (?, ?, ?, ?, ?)
''', (user_id, 'SBER', 10, 250, 'Тинькофф'))

# Добавим запись в транзакции
cursor.execute('''
INSERT OR IGNORE INTO transactions (user_id, ticker, count, price, broker, datetime)
VALUES (?, ?, ?, ?, ?, ?)
''', (user_id, 'SBER', 10, 249.5, 'Тинькофф', '2024-05-24 20:00:00'))

conn.commit()
conn.close()
