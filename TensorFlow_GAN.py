import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from config import data_base_path
import random
import requests
import retrying

forecast_price = {}

binance_data_path = os.path.join(data_base_path, "binance/futures-klines")
MAX_DATA_SIZE = 100  # Giới hạn số lượng dữ liệu tối đa khi lưu trữ
INITIAL_FETCH_SIZE = 100  # Số lượng nến lần đầu tải về

@retrying.retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=5)
def fetch_prices(symbol, interval="1m", limit=100, start_time=None, end_time=None):
    try:
        base_url = "https://fapi.binance.com"
        endpoint = f"/fapi/v1/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        url = base_url + endpoint
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f'Failed to fetch prices for {symbol} from Binance API: {str(e)}')
        raise e

def download_data(token):
    symbols = f"{token.upper()}USDT"
    interval = "5m"
    current_datetime = datetime.now()
    download_path = os.path.join(binance_data_path, token.lower())
    
    # Đường dẫn file CSV để lưu trữ
    file_path = os.path.join(download_path, f"{token.lower()}_5m_data.csv")
    # file_path = os.path.join(data_base_path, f"{token.lower()}_price_data.csv")

    # Kiểm tra xem file có tồn tại hay không
    if os.path.exists(file_path):
        # Tính thời gian bắt đầu cho 100 cây nến 5 phút
        start_time = int((current_datetime - timedelta(minutes=500)).timestamp() * 1000)
        end_time = int(current_datetime.timestamp() * 1000)
        new_data = fetch_prices(symbols, interval, 100, start_time, end_time)
    else:
        # Nếu file không tồn tại, tải về số lượng INITIAL_FETCH_SIZE nến
        start_time = int((current_datetime - timedelta(minutes=INITIAL_FETCH_SIZE*5)).timestamp() * 1000)
        end_time = int(current_datetime.timestamp() * 1000)
        new_data = fetch_prices(symbols, interval, INITIAL_FETCH_SIZE, start_time, end_time)

    # Chuyển dữ liệu thành DataFrame
    new_df = pd.DataFrame(new_data, columns=[
        "start_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", 
        "taker_buy_quote_asset_volume", "ignore"
    ])

    # Kiểm tra và đọc dữ liệu cũ nếu tồn tại
    if os.path.exists(file_path):
        old_df = pd.read_csv(file_path)
        # Kết hợp dữ liệu cũ và mới
        combined_df = pd.concat([old_df, new_df])
        # Loại bỏ các bản ghi trùng lặp dựa trên 'start_time'
        combined_df = combined_df.drop_duplicates(subset=['start_time'], keep='last')
    else:
        combined_df = new_df

    # Giới hạn số lượng dữ liệu tối đa
    if len(combined_df) > MAX_DATA_SIZE:
        combined_df = combined_df.iloc[-MAX_DATA_SIZE:]

    # Lưu dữ liệu đã kết hợp vào file CSV
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    combined_df.to_csv(file_path, index=False)
    print(f"Updated data for {token} saved to {file_path}. Total rows: {len(combined_df)}")

def format_data(token):
    path = os.path.join(binance_data_path, token.lower())
    file_path = os.path.join(path, f"{token.lower()}_5m_data.csv")

    if not os.path.exists(file_path):
        print(f"No data file found for {token}")
        return

    df = pd.read_csv(file_path)

    # Sử dụng các cột sau (đúng với dữ liệu bạn đã lưu)
    columns_to_use = [
        "start_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
    ]

    # Kiểm tra nếu tất cả các cột cần thiết tồn tại trong DataFrame
    if set(columns_to_use).issubset(df.columns):
        df = df[columns_to_use]
        df.columns = [
            "start_time", "open", "high", "low", "close", "volume",
            "end_time", "quote_asset_volume", "n_trades", 
            "taker_volume", "taker_volume_usd"
        ]
        df.index = pd.to_datetime(df["start_time"], unit='ms')
        df.index.name = "date"

        output_path = os.path.join(data_base_path, f"{token.lower()}_price_data.csv")
        df.sort_index().to_csv(output_path)
        print(f"Formatted data saved to {output_path}")
    else:
        print(f"Required columns are missing in {file_path}. Skipping this file.")

def train_model(token):
    # Load the token price data
    price_data = pd.read_csv(os.path.join(data_base_path, f"{token.lower()}_price_data.csv"))
    df = pd.DataFrame()

    # Convert 'date' to datetime
    price_data["date"] = pd.to_datetime(price_data["date"])

    # Set the date column as the index for resampling
    price_data.set_index("date", inplace=True)

    # Resample the data to 10-minute frequency and compute the mean price
    df = price_data.resample('10T').mean()

    # Prepare data for Linear Regression
    df = df.dropna()  # Loại bỏ các giá trị NaN (nếu có)
    X = np.array(range(len(df))).reshape(-1, 1)  # Sử dụng chỉ số thời gian làm đặc trưng
    y = df['close'].values  # Sử dụng giá đóng cửa làm mục tiêu

    # Khởi tạo mô hình Linear Regression
    # Побудова генератора
    def build_generator(latent_dim):
        model = Sequential([
            Dense(128, activation=LeakyReLU(0.2), input_dim=latent_dim),
            Dense(256, activation=LeakyReLU(0.2)),
            Dense(512, activation=LeakyReLU(0.2)),
            Dense(X.shape[1], activation='tanh')  # Вихідний шар має таку ж форму, як і вхідні дані
        ])
        return model

    # Побудова дискримінатора
    def build_discriminator():
        model = Sequential([
            Dense(512, activation=LeakyReLU(0.2), input_shape=(X.shape[1],)),
            Dense(256, activation=LeakyReLU(0.2)),
            Dense(1, activation='sigmoid')  # Вихідний шар дискримінатора
        ])
        return model

    # Розмір латентного простору
    latent_dim = 100

    # Створення генератора і дискримінатора
    generator = build_generator(latent_dim)
    discriminator = build_discriminator()

    # Компіляція дискримінатора
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Побудова GAN, з'єднання генератора і дискримінатора
    discriminator.trainable = False  # Фіксація ваг дискримінатора під час навчання GAN
    model = Sequential([generator, discriminator])
    model.compile(optimizer='adam', loss='binary_crossentropy')

    # Навчання GAN - зразок
    def train_gan(model, generator, discriminator, X, epochs=10000, batch_size=32):
        for epoch in range(epochs):
            # Генерація випадкових точок у латентному просторі
            random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
            
            # Генерація фейкових даних
            generated_data = generator.predict(random_latent_vectors)
            
            # Об'єднання з реальними даними
            real_data = X[np.random.randint(0, X.shape[0], size=batch_size)]
            
            # Створення міток
            combined_data = np.concatenate([real_data, generated_data])
            labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
            
            # Додавання шуму до міток
            labels += 0.05 * np.random.random(labels.shape)
            
            # Навчання дискримінатора
            d_loss = discriminator.train_on_batch(combined_data, labels)
            
            # Генерація міток для обману дискримінатора
            misleading_labels = np.ones((batch_size, 1))
            
            # Навчання генератора через GAN
            a_loss = model.train_on_batch(random_latent_vectors, misleading_labels)
            
            # Виведення інформації про втрати через кожні 1000 епох
            if epoch % 1000 == 0:
                print(f"Epoch: {epoch}, D Loss: {d_loss[0]}, A Loss: {a_loss}")

    # Виклик навчання GAN
    train_gan(model, generator, discriminator, X)

    # Dự đoán giá tiếp theo
    next_time_index = np.array([[len(df)]])  # Наступний індекс часу
    predicted_price = model.predict(next_time_index)[0]  # Прогнозування ціни

    # Визначення коливального діапазону навколо прогнозованої ціни
    fluctuation_range = 0.001 * predicted_price  # Визначаємо діапазон коливання як 0.1% від прогнозованої ціни
    min_price = predicted_price - fluctuation_range
    max_price = predicted_price + fluctuation_range

    # Вибір випадкового значення в межах коливального діапазону
    price_predict = random.uniform(min_price, max_price)
    forecast_price[token] = price_predict

    # Виведення інформації про прогнозовану ціну та діапазон
    print(f"Predicted_price: {predicted_price}, Min_price: {min_price}, Max_price: {max_price}")
    print(f"Forecasted price for {token}: {forecast_price[token]}")


def update_data():
    tokens = ["ETH", "BTC", "SOL"]
    for token in tokens:
        download_data(token)
        format_data(token)
        train_model(token)

if __name__ == "__main__":
    update_data()
