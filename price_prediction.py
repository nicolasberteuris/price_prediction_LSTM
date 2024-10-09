import aiohttp
import asyncio
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import datetime

# Función asíncrona para obtener los datos históricos de Binance con manejo de paginación
async def get_binance_data(symbol="BTCUSDT", interval="1d", start_date="1 Jan 2020", end_date="now"):
    base_url = "https://api.binance.com/api/v3/klines"
    start_time = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_time = int(pd.Timestamp(end_date).timestamp() * 1000)
    all_data = []
    limit = 1000

    async with aiohttp.ClientSession() as session:
        while start_time < end_time:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_time,
                "endTime": end_time,
                "limit": limit
            }

            async with session.get(base_url, params=params) as response:
                data = await response.json()

                if not data:
                    break

                all_data += data
                start_time = int(data[-1][6]) + 1  # Actualiza la fecha de inicio para la siguiente solicitud
                await asyncio.sleep(1)  # Pausa de 1 segundo para evitar ser bloqueado por la API

    df = pd.DataFrame(all_data, columns=[
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
        'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
        'Taker buy quote asset volume', 'Ignore'
    ])
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df.set_index('Open time', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close']].astype(float)
    return df

# Función para agregar indicadores técnicos
def add_technical_indicators(df):
    # Media móvil de 14 días
    df['SMA_14'] = df['Close'].rolling(window=14).mean()
    # Índice de Fuerza Relativa (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df.dropna()

# Preprocesamiento de datos
def preprocess_data(df):
    df = add_technical_indicators(df)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close', 'SMA_14', 'RSI']])
    return scaled_data, scaler

# Crear conjuntos de datos para entrenamiento
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :])  # Tomamos todas las características
        y.append(data[i + time_step, 0])  # Precio de cierre
    return np.array(X), np.array(y)

# Crear y entrenar el modelo LSTM
def create_lstm_model(X_train):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Predicción para el siguiente día
def predict_next_day(model, X_input, scaler):
    X_input = X_input.reshape(1, X_input.shape[0], X_input.shape[1])
    pred_price = model.predict(X_input)
    pred_price = scaler.inverse_transform(np.concatenate((pred_price, np.zeros((1, 2))), axis=1))  # Asegura que se invierta correctamente
    return pred_price

# Predicción para múltiples días de una sola vez
def predict_multiple_days(model, last_data, scaler, days=30):
    predictions = []
    current_input = last_data.copy()

    for _ in range(days):
        pred_price = predict_next_day(model, current_input, scaler)
        predictions.append(pred_price[0][0])

        # Actualizar current_input con la nueva predicción
        current_input = np.append(current_input[1:], np.array([[pred_price[0][0], 0, 0]]), axis=0)  # Desplazar los datos
    return predictions

# Main script asíncrono
async def main():
    # Parámetros de entrada
    symbol = input("Ingresa el ticker del símbolo (por ejemplo, BTCUSDT): ")
    start_date = "1 Jan 2017"
    end_date = datetime.datetime.now().strftime("%d %b %Y")

    # Obtener los datos de Binance de forma asíncrona con paginación
    df = await get_binance_data(symbol=symbol, start_date=start_date, end_date=end_date)

    # Imprimir la fecha más antigua y la más reciente de los datos junto con los precios
    fecha_mas_antigua = df.index.min()
    precio_mas_antiguo = df.loc[fecha_mas_antigua, 'Close']

    fecha_mas_reciente = df.index.max()
    precio_mas_reciente = df.loc[fecha_mas_reciente, 'Close']

    print(f"Fecha más antigua: {fecha_mas_antigua}, Precio: {precio_mas_antiguo}")
    print(f"Fecha más reciente: {fecha_mas_reciente}, Precio: {precio_mas_reciente}")

    # Preprocesar datos
    scaled_data, scaler = preprocess_data(df)

    # Crear conjunto de entrenamiento y prueba
    time_step = 60
    X, y = create_dataset(scaled_data, time_step)
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

    # Dividir los datos en conjuntos de entrenamiento y validación
    training_size = int(len(X) * 0.8)
    X_train, X_test = X[:training_size], X[training_size:]
    y_train, y_test = y[:training_size], y[training_size:]

    # Crear y entrenar el modelo
    model = create_lstm_model(X_train)
    
    # Configurar callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Entrenamiento del modelo
    model.fit(X_train, y_train, batch_size=64, epochs=50, validation_split=0.2, callbacks=[early_stopping])

    # Elegir método de predicción
    method = input("Elige el método de predicción (1 para día a día, 2 para predicción única): ")

    if method == '1':
        # Predecir día a día
        last_60_days = scaled_data[-time_step:]

        # Pedir al usuario cuántos días quiere predecir
        input_date = input("Ingresa la fecha hasta la que deseas predecir (YYYY-MM-DD): ")
        end_date = pd.to_datetime(input_date)
        days_to_predict = (end_date - df.index.max()).days  # Calcular cuántos días quedan hasta la fecha ingresada

        predicted_prices = []  # Inicializar la lista vacía antes de utilizarla

        next_day = df.index.max()  # Inicializar la variable next_day

        for _ in range(days_to_predict):
            pred_price = predict_next_day(model, last_60_days, scaler)
            predicted_prices.append(pred_price[0][0])
            next_day += pd.Timedelta(days=1)  # Actualizar next_day
            print(f"Predicción para la fecha: {next_day} - Precio predicho: {pred_price[0][0]:.2f} USD")
            last_60_days = np.append(last_60_days[1:], np.array([[pred_price[0][0], 0, 0]]), axis=0)  # Desplazar los datos

    elif method == '2':
        # Predecir múltiples días de una sola vez
        last_60_days = scaled_data[-time_step:]
        
        # Pedir al usuario cuántos días quiere predecir
        input_date = input("Ingresa la fecha hasta la que deseas predecir (YYYY-MM-DD): ")
        end_date = pd.to_datetime(input_date)
        days_to_predict = (end_date - df.index.max()).days  # Calcular cuántos días quedan hasta la fecha ingresada

        # Generar predicciones para múltiples días
        predicted_prices = predict_multiple_days(model, last_60_days, scaler, days=days_to_predict)

        for i, price in enumerate(predicted_prices):
            next_day = df.index.max() + pd.Timedelta(days=i + 1)  # Calcular la fecha correspondiente
            print(f"Predicción para la fecha: {next_day} - Precio predicho: {price:.2f} USD")

# Ejecutar la función principal asíncrona
asyncio.run(main())
