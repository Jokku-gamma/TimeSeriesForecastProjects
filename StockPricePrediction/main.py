import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime,timedelta

ticker='AAPL'
try:
    data=yf.download(ticker,start="2020-01-01",end="2025-05-26",auto_adjust=False,progress=False)
    if data.empty:
        raise ValueError("No data found for the ticker symbol.")
except Exception as e:
    print(f"Error fetching data for {ticker}: {e}")
    exit()

prices=data[['Adj Close','Volume']].copy()
prices['Adj Close']=np.log(prices['Adj Close'])
train=prices[:-60]
test=prices[-60:]

print("Training data shape:", train.shape)
print("Test data shape:", test.shape)
print(prices.head())

def check_stats(series):
    results=adfuller(series)
    p_v=results[1]
    if p_v<0.05:
        print("Series is stationary (p-value < 0.05)")
    else:
        print("Series is non-stationary (p-value >= 0.05)")
    return p_v<0.05

if not check_stats(train['Adj Close']):
    print("Using auto-arima to find best parameters")
try:
    arima_model=auto_arima(train['Adj Close'],seasonal=True,m=5,trace=True,suppress_warnings=True)
    arima_forecast=arima_model.predict(n_periods=60)
    arima_forecast=np.exp(arima_forecast) 
except Exception as e:
    print(f"Error in ARIMA model: {e}")
    arima_forecast = np.full(60,np.nan)

try:
    es_model=ExponentialSmoothing(train['Adj Close'],trend='add',seasonal='add',seasonal_periods=5).fit()
    es_forecast=es_model.forecast(steps=60)
    es_forecast=np.exp(es_forecast)
except Exception as e:
    print(f"Error in Exponential Smoothing model: {e}")
    es_forecast = np.full(60,np.nan)


scaler_price=MinMaxScaler()
scaler_vol=MinMaxScaler()
train_scaled=train.copy()
train_scaled_price=scaler_price.fit_transform(train['Adj Close'].values.reshape(-1,1))
train_scaled_vol=scaler_vol.fit_transform(train['Volume'].values.reshape(-1,1))
test_scaled_price=scaler_price.transform(test['Adj Close'].values.reshape(-1,1))
test_scaled_vol=scaler_vol.transform(test['Volume'].values.reshape(-1,1))

train_scaled=np.hstack((train_scaled_price,train_scaled_vol))
test_scaled=np.hstack((test_scaled_price,test_scaled_vol))

def create_seqs(data,seq_len):
    X,Y=[],[]
    for i in range(len(data)-seq_len):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len,0])
    return np.array(X),np.array(Y)

seq_len=60
x_train,y_train=create_seqs(train_scaled,seq_len)
x_test,y_test=create_seqs(test_scaled,seq_len)

lstm_model=Sequential(
    [
        LSTM(100,activation='relu',input_shape=(seq_len,2),return_sequences=True),
        Dropout(0.2),
        LSTM(100,activation='relu',return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ]
)

lstm_model.compile(optimizer='adam',loss='mse')
early_stopping=EarlyStopping(monitor='loss',patience=5,restore_best_weights=True)
lstm_model.fit(x_train,y_train,epochs=50,batch_size=32,verbose=1,callbacks=[early_stopping])
if len(x_test)>0:
    x_test_resh=x_test.reshape(x_test.shape[0],x_test.shape[1],2)
    lstm_preds=lstm_model.predict(x_test_resh,verbose=0)
    lstm_preds=scaler_price.inverse_transform(lstm_preds).flatten()
    lstm_preds=np.exp(lstm_preds)
    y_test_unscaled=np.exp(scaler_price.inverse_transform(y_test.reshape(-1,1)).flatten())
else:
    print("Not enough data")
    last_seq=train_scaled[-seq_len:]
    lstm_preds=[]
    for _ in range(60):
        last_seq_resh=last_seq.reshape(1,seq_len,2)
        next_pred=lstm_model.predict(last_seq_resh,verbose=0)
        lstm_preds.append(next_pred[0,0])
        last_seq=np.roll(last_seq,-1)
        last_seq[-1,0]=next_pred
        last_seq[-1,1]=last_seq[-2,1]
    lstm_preds=scaler_price.inverse_transform(np.array(lstm_preds).reshape(-1,1)).flatten()
    lstm_preds=np.exp(lstm_preds)
    y_test_unscaled=np.exp(test['Adj Close'].values)

print("ARIMA forecast",len(arima_forecast))
print(f"Exp. Smoothign forecast length : {len(es_forecast)}")
print(f"LSTM forecast length : {len(lstm_preds)}")
print(f"Any Nan in test data : {test.isna().any().any()}")
print(f"Any Nan in ARIMA forecast :{np.any(np.isnan(arima_forecast))}")
print(f"Any Nan in Exponential Smoothing forecast :{np.any(np.isnan(es_forecast))}")
print(f"Any Nan in LSTM forecast :{np.any(np.isnan(lstm_preds))}")

def print_mats(actual, pred, model_name):
    if not np.any(np.isnan(pred)):
        actual_subset = actual[-len(pred):]
        mae = mean_absolute_error(actual_subset, pred)
        rmse = np.sqrt(mean_squared_error(actual_subset, pred))
        print(f"{model_name} MAE: {mae:.2f}")
        print(f"{model_name} RMSE: {rmse:.2f}")
    else:
        print(f"{model_name} predictions contain NaN values, skipping metrics calculation.")

print_mats(test['Adj Close'].values, arima_forecast, 'ARIMA')
print_mats(test['Adj Close'].values, es_forecast, 'Exponential Smoothing')
print_mats(test['Adj Close'].values[-len(lstm_preds):], lstm_preds, 'LSTM')

# Plot results with proper date formatting
plt.figure(figsize=(12, 6))
plt.plot(test.index, test['Adj Close'], label='Actual Prices', color='black')
plt.plot(test.index, arima_forecast, label='ARIMA Forecast', color='blue')
plt.plot(test.index, es_forecast, label='Exponential Smoothing Forecast', color='green')
if len(x_test) > 0:
    plt.plot(test.index[-len(lstm_preds):], lstm_preds, label='LSTM Predictions (Test Set)', color='red')
else:
    plt.plot(test.index, lstm_preds, label='LSTM Predictions (Iterative)', color='red')

plt.title('AAPL Stock Price Prediction (Adj Close)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gcf().autofmt_xdate()
plt.savefig('stock_price_prediction_fixed.png')
plt.show()

def extend_fore(model,current_forecast,steps_needed):
    if isinstance(model,auto_arima.ARIMA):
        extended=model.predict(n_periods=len(current_forecast)+steps_needed)
        return np.exp(extended[-steps_needed:])
    elif isinstance(model,ExponentialSmoothing):
        extended=model.forecast(steps=len(current_forecast)+steps_needed)
        return np.exp(extended[-steps_needed:])
    else:
        raise ValueError("Unsupported model type for extension")

def pred_for_date(target_date,test_data,arima_model,es_model,lstm_model,scaler_price,scaler_volume,seq_len,train_scaled,test_scaled):
    try:
        target_date=pd.to_datetime(target_date)
    except ValueError:
        print("Invalid date format")
        return
    test_start=test_data.index[0]
    test_end=test_data.index[-1]
    if target_date<test_start:
        print(f"Date {target_date} is before test period")
        return
    elif target_date<=test_end:
        if target_date in test_data.index:
            act_price=test_data['Adj Close'][target_date]
        else:
            print(f"Date is not a trading day")
            return
        idx=np.where(test_data.index==target_date)[0][0]
        arima_pred=arima_forecast[idx] if not np.isnan(arima_forecast[idx]) else None
        es_pred=es_forecast[idx] if not np.isnan(es_forecast[idx]) else None
        lstm_start=len(test_data)-len(lstm_preds)
        lstm_pred=lstm_preds[idx-lstm_start] if idx-lstm_start>=0 else None
    else:
        days_ahead=(target_date-test_end).days
        trading_days=int(days_ahead * (5/7))
        if trading_days<=0:
            trading_days=1
        print(f"Extending forecast by {trading_days}")
        arima_ext=extend_fore(arima_model,arima_forecast,trading_days)
        es_extended=extend_fore(es_model,es_forecast,trading_days)
        arima_pred=arima_ext[-1] if not np.isnan(arima_ext[-1]) else None
        es_pred=es_extended[-1] if not np.isnan(es_extended[-1]) else None

        last_seq=train_scaled[-seq_len:] if len(x_test)==0 else test_scaled[-seq_len:]
        lstm_extended=[]
        for _ in range(len(test_data)+trading_days):
            last_seq_resh=last_seq.reshape(1,seq_len,2)
            next_pred=lstm_model.predict(last_seq_resh,verbose=0)
            lstm_extended.append(next_pred[0,0])
            last_seq=np.roll(last_seq,-1)
            last_seq[-1,0]=next_pred
            last_seq[-1,1]=last_seq[-2,1]
        lstm_extended=scaler_price.inverse_transform(np.array(lstm_extended).reshape(-1,1)).flatten()
        lstm_extended=np.exp(lstm_extended)
        lstm_pred=lstm_extended[-1]

        actual_price=None
    
    print(f"\nResults for {target_date_str}:")
    if actual_price is not None:
        print(f"Actual Price: ${actual_price:.2f}")
    else:
        print("Actual Price: Not available (future date)")

    print(f"ARIMA Prediction: ${arima_pred:.2f}" if arima_pred is not None else "ARIMA Prediction: Not available")
    if actual_price is not None and arima_pred is not None:
        print(f"ARIMA Error (Actual - Predicted): ${actual_price - arima_pred:.2f}")

    print(f"Exponential Smoothing Prediction: ${es_pred:.2f}" if es_pred is not None else "Exponential Smoothing Prediction: Not available")
    if actual_price is not None and es_pred is not None:
        print(f"Exponential Smoothing Error (Actual - Predicted): ${actual_price - es_pred:.2f}")

    print(f"LSTM Prediction: ${lstm_pred:.2f}" if lstm_pred is not None else "LSTM Prediction: Not available")
    if actual_price is not None and lstm_pred is not None:
        print(f"LSTM Error (Actual - Predicted): ${actual_price - lstm_pred:.2f}")


while True:
    date_input=input("Enter a date in YYYY-MM-DD format :")
    if date_input.lower() in ['exit', 'quit']:
        print("Exiting the program.")
        break
    pred_for_date(date_input, test, arima_model, es_model, lstm_model, scaler_price, scaler_vol, seq_len, train_scaled, test_scaled)