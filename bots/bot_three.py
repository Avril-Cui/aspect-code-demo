#This file shows some examples of Bot Type Three

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import yfinance as yf
yf.pdr_override()
import warnings
warnings.filterwarnings("ignore")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

class BotThree:
    """
        BotThree are special bots with unique algorithm/learning-based AI trading bots.
        Compared to BotOne, BotThree have more complicated strategies.
        They are more statistical and computational based.
    """

    def __init__(self) -> None:
        pass
    
    def prepare_signal1(self, price_data):
        signal = np.where(price_data.shift(-1)>price_data,1,-1)
        return signal
    
    def knn_trainer(self, price_data, k):
        price_data["diff"] = price_data["Adj Close"].diff()
        price_data = price_data.dropna()
        X = price_data[["diff"]]
        X_train = X
        Y_train = self.prepare_signal1(price_data["Adj Close"])
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, Y_train)
        return knn

    def knn_evaluator(self, knn, current_price, last_price):
        x_data = pd.DataFrame([current_price-last_price], columns=["diff"])
        X = x_data[["diff"]]
        predicted_signal = knn.predict(X)

        coefficient = predicted_signal[-1]
        return coefficient
    
    def svc_trainer(self, price_data):
        price_data["diff"] = price_data["Adj Close"].diff()
        price_data = price_data.dropna()
        X = price_data[["diff"]]
        X_train = X[:-1]
        model = SVC()
        Y_train = self.prepare_signal1(price_data["Adj Close"][:-1])
        model.fit(X_train, Y_train)
        return model

    def svc_evaluator(self, model, current_price, last_price):
        x_data = pd.DataFrame([current_price-last_price], columns=["diff"])
        X = x_data[["diff"]]
        predicted_signal = model.predict(X)
        coefficient = predicted_signal[-1]
        return coefficient
    
    def arima_forecaster(self, price_data, current_price):
        y = price_data.values
        model = ARIMA(y, order=(6, 1, 0)).fit()
        forecast = model.forecast(step=1)[0]
        if forecast > current_price:
            return 1, forecast
        elif forecast < current_price:
            return -1, forecast
        else:
            return 0, 0
        # return forecast
    
    def arima_evaluator(self, price_data, current_price):
        y = price_data.values
        model = ARIMA(y, order=(6, 1, 0)).fit()
        forecast = model.forecast(step=1)[0]
        # print(forecast)
        if forecast > current_price:
            return 1
        elif forecast < current_price:
            return -1
        else:
            return 0