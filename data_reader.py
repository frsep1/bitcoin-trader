import pandas as pd
import numpy as np
import time
import datetime
from copy import deepcopy
import NatureBasedAlgorithm
from equations import MACD, original


class HistoricData:
    def __init__(self, start, end, step_size, buffer_days=0):
        dtype = {
            'unix': np.int64,
            'open': np.float64,
            'high': np.float64,
            'low': np.float64,
            'close': np.float64,
            'Volume BTC': np.float64,
            'Volume USD': np.float64,
            'date': 'str',
            'symbol': 'str'
        }

        self.step_size = step_size
        # doing minute data requires too many changes so will not allow it to be done
        if self.step_size < 60:
            raise ValueError("Step Size must be above 60")
        # allow a range for testing and faster/slower run time
        elif self.step_size < 86400:
            print("hourly")
            self.df = pd.read_csv('BTC-Hourly.csv', dtype=dtype)
        # going to default to daily data
        else:
            print("daily")
            self.df = pd.read_csv('BTC-Daily.csv', dtype=dtype)

        self.df = self.df.drop(columns=['Volume BTC', 'Volume USD', 'symbol', 'date'])

        # Include a buffer period to allow back-calculating WMAs
        buffer_start = start - buffer_days * self.step_size
        self.df = self.df[(self.df['unix'] >= buffer_start) & (self.df['unix'] <= end)]
        self.df = self.df.set_index('unix')
        self.df = self.df.sort_index()

    def pad(self, p, n):
        padding = -np.flip(p[1:n])
        return np.append(padding, p)

    def SMA(self, n):
        return np.ones(n) / n

    def EMA(self, n, alpha):
        return np.flip(np.array([alpha * (1 - alpha) ** i for i in range(n)]))

    def LMA(self, n):
        return np.array((2 / (n + 1)) * (1 - np.arange(n) / n))

    def WMA(self, p, n, kernel):
        return np.convolve(self.pad(p, n), kernel, mode='valid')

    def current_price(self, time):
        return self.df.loc[time]['close']

    def get_start(self):
        return self.start

    def get_end(self):
        return self.end


class Balance():
    def __init__(self, balance=1000):
        self.my_balance = balance
        self.bitcoin = 0

    def get_my_balance(self):
        return self.my_balance

    def buy(self, price):
        bitcoins = (self.my_balance / price) * 0.97
        self.bitcoin += bitcoins
        self.my_balance = 0

    def sell(self, price):
        self.my_balance += (self.bitcoin * price) * 0.97
        self.bitcoin = 0


class Train():
    def __init__(self, train_start, train_end, test_start, test_end, step_size=86400):
        self.train_start = self.to_unix(train_start)
        self.train_end = self.to_unix(train_end)
        self.test_start = self.to_unix(test_start)
        self.test_end = self.to_unix(test_end)
        self.step_size = step_size
        self.train_data = HistoricData(self.train_start, self.train_end,self.step_size,  buffer_days=0, )
        self.test_data = HistoricData(self.test_start, self.test_end,self.step_size, buffer_days=0, )
        self.models = {}

    # this excel records it as +8 which is different from original excel which was +0.
    def to_unix(self, date_str):
        dt = datetime.datetime.strptime(date_str, "%d/%m/%Y")
        dt = dt + datetime.timedelta(hours=8)
        return int(time.mktime(dt.timetuple()))

    def score(self, data, equation, weights=[], days=[], alphas=[]):
        # Making days integers and positive
        days = [min(30, max(1, abs(int(round(d))))) for d in days]
        # Making alphas floats and between 0 and 1
        alphas = [min(1, max(0, a)) for a in alphas]

        my_balance = Balance()
        last_signal = "sell"
        highs, lows = equation({'weights':weights, 'days':days, 'alphas':alphas}, data)

        for i, t in enumerate(data.df.index):      

            if highs[i] > lows[i]:
                if last_signal == "sell":
                    my_balance.buy(data.current_price(t))
                    last_signal = "buy"
            elif highs[i] < lows[i]:
                if last_signal == "buy":
                    my_balance.sell(data.current_price(t))
                    last_signal = "sell"

        if last_signal == "buy":
            my_balance.sell(data.current_price(data.df.index[-1]))
        return my_balance.get_my_balance()

    # returns score if we were to just buy at the start and sell at the end
    def baseline_score(self):
        start_price = self.test_data.current_price(self.test_start)
        end_price = self.test_data.current_price(self.test_end)
        # print("Start and end price is:" + str(start_price), str(end_price), self.test_start, self.test_end)
        bal = Balance()
        bal.buy(start_price)
        bal.sell(end_price)
        return bal.get_balance()

    # adds the returned value of models to the models dict
    # the value returned should be in the form [weight1, ..., weightsn, day1, ..., dayn, alpha1, ..., alphan]
    def train_model(self, model: NatureBasedAlgorithm, num_agents, num_iterations):
        model.best_pos = model.optimise(num_agents, num_iterations, constant=1)
        if len(self.models[model]) == 14:
            model.best_score = self.score(self.test_data,
                                          original,
                                          weights = model.best_pos[0:6],
                                          days = model.best_pos[6:12],
                                          alphas = model.best_pos[12:14])
        else:
            model.best_score = self.score(
                self.train_data,
                MACD,
                days = self.models[model][:3],
                alphas = self.models[model][3:])
        print(f"Training model: {model.name}, Training Score: {model.best_score}")
        self.models[model.name] = model

        return model.best_score

    def test_model(self, model):
        if len(self.models[model]) == 14:
            return self.score(
                self.test_data,
                original,
                weights = self.models[model][:6],
                days = self.models[model][6:12],
                alphas = self.models[model][12:14]
            )
        else:
            return self.score(
                self.test_data,
                MACD,
                days = self.models[model][:3],
                alphas = self.models[model][3:]
            )

    # compares all the models in the models dict
    # prints the score of each model and the best model
    def compare_models(self):
        baseline = self.baseline_score()
        results = {}
        for model_used in self.models:
            train_model: NatureBasedAlgorithm  = self.models[model_used]
            test_model: NatureBasedAlgorithm = deepcopy(train_model)
            print(f"-------------------------------------------------------------------")
            results[model_used] = self.test_model(test_model)
            print(f"Model: {model_used}")
            print(f"Score: ${results[model_used]:.2f}")
            if len(test_model.best_pos) == 14:
                print("Equation: Original")
                print(f"Weights: {np.array2string(test_model.best_pos[0:6], precision=4, floatmode='fixed')}")
            else:
                print("Equation: MACD")
            print(f"Days: {np.array2string(test_model.best_pos[6:12], precision=0, floatmode='fixed')}") # fixed the order so it matches what is put into train_models
            print(f"Alphas: {np.array2string(test_model.best_pos[12:14], precision=4, floatmode='fixed')}\n")

        print(f"{"BEST MODEL":-14}")
        print(f"Baseline score: {baseline:.2f}\n")
        print(f"Best model: {max(results, key=results.get)}, Score: {max(results.values())}")
        print(f"Model made {max(results.values()) - baseline} profit over baseline")