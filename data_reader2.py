import pandas as pd
import numpy as np
import time
import datetime


class historic_data:
    def __init__(self, start, end):
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
        self.df = pd.read_csv('BTC-Daily.csv', dtype=dtype)
        self.df = self.df.drop(columns=['Volume BTC', 'Volume USD', 'symbol', 'date'])
        self.df = self.df[(self.df['unix'] >= start) & (self.df['unix'] <= end)]
        self.df = self.df.set_index('unix')
        # didn't originally work because it was finding ascending. Therefore need to sort
        self.df = self.df.sort_index()

    # \/\/\/ private methods \/\/\/

    def pad(self, p, n):
        padding = -np.flip(p[1:n])
        return np.append(padding, p)

    def SMA(self, n):
        return np.ones(n) / n

    def EMA(self, n, alpha):
        return np.array([alpha * (1 - alpha) ** i for i in range(n)])

    def LMA(self, n):
        return np.array((2 / (n + 1)) * (1 - np.arange(n) / n))

    def WMA(self, p, n, kernel):
        return np.convolve(self.pad(p, n), kernel, mode='valid')

    # \/\/\/ public methods \/\/\/
    # inut: n = number of periods in minutes , kernel = specific wma filter ie SMA, LMA, EMA , time = timestamp in unix time
    def current_WMA(self, n, kernel, time):
        current_df = self.df.loc[time - n * 86400:time]
        p = current_df['close'].values
        if len(p) < n:
            raise ValueError("Not enough data points to calculate WMA.")
        else:
            return self.WMA(p, n, kernel)[-1]

    def current_price(self, time):
        return self.df.loc[time]['close']

    def get_start(self):
        return self.start

    def get_end(self):
        return self.end


class balance():
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


class train():
    def __init__(self, train_start, train_end, test_start, test_end, step_size=86400):
        self.train_start = self.to_unix(train_start)
        self.train_end = self.to_unix(train_end)
        self.test_start = self.to_unix(test_start)
        self.test_end = self.to_unix(test_end)
        self.train_data = historic_data(self.train_start, self.train_end)
        self.test_data = historic_data(self.test_start, self.test_end)
        self.models = {}
        self.step_size = step_size

    # this excel records it as +8 which is different from original excel which was +0.
    def to_unix(self, date_str):
        dt = datetime.datetime.strptime(date_str, "%d/%m/%Y")
        dt = dt + datetime.timedelta(hours=8)
        return int(time.mktime(dt.timetuple()))

    def score(self, weights, days, alphas, start, end, my_data):
        days = [days[0], days[1], days[2], days[3], days[4], days[5]]
        for i in range(len(days)):
            days[i] = abs(int(round(days[i])))
            if days[i] < 1:
                days[i] = 1  ## possibly change this to raising an error instead of changing to 1
        for i in range(len(alphas)):
            if alphas[i] < 0:
                alphas[i] = 0
            if alphas[i] > 1:
                alphas[i] = 1

        current_time = start + (max(days) * 86400)
        my_balance = balance()
        data = my_data
        current_signal = -1
        while current_time < start + (max(days) * 86400):
            current_time += self.step_size
        current_time = start + (max(days) * 86400)

        while current_time <= end:
            high = (weights[0] * data.current_WMA(days[0], data.SMA(days[0]), current_time) +
                    weights[1] * data.current_WMA(days[1], data.LMA(days[1]), current_time) +
                    weights[2] * data.current_WMA(days[2], data.EMA(days[2], alphas[0]), current_time)) / sum(
                weights[:3])
            low = (weights[3] * data.current_WMA(days[3], data.SMA(days[3]), current_time) +
                   weights[4] * data.current_WMA(days[4], data.LMA(days[4]), current_time) +
                   weights[5] * data.current_WMA(days[5], data.EMA(days[5], alphas[1]), current_time)) / sum(
                weights[3:])
            last_signal = current_signal
            if high < low:
                current_signal = 1
                if last_signal == -1:
                    my_balance.buy(data.current_price(current_time))
            elif high > low:
                current_signal = -1
                if last_signal == 1:
                    my_balance.sell(data.current_price(current_time))
            else:
                current_signal = last_signal
            current_time += self.step_size
        if current_signal == 1:
            my_balance.sell(data.current_price(current_time - self.step_size))
        return my_balance.get_my_balance()

    # returns score if we were to just buy at the start and sell at the end
    def baseline_score(self):
        start_price = self.test_data.current_price(self.test_start)
        end_price = self.test_data.current_price(self.test_end)
        my_balance = balance()
        my_balance.buy(start_price)
        my_balance.sell(end_price)
        return my_balance.get_my_balance()

    # adds the returned value of models to the models dict
    # the value returned should be in the form [weight1, ..., weightsn, day1, ..., dayn, alpha1, ..., alphan]
    def train_model(self, model, weights, days, alphas, max_iter=1000, num_pop=10, constant=1, ):
        self.models[model] = model(self.score, weights, days, alphas, num_pop, max_iter, self.step_size,
                                   self.train_start, self.train_end, self.train_data, constant=constant)

    def test_model(self, model):
        result = self.score(self.models[model][0:6], self.models[model][6:12], self.models[model][12:14],
                            self.test_start, self.test_end, self.test_data)
        return result

    # compares all the models in the models dict
    # prints the score of each model and the best model
    def compare_models(self):
        baseline = self.baseline_score()
        results = {}
        for model in self.models:
            results[model] = self.test_model(model)
            print(f"Baseline score: {baseline}/n")
            print(f"Model: {model}, Score: {results[model]}")
            print(f"days: {self.models[model][6:12]}") # fixed the order so it matches what is put into train_models
            print(f"weights: {self.models[model][0:6]}")
            print(f"alphas: {self.models[model][12:14]}/n")
        print(f"/n")
        print(f"Best model: {max(results, key=results.get)}, Score: {max(results.values())}")
        print(f"Model made {max(results.values()) - baseline} profit over baseline")




