import pandas as pd
import numpy as np
import time
import datetime


class HistoricData:
    def __init__(self, start, end, buffer_days=0):
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

        # Include a buffer period to allow back-calculating WMAs
        buffer_start = start - buffer_days * 86400

        self.df = self.df[(self.df['unix'] >= buffer_start) & (self.df['unix'] <= end)]


        self.df = self.df.set_index('unix')

        # doesn't work without this because of the excel order
        self.df = self.df.sort_index()


    def SMA(self, n):
        return np.ones(n) / n

    def EMA(self, n, alpha):
        return np.array([alpha * (1 - alpha) ** i for i in range(n)])

    def LMA(self, n):
        return np.array((2 / (n + 1)) * (1 - np.arange(n) / n))

    # This modifies all points
    def pad(self, p, n):
        padding = -np.flip(p[1:n])
        #print([n,len(p),len(padding)])
        return np.append(padding, p)

    def WMA(self, p, n, kernel):
        return np.convolve(p, kernel, mode='valid')

    # Calculates the WMA and checks to see if padding is required
    def current_WMA(self, n, kernel, timestamp):
        # Gets data points from time t - n_days to time t
        p = self.df.loc[timestamp - n * 86400:timestamp]['close'].values
        # Check if we have enough data points; pad if necessary
        if len(p) < n:
            # Gets t to t + n_days datapoints
            p = self.df.loc[timestamp:timestamp + n * 86400]['close'].values
            # Creates new padded data points
            p = self.pad(p, n)

        return self.WMA(p, n, kernel)[-1]

    def current_price(self, timestamp):
        return self.df.loc[timestamp]['close']


class Balance:
    def __init__(self, initial_balance=1000):
        self.fiat = initial_balance
        self.btc = 0

    def buy(self, price):
        self.btc += (self.fiat / price) * 0.97
        self.fiat = 0

    def sell(self, price):
        self.fiat += (self.btc * price) * 0.97
        self.btc = 0

    def get_balance(self):
        return self.fiat


class Train:
    def __init__(self, train_start, train_end, test_start, test_end, step_size=86400):
        self.train_start = self.to_unix(train_start)
        self.train_end = self.to_unix(train_end)
        self.test_start = self.to_unix(test_start)
        self.test_end = self.to_unix(test_end)
        self.train_data = HistoricData(self.train_start, self.train_end, buffer_days=0)
        self.test_data = HistoricData(self.test_start, self.test_end, buffer_days=0)
        self.models = {}
        self.step_size = step_size

    def to_unix(self, date_str):

        dt = datetime.datetime.strptime(date_str, "%d/%m/%Y")
        dt = dt + datetime.timedelta(hours=8)
        return int(time.mktime(dt.timetuple()))

    def score(self, weights, days, alphas, start, end, data):
        days = [min(30, max(1, abs(int(round(d))))) for d in days]
        alphas = [min(1, max(0, a)) for a in alphas]
        max_days_needed = max(days)
        if end - start < max_days_needed * 86400:
            raise ValueError("Training/test window is too short for chosen WMA window sizes.")
        current_time = start # + 30 * 86400
        balance = Balance()
        current_signal = -1
        day_counter = 0

        while current_time <= end:
            high = sum([
                weights[0] * data.current_WMA(days[0], data.SMA(days[0]), current_time),
                weights[1] * data.current_WMA(days[1], data.LMA(days[1]), current_time),
                weights[2] * data.current_WMA(days[2], data.EMA(days[2], alphas[0]), current_time)
            ]) / sum(weights[:3])

            low = sum([
                weights[3] * data.current_WMA(days[3], data.SMA(days[3]), current_time),
                weights[4] * data.current_WMA(days[4], data.LMA(days[4]), current_time),
                weights[5] * data.current_WMA(days[5], data.EMA(days[5], alphas[1]), current_time)
            ]) / sum(weights[3:])

            if high < low:
                if current_signal == -1:
                    balance.buy(data.current_price(current_time))
                    current_signal = 1
            elif high > low:
                if current_signal == 1:
                    balance.sell(data.current_price(current_time))
                    current_signal = -1
            current_time += self.step_size
            day_counter += 1

        if current_signal == 1:
            balance.sell(data.current_price(current_time - self.step_size))

        #print(f"[DEBUG - score] Number of training days simulated: {day_counter}")
        return balance.get_balance()

    def baseline_score(self):
        start_price = self.test_data.current_price(self.test_start)
        end_price = self.test_data.current_price(self.test_end)
        bal = Balance()
        bal.buy(start_price)
        bal.sell(end_price)
        return bal.get_balance()

    def train_model(self, model, days, weights, alphas, max_iter=1000, num_pop=10, constant=1):
        best, error = model(
            self.score, days, weights, alphas, num_pop, max_iter,
            self.step_size, self.train_start, self.train_end,
            self.train_data, constant=constant
        )
        self.models[model] = best
        return error

    def test_model(self, model):
        result = self.score(
            self.models[model][:6],
            self.models[model][6:12],
            self.models[model][12:14],
            self.test_start, self.test_end, self.test_data
        )
        return result

    def compare_models(self):
        baseline = self.baseline_score()
        print(f"Baseline score: {baseline}\n")
        results = {}
        for model in self.models:
            results[model] = self.test_model(model)
            print(f"Model: {model}, Score: {results[model]}")
            print(f"Weights: {self.models[model][:6]}")
            print(f"Days: {self.models[model][6:12]}")
            print(f"Alphas: {self.models[model][12:14]}\n")
            print([results[model], self.models[model][12:14]])
        best_model = max(results, key=results.get)
        print(f"Best model: {best_model}, Score: {results[best_model]}")
        print(f"Profit over baseline: {results[best_model] - baseline}")
