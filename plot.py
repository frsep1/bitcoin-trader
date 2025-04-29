import matplotlib.pyplot as plt
from datetime import datetime
import data_reader as dr

class Plotter:
    def __init__(self, data, start, end, params, freq_function):
        self.data = data
        self.start = start
        self.end = end
        self.params =params
        self.freq_function = freq_function

    def get_plotting_data(self):
        timestamps = self.data.df.index  # start after buffer
        self.params['days'] = [min(30, max(1, abs(int(round(d))))) for d in self.params['days']] # Potentially remove after changing to int random sample
        prices = []
        highs = []
        lows = []
        times = []

        for t in timestamps:
            price = self.data.current_price(t)
            high, low = self.freq_function(self.params, t, self.data)

            prices.append(price)
            highs.append(high)
            lows.append(low)
            # convert unix timestamp to readable date for x-axis
            times.append(datetime.fromtimestamp(t).strftime('%d-%m-%y'))
        self.prices = prices
        self.highs = highs
        self.lows = lows
        self.times = times

    def plot_signals(self):
        plt.figure(figsize=(14, 6))
        plt.plot(self.times, self.prices, label='Price (Close)', color='black', linewidth=1.5)
        plt.plot(self.times, self.highs, label='High Signal', color='green', linestyle='--')
        plt.plot(self.times, self.lows, label='Low Signal', color='red', linestyle='--')
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.title("Price, High Signal, and Low Signal Over Time")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()

def equation(params, t, data):
    high = sum([
            params['weights'][0] * data.current_WMA(params['days'][0], data.SMA(params['days'][0]), t),
            params['weights'][1] * data.current_WMA(params['days'][1], data.LMA(params['days'][1]), t),
            params['weights'][2] * data.current_WMA(params['days'][2], data.EMA(params['days'][2], params['alphas'][0]), t)
            ]) / sum(params['weights'][:3])
    low = sum([
            params['weights'][3] * data.current_WMA(params['days'][3], data.SMA(params['days'][3]), t),
            params['weights'][4] * data.current_WMA(params['days'][4], data.LMA(params['days'][4]), t),
            params['weights'][5] * data.current_WMA(params['days'][5], data.EMA(params['days'][5], params['alphas'][1]), t)
            ]) / sum(params['weights'][3:])
    return high, low

TRAIN_START = "28/11/2014"
TRAIN_END = "31/12/2019"
TEST_START = "01/01/2020"
TEST_END = "01/03/2023"

train = dr.Train(TRAIN_START, TRAIN_END, TEST_START, TEST_END, step_size=86400 )

params = {
    'days': [30, 19.69532712, 1, 1, 6.89587399, 30],
    'weights': [0.86662705, 0.81100534, 1.00215413, 0.1, 0.1, 0.1],
    'alphas': [0.99172123, 0.13849593]
}

test_plot = Plotter(train.test_data, train.test_start, train.test_end, params, equation)
test_plot.get_plotting_data()
test_plot.plot_signals()