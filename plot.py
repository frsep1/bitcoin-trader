import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import data_reader as dr

class Plotter:
    def __init__(self, data, start, end, params, freq_function):
        self.data = data
        self.start = start
        self.end = end
        self.params = params
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
            times.append(datetime.fromtimestamp(t))
        self.prices = prices
        self.highs = highs
        self.lows = lows
        self.times = times

    def plot_signals(self):
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(self.times, self.prices, label='Price (Close)', color='black', linewidth=1.5)
        ax.plot(self.times, self.highs, label='High Signal', color='green', linestyle='--')
        ax.plot(self.times, self.lows, label='Low Signal', color='red', linestyle='--')

        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.set_title("Price, High Signal, and Low Signal Over Time")

        # Show only 10 x-axis ticks (or fewer if the dataset is small)
        step = max(1, len(self.times) // 10)
        ax.set_xticks(self.times[::step])
        ax.set_xticklabels(
            [dt.strftime("%d-%m-%y") for dt in self.times[::step]],
            rotation=45
        )

        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        plt.show()

    def plot_buy_sell(self):
        buy_signals = []
        sell_signals = []
        signal_state = -1  # assume start in cash

        for i in range(1, len(self.highs)):
            if self.highs[i] > self.lows[i] and self.highs[i - 1] <= self.lows[i - 1]:
                if signal_state == -1:  # buy signal
                    buy_signals.append((self.times[i], self.prices[i]))
                    signal_state = 1
            elif self.highs[i] < self.lows[i] and self.highs[i - 1] >= self.lows[i - 1]:
                if signal_state == 1:  # sell signal
                    sell_signals.append((self.times[i], self.prices[i]))
                    signal_state = -1

        fig, ax1 = plt.subplots(figsize=(14, 6))

        # Main signals
        ax1.plot(self.times, self.prices, label='P', color='black', linewidth=1.5)
        ax1.plot(self.times, self.highs, label='High', color='orange')
        ax1.plot(self.times, self.lows, label='Low', color='deepskyblue')
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price", color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.set_title("Buy/Sell Signal Plot")

        # Markers for buy/sell
        for t, p in buy_signals:
            ax1.scatter(t, p, marker='^', color='green', label='buy' if t == buy_signals[0][0] else "")
        for t, p in sell_signals:
            ax1.scatter(t, p, marker='v', color='red', label='sell' if t == sell_signals[0][0] else "")

        # Secondary y-axis for difference signal
        ax2 = ax1.twinx()
        diff = [h - l for h, l in zip(self.highs, self.lows)]
        ax2.plot(self.times, diff, label='High-Low', color='gray', linestyle='dotted')
        ax2.set_ylabel("Signal Difference", color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')

        # Combined legend
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2)

        fig.tight_layout()
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()

    def plot_profit(self):
        baseline = {'Balance_line':[]}
        optimiser = {"Balance":dr.Balance(), 'Balance_line':[]}
        current_signal = -1

        for i in range(len(self.times)):
            
            if self.highs[i] < self.lows[i]:
                if current_signal == -1:
                    optimiser['Balance'].buy(self.prices[i])
                    current_signal = 1
            elif self.highs[i] > self.lows[i]:
                if current_signal == 1:
                    optimiser['Balance'].sell(self.prices[i])
                    current_signal = -1
            
            baseline['Balance_line'].append((self.prices[i]/self.prices[0])*1000*0.97*0.97)
            if optimiser['Balance'].get_balance() == 0:
                optimiser['Balance_line'].append((optimiser['Balance'].btc * self.prices[i])*0.97)
            else:
                optimiser['Balance_line'].append(optimiser['Balance'].get_balance())

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(self.times, baseline['Balance_line'], label=f'Baseline {round(baseline['Balance_line'][-1],2)}', color='black', linewidth=1.5)
        ax.plot(self.times, optimiser['Balance_line'], label=f'Optimiser {round(optimiser['Balance_line'][-1],2)}', color='green', linewidth=1.5)

        ax.set_xlabel("Date")
        ax.set_ylabel("Balance (USD)")
        ax.set_title("Balances Over Time")

        # Show only 10 x-axis ticks (or fewer if the dataset is small)
        step = max(1, len(self.times) // 10)
        ax.set_xticks(self.times[::step])
        ax.set_xticklabels(
            [dt.strftime("%d-%m-%y") for dt in self.times[::step]],
            rotation=45
        )

        ax.legend()
        ax.grid(True)
        fig.tight_layout()
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
TEST_END = "01/03/2022"

train = dr.Train(TRAIN_START, TRAIN_END, TEST_START, TEST_END, step_size=86400)

params = {
    'weights': [ 0.1,         0.1,         0.1,         0.1,        54.55074943,  0.1,       ],
    'days': [16.43768618,  1.,         16.43768618, 12.59468119, 16.43768618, 16.43768618],
    'alphas': [0.1, 0.1]
}
#print(train.test_data.df.index)
test_plot = Plotter(train.test_data, train.test_start, train.test_end, params, equation)
test_plot.get_plotting_data()
test_plot.plot_profit()

# ------------------ MRFO ----------------------------
# max_iter=100, num_pop=10, constant=1
# [1, 100, 6], [0.01, 5, 6], [0.01, 1, 2]

# Baseline score: 5660.26318654683

# Model: <function manta_ray_algo at 0x000001774477BB00>, Score: 5315.578046862433
# Weights: [ 0.1         0.1         0.1         0.1        54.55074943  0.1       ]
# Days: [16.43768618  1.         16.43768618 12.59468119 16.43768618 16.43768618]
# Alphas: [0.1 0.1]

# [5315.578046862433, array([0.1, 0.1])]
# Best model: <function manta_ray_algo at 0x000001774477BB00>, Score: 5315.578046862433
# Profit over baseline: -344.68513968439674

# ------------------- Plot Results ----------------------
# Baseline profit = 5660.26
# Bot profit = 5315.58