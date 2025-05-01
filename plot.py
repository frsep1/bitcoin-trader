import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import data_reader as dr

class Plotter:
    def __init__(self, data, start, end, models, freq_function):
        self.data = data
        self.start = start
        self.end = end
        self.models = models
        self.freq_function = freq_function
        self.plot_data = {}

    def get_plotting_data(self):
        timestamps = self.data.df.index  # start after buffer
        self.times = [datetime.fromtimestamp(t) for t in timestamps]
        self.prices = [self.data.current_price(t) for t in timestamps]
        
        for model in self.models:
            highs = []
            balance = dr.Balance()
            balance_line = []
            lows = []
            buy_signals = []
            sell_signals = []
            days = [min(30, max(1, abs(int(round(d))))) for d in self.models[model][6:12]]
            current_signal = -1
            for t in timestamps:

                high, low = self.freq_function({'weights':self.models[model][:6], "days":days, "alphas":self.models[model][12:14]}, t, self.data)
                highs.append(high)
                lows.append(low)

                if high < low:
                    if current_signal == -1:
                        balance.buy(self.data.current_price(t))
                        buy_signals.append((t, self.data.current_price(t)))
                        current_signal = 1
                elif high > low:
                    if current_signal == 1:
                        balance.sell(self.data.current_price(t))
                        sell_signals.append((t, self.data.current_price(t)))
                        current_signal = -1

                if balance.get_balance() == 0:
                    balance_line.append((balance.btc * self.data.current_price(t))*0.97)
                else:
                    balance_line.append(balance.get_balance())

            self.plot_data[model] = {"highs":highs, "lows":lows, "balance_line":balance_line, "balance":balance, 'buy_signals':buy_signals, 'sell_signals':sell_signals}
 
    def plot_signals(self, model):
        model_data = self.plot_data[model]
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(self.times, self.prices, label='Price (Close)', color='black', linewidth=1.5)
        ax.plot(self.times, model_data['highs'], label='High Signal', color='green', linestyle='--')
        ax.plot(self.times, model_data['lows'], label='Low Signal', color='red', linestyle='--')

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

    def plot_buy_sell(self, model):
        buy_signals = []
        sell_signals = []
        model_data = self.plot_data[model]
        signal_state = -1  # cash signal

        for i in range(1, len(model_data['lows'])):
            if model_data['highs'][i] > model_data['lows'][i] and model_data['highs'][i - 1] <= model_data['lows'][i - 1]:
                if signal_state == -1:  # buy signal
                    buy_signals.append((self.times[i], self.prices[i]))
                    signal_state = 1
            elif model_data['highs'][i] < model_data['lows'][i] and model_data['highs'][i - 1] >= model_data['lows'][i - 1]:
                if signal_state == 1:  # sell signal
                    sell_signals.append((self.times[i], self.prices[i]))
                    signal_state = -1

        fig, ax1 = plt.subplots(figsize=(14, 6))

        # Main signals
        ax1.plot(self.times, self.prices, label='P', color='black', linewidth=1.5)
        ax1.plot(self.times, model_data['highs'], label='High', color='orange')
        ax1.plot(self.times, model_data['lows'], label='Low', color='deepskyblue')
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
        diff = [h - l for h, l in zip(model_data['highs'], model_data['lows'])]
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
        baseline = []
        for i in range(len(self.times)):
            baseline.append((self.prices[i]/self.prices[0])*1000*0.97*0.97)

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(self.times, baseline, label=f'Baseline {round(baseline[-1],2)}', color='black', linewidth=1.5)
        for model in self.models:
            ax.plot(self.times, self.plot_data[model]['balance_line'], label=f'{model} {round(self.plot_data[model]['balance_line'][-1],2)}', linewidth=1.5)
        ax.set_xlabel("Date")
        ax.set_ylabel("Balance (USD)")
        ax.set_title("Balances Over Time")

        # Show only 10 x-axis ticks
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