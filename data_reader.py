import pandas as pd
import numpy as np
import time
import datetime
start = "01/01/2023"
end = "30/12/2024"

class historic_data:
    def __init__(self, start=start, end=end):
        self.start = time.mktime(datetime.datetime.strptime(start, "%d/%m/%Y").timetuple())
        self.end = time.mktime(datetime.datetime.strptime(end, "%d/%m/%Y").timetuple())
        dtype = {
            'Timestamp': np.int64,
            'Open': np.float64,
            'High': np.float64,
            'Low': np.float64,
            'Close': np.float64,
            'Volume': np.float64,
            'datetime': 'str'
        }
        self.df = pd.read_csv('btcusd_1-min_data.csv', dtype=dtype)
        self.df = self.df.drop(columns=['Volume', 'datetime']) ## not sure what volume is but maybe we need it?
        self.df = self.df[(self.df['Timestamp'] >= self.start) & (self.df['Timestamp'] <= self.end)]
        self.df = self.df.set_index('Timestamp') # setting the timestamp as the index to speed up finding the current price from O(n) to O(1)

    #\/\/\/ private methods \/\/\/

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
    
    #\/\/\/ public methods \/\/\/
    # inut: n = number of periods in minutes , kernel = specific wma filter ie SMA, LMA, EMA , time = timestamp in unix time
    def current_WMA(self, n, kernel, time):
        current_df = self.df.loc[time - n * 60:time]
        p = current_df['Close'].values
        if len(p) < n:
            raise ValueError("Not enough data points to calculate WMA.")
        else:
            return self.WMA(p, n, kernel)[-1]
    
    def current_price(self, time):
        current_row = self.df.loc[time].iloc[-1]
        return current_row
    
    def get_start(self):
        return self.start
    
    def get_end(self):
        return self.end
    




class balance():
    def __init__(self, balance = 1000):
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
    

#scroing function designed to get the final my_balance with strategy given on handout within a certain time frame
#weights = [w1, w2, w3, w4, w5, w6]
#days = [d1, d2, d3, d4, d5, d6] <<<<<-----  is actually in seconds not days
#alphas = [a1, a2]
def scoring(weights, days, alphas, start, end, my_data=historic_data(), intervals=60):
    days = [days[0], days[1], days[2], days[3], days[4], days[5]]
    for i in range(len(days)):
        days[i] = abs(int(round(days[i])))
        if days[i] < 1:
            days[i] = 1 ## possibly change this to raising an error instead of changing to 1
    for i in range(len(alphas)):
        if alphas[i] < 0:
            alphas[i] = 0
        if alphas[i] > 1:
            alphas[i] = 1
    current_time = start + (max(days) * 60)
    my_balance = balance()
    data = my_data
    current_signal = -1
    while current_time <= end:
        """
        if my_balance.get_my_balance() <= 1:
            print("ran out of money")
            return my_balance.get_my_balance()
        """
        high = (weights[0] * data.current_WMA(days[0], data.SMA(days[0]), current_time) +
                weights[1] * data.current_WMA(days[1], data.LMA(days[1]), current_time) +
                weights[2] * data.current_WMA(days[2], data.EMA(days[2], alphas[0]), current_time)) / sum(weights[:3])
        low = (weights[3] * data.current_WMA(days[3], data.SMA(days[3]), current_time) +
                weights[4] * data.current_WMA(days[4], data.LMA(days[4]), current_time) +
                weights[5] * data.current_WMA(days[5], data.EMA(days[5], alphas[1]), current_time)) / sum(weights[3:])
        last_signal = current_signal
        if high < low:
            current_signal = 1
            if last_signal == -1:
                #print(my_balance.get_my_balance())
                my_balance.buy(data.current_price(current_time))
        elif high > low:
            current_signal = -1
            if last_signal == 1:
                my_balance.sell(data.current_price(current_time))
                #print(my_balance.get_my_balance())
        else:
            current_signal = last_signal
        current_time += intervals
    if current_signal == 1:
        my_balance.sell(data.current_price(current_time - intervals)) #if balance is in bitcoin at the end of the time frame, sell it at the last available price
    return my_balance.get_my_balance()

def test(my_data=historic_data(), start=start, end=end):
    data = my_data
    start_price = data.current_price(start)
    end_price = data.current_price(end)
    my_balance = balance()
    my_balance.buy(start_price)
    my_balance.sell(end_price)
    return my_balance.get_my_balance()

"""
score = scoring([1, 1, 1, 1, 1, 1], [5, 10, 15, 20, 25, 30], [0.5, 0.5], time.mktime(datetime.datetime.strptime(start, "%d/%m/%Y").timetuple()), time.mktime(datetime.datetime.strptime(end, "%d/%m/%Y").timetuple()))
print("Final balance: ", score)
"""
        
            