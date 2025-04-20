import pandas as pd
import numpy as np
import time
import datetime
start = "01/12/2023"
end = "31/12/2023"

class historic_data:
    def __init__(self, start=start, end=end):
        self.start = time.mktime(datetime.datetime.strptime(start, "%d/%m/%Y").timetuple())
        self.end = time.mktime(datetime.datetime.strptime(end, "%d/%m/%Y").timetuple())        
        self.df = pd.read_csv('btcusd_1-min_data.csv')
        self.df = self.df[(self.df['Timestamp'] >= self.start) & (self.df['Timestamp'] <= self.end)]

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
        current_df = self.df[self.df['Timestamp'] <= time]
        p = current_df['Close'].values
        if len(p) < n:
            raise ValueError("Not enough data points to calculate WMA.")
        else:
            return self.WMA(p, n, kernel)[-1]
    
    def current_price(self, time):
        current_row = self.df[self.df['Timestamp'] <= time].iloc[-1]
        return current_row['Close']
        
    
    

historic_data = historic_data()
t = time.mktime(datetime.datetime.strptime("10/12/2023", "%d/%m/%Y").timetuple())
print(historic_data.current_price(t))
print(historic_data.current_WMA(5, historic_data.LMA(5), t))
