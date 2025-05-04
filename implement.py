import time
from data_reader2wpadding import Train
from plot import Plotter
#from MRO import manta_ray_algo
import matplotlib.pyplot as plt
import numpy as np
#from hho import HHO
#from woa import WOA
#from sinecosine import sinecosine

TRAIN_START = "28/11/2017"
TRAIN_END = "31/12/2019"
TEST_START = "01/01/2020"
TEST_END = "01/03/2021"
DAY_BOUNDS = [1, 100, 6]
WEIGHT_BOUNDS = [0.001, 10, 6]
ALPHA_BOUNDS = [0.01, 1, 2]
MAX_ITER = 5

def original(params, data):
    points = data.df['close'].values
    high = np.add(
            params['weights'][0] * data.WMA(points, params['days'][0], data.SMA(params['days'][0])),
            params['weights'][1] * data.WMA(points, params['days'][1], data.LMA(params['days'][1])),
            params['weights'][2] * data.WMA(points, params['days'][2], data.EMA(params['days'][2], params['alphas'][0]))
            ) / sum(params['weights'][:3])
    low = np.add(
            params['weights'][3] * data.WMA(points, params['days'][3], data.SMA(params['days'][3])),
            params['weights'][4] * data.WMA(points, params['days'][4], data.LMA(params['days'][4])),
            params['weights'][5] * data.WMA(points, params['days'][5], data.EMA(params['days'][5], params['alphas'][1]))
            ) / sum(params['weights'][3:])
    return high, low

def MACD(params, data):
    points = data.df['close'].values
    macd = np.subtract(
            data.WMA(points, params['days'][0], data.EMA(params['days'][0], params['alphas'][0])),
            data.WMA(points, params['days'][1], data.EMA(params['days'][1], params['alphas'][1]))
            )
    signal = data.WMA(macd, params['days'][2], data.EMA(params['days'][2], params['alphas'][2]))
    return macd, signal

models = Train(TRAIN_START, TRAIN_END, TEST_START, TEST_END, step_size=86400 )  
#MACD({'days': [12, 26, 9], 'weights': [0.1, 0.2, 0.3, 0.4, 0.3, 0.4], 'alphas': [0.5, 0.5, 0.5]}, models.train_data)
#original({'days': [1, 2, 3, 1, 2, 3], 'weights': [0.1, 0.2, 0.3, 0.4, 0.3, 0.4], 'alphas': [0.1, 0.2, 0.3]}, models.train_data)
start_time = time.perf_counter()
models.score(models.train_data, MACD, [0.1, 0.2, 0.3, 0.4, 0.3, 0.4], [1, 2, 3, 1, 2, 3], [0.1, 0.2, 0.3])
print(f"Scoring Time: {time.perf_counter() - start_time:.4f}")

#error = models.train_model(manta_ray_algo, DAY_BOUNDS, WEIGHT_BOUNDS, ALPHA_BOUNDS, MAX_ITER, num_pop=10, constant=1)
#print(f"MFRO Time: {time.perf_counter() - start_time:.4f}")

#models.train_model(HHO, days_bound, weights_bound, alphas_bound, max_iter, num_pop=10, constant=1)
#print(f"HHO Time: {time.perf_counter() - start_time:.4f}")

#models.train_model(WOA, days_bound, weights_bound, alphas_bound, max_iter, num_pop=10, constant=1)
#print(f"WOA Time: {time.perf_counter() - start_time:.4f}")

#models.train_model(sinecosine, days_bound, weights_bound, alphas_bound, max_iter, num_pop=10, constant=1)
#print(f"SCO Time: {time.perf_counter() - start_time:.4f}")

#models.compare_models()

#plt.plot(list(range(1,MAX_ITER+1)),error)
#plt.show()

##test_plot = Plotter(models.test_data, models.models, original)
#test_plot.get_plotting_data()
#for model in models.models:
#    test_plot.plot_signals(model)
#    test_plot.plot_buy_sell(model)
#test_plot.plot_profit()