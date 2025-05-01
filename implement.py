import time
from data_reader import Train
from plot import Plotter
from MRO import manta_ray_algo
import matplotlib.pyplot as plt
#from hho import HHO
#from woa import WOA
#from sinecosine import sinecosine

TRAIN_START = "28/11/2016"
TRAIN_END = "31/12/2019"
TEST_START = "01/01/2020"
TEST_END = "01/03/2022"
DAY_BOUNDS = [1, 100, 6]
WEIGHT_BOUNDS = [0.001, 10, 6]
ALPHA_BOUNDS = [0.01, 1, 2]
MAX_ITER = 5

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

models = Train(TRAIN_START, TRAIN_END, TEST_START, TEST_END, step_size=86400 )  

start_time = time.perf_counter()

error = models.train_model(manta_ray_algo, DAY_BOUNDS, WEIGHT_BOUNDS, ALPHA_BOUNDS, MAX_ITER, num_pop=10, constant=1)
print(f"MFRO Time: {time.perf_counter() - start_time:.4f}")

#models.train_model(HHO, days_bound, weights_bound, alphas_bound, max_iter, num_pop=10, constant=1)
#print(f"HHO Time: {time.perf_counter() - start_time:.4f}")

#models.train_model(WOA, days_bound, weights_bound, alphas_bound, max_iter, num_pop=10, constant=1)
#print(f"WOA Time: {time.perf_counter() - start_time:.4f}")

#models.train_model(sinecosine, days_bound, weights_bound, alphas_bound, max_iter, num_pop=10, constant=1)
#print(f"SCO Time: {time.perf_counter() - start_time:.4f}")

models.compare_models()

#plt.plot(list(range(1,MAX_ITER+1)),error)
#plt.show()

test_plot = Plotter(models.test_data, TEST_START, TEST_END, models.models, equation)
test_plot.get_plotting_data()
for model in models.models:
    test_plot.plot_signals(model)
    test_plot.plot_buy_sell(model)
test_plot.plot_profit()