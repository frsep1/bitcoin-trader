import time
from data_reader2wpadding import Train
from plot import Plotter
from MRO import manta_ray_algo
import matplotlib.pyplot as plt
from equations import MACD, original
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

models = Train(TRAIN_START, TRAIN_END, TEST_START, TEST_END, step_size=86400 )  

start_time = time.perf_counter()

models.train_model(manta_ray_algo, DAY_BOUNDS, WEIGHT_BOUNDS, ALPHA_BOUNDS, MAX_ITER, num_pop=10, constant=1)
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
#test_models = {"TESTER": [0.5, 1.5, 0.5, 0.5, 1.5, 0.5, 9, 8, 7, 26, 25, 20, 0.1, 0.2]}#, "TESTER2": [12, 26, 9, 0.5, 0.1, 0.2]}
#test_plot = Plotter(models.test_data, test_models)
#test_plot.get_plotting_data()
#for model in test_models:
#    test_plot.plot_signals(model)
#    test_plot.plot_buy_sell(model)
#test_plot.plot_profit()















#MACD({'days': [12, 26, 9], 'weights': [0.1, 0.2, 0.3, 0.4, 0.3, 0.4], 'alphas': [0.5, 0.5, 0.5]}, models.train_data)
#original({'days': [1, 2, 3, 1, 2, 3], 'weights': [0.1, 0.2, 0.3, 0.4, 0.3, 0.4], 'alphas': [0.1, 0.2, 0.3]}, models.train_data)