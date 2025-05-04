import time
from data_reader import Train
from plot import Plotter
from MRO import MRFO
#from hho import HHO
from woa import whale
from NatureBasedAlgorithm import NatureBasedAlgorithm
from sinecosine import SCA

TRAIN_START = "28/11/2017"
TRAIN_END = "31/12/2019"
TEST_START = "01/01/2020"
TEST_END = "01/03/2021"
OG_DAY_BOUNDS = [1, 100, 6]
OG_WEIGHT_BOUNDS = [0.001, 10, 6]
OG_ALPHA_BOUNDS = [0.01, 1, 2]
MACD_DAY_BOUNDS = [1, 100, 3]
MACD_WEIGHT_BOUNDS = [0.001, 10, 0]
MACD_ALPHA_BOUNDS = [0.01, 1, 3]
MAX_ITER = 1000
NUM_AGENTS = 20

models = Train(TRAIN_START, TRAIN_END, TEST_START, TEST_END, step_size=86400)
print("\n")

for d, w, a in [(MACD_DAY_BOUNDS, MACD_WEIGHT_BOUNDS, MACD_ALPHA_BOUNDS), (OG_DAY_BOUNDS, OG_WEIGHT_BOUNDS, OG_ALPHA_BOUNDS)]:
    for alg in [MRFO, whale, SCA]:
        start_time = time.perf_counter()
        alg_instance = alg(models.score, d, w, a, models.train_data)
        models.train_model(alg_instance,  num_agents=NUM_AGENTS, num_iterations=MAX_ITER)
        print(f"{alg} Time: {time.perf_counter() - start_time:.4f}")

models.compare_models()

test_plots = Plotter(models.test_data, models.models)
test_plots.get_plotting_data()
for model in models.models:
    test_plots.plot_signals(model)
test_plots.plot_profit()















#models.train_model(HHO, days_bound, weights_bound, alphas_bound, max_iter, num_pop=10, constant=1)
#print(f"HHO Time: {time.perf_counter() - start_time:.4f}")

#models.train_model(WOA, days_bound, weights_bound, alphas_bound, max_iter, num_pop=10, constant=1)
#print(f"WOA Time: {time.perf_counter() - start_time:.4f}")

#models.train_model(sinecosine, days_bound, weights_bound, alphas_bound, max_iter, num_pop=10, constant=1)
#print(f"SCO Time: {time.perf_counter() - start_time:.4f}")



#MACD({'days': [12, 26, 9], 'weights': [0.1, 0.2, 0.3, 0.4, 0.3, 0.4], 'alphas': [0.5, 0.5, 0.5]}, models.train_data)
#original({'days': [1, 2, 3, 1, 2, 3], 'weights': [0.1, 0.2, 0.3, 0.4, 0.3, 0.4], 'alphas': [0.1, 0.2, 0.3]}, models.train_data)