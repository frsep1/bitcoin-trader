import time
from data_reader import Train
from equations import original, MACD
from plot import Plotter
from MRO import MRFO
from hho import hawk
from woa import whale
from sinecosine import SCA

CSV_OUTPUT = True
TRAIN_START = "28/11/2014"
TRAIN_END = "31/12/2019"
TEST_START = "01/01/2020"
TEST_END = "01/03/2022"
OG_DAY_BOUNDS = [5, 50, 6]
OG_WEIGHT_BOUNDS = [0.001, 20, 6]
OG_ALPHA_BOUNDS = [0.01, 1, 2]
MACD_DAY_BOUNDS = [5, 50, 3]
MACD_WEIGHT_BOUNDS = [0.001, 20, 0]
MACD_ALPHA_BOUNDS = [0.01, 1, 3]
MAX_ITER = 10
NUM_AGENTS = 30

print("\n++++++++++++++++++++++++ ORIGINAL EQUATION ++++++++++++++++++++++++ ")
original_models = Train(TRAIN_START, TRAIN_END, TEST_START, TEST_END, step_size=86400)
for alg in [MRFO, whale, SCA, hawk]:
    start_time = time.perf_counter()
    alg_instance = alg(original_models.score, days=OG_DAY_BOUNDS, weights=OG_WEIGHT_BOUNDS, alphas=OG_ALPHA_BOUNDS, intervals = 86000, start = TRAIN_START, end = TRAIN_END, data=original_models.train_data)
    original_models.train_model(alg_instance,  num_agents=NUM_AGENTS, num_iterations=MAX_ITER)
    print(f"{alg} Time: {time.perf_counter() - start_time:.4f}")
original_models.compare_models()

print("\n++++++++++++++++++++++++ MACD EQUATION ++++++++++++++++++++++++ ")
MACD_models = Train(TRAIN_START, TRAIN_END, TEST_START, TEST_END, step_size=86400)
for alg in [MRFO, whale, SCA, hawk]:
    start_time = time.perf_counter()
    alg_instance = alg(MACD_models.score, days=MACD_DAY_BOUNDS, weights=MACD_WEIGHT_BOUNDS, alphas=MACD_ALPHA_BOUNDS, intervals = 86000, start = TRAIN_START, end = TRAIN_END, data=MACD_models.train_data)
    MACD_models.train_model(alg_instance,  num_agents=NUM_AGENTS, num_iterations=MAX_ITER)
    print(f"{alg} Time: {time.perf_counter() - start_time:.4f}")
MACD_models.compare_models()

og_plots = Plotter(original_models.test_data, original_models.models)
og_plots.get_plotting_data()
for model in original_models.models:
    og_plots.plot_signals(model)
og_plots.plot_profit()

macd_plots = Plotter(MACD_models.test_data, MACD_models.models)
macd_plots.get_plotting_data()
for model in MACD_models.models:
    macd_plots.plot_buy_sell(model)
macd_plots.plot_profit()


if CSV_OUTPUT:
    with open("results.csv", "w") as file:
        file.write("Alg,Equation,Iterations,Agents,Train Score,Test Score,Params\n")
        for og_model in original_models.models:
            m = original_models.models[og_model]
            line = ",".join(list(map(str, [og_model,"Original",MAX_ITER,NUM_AGENTS,m.best_score,original_models.score(original_models.test_data, original, m.best_pos)] + list(m.best_pos))))
            file.write(line+"\n")

        for macd_model in MACD_models.models:
            m = MACD_models.models[macd_model]
            line = ",".join(list(map(str, [macd_model,"MACD",MAX_ITER,NUM_AGENTS,m.best_score,original_models.score(MACD_models.test_data, MACD, m.best_pos)] + list(m.best_pos))))
            file.write(line+"\n")















#models.train_model(HHO, days_bound, weights_bound, alphas_bound, max_iter, num_pop=10, constant=1)
#print(f"HHO Time: {time.perf_counter() - start_time:.4f}")

#models.train_model(WOA, days_bound, weights_bound, alphas_bound, max_iter, num_pop=10, constant=1)
#print(f"WOA Time: {time.perf_counter() - start_time:.4f}")

#models.train_model(sinecosine, days_bound, weights_bound, alphas_bound, max_iter, num_pop=10, constant=1)
#print(f"SCO Time: {time.perf_counter() - start_time:.4f}")



#MACD({'days': [12, 26, 9], 'weights': [0.1, 0.2, 0.3, 0.4, 0.3, 0.4], 'alphas': [0.5, 0.5, 0.5]}, models.train_data)
#original({'days': [1, 2, 3, 1, 2, 3], 'weights': [0.1, 0.2, 0.3, 0.4, 0.3, 0.4], 'alphas': [0.1, 0.2, 0.3]}, models.train_data)