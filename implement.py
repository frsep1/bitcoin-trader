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

OG_DAY_BOUNDS = [5, 40, 6]
OG_WEIGHT_BOUNDS = [0.001, 10, 6]
OG_ALPHA_BOUNDS = [0.01, 1, 2]

MACD_DAY_BOUNDS = [5, 100, 3]
MACD_WEIGHT_BOUNDS = [0.001, 10, 0]
MACD_ALPHA_BOUNDS = [0.01, 1, 3]

MAX_ITER = [1000]
NUM_AGENTS = [100]

def run(times, equation_name):
    csv_data = []
    if equation_name == "Original":
        days_bound = OG_DAY_BOUNDS
        weights_bound = OG_WEIGHT_BOUNDS
        alphas_bound = OG_ALPHA_BOUNDS
        equation= original

    elif equation_name == "MACD":
        days_bound = MACD_DAY_BOUNDS
        weights_bound = MACD_WEIGHT_BOUNDS
        alphas_bound = MACD_ALPHA_BOUNDS
        equation= MACD
    for i in range(times):
        print(f"Run {i+1} of {times}")
        for max_iter in MAX_ITER:
            for num_ag in NUM_AGENTS:
                print([max_iter,num_ag])
                base = Train(TRAIN_START, TRAIN_END, TEST_START, TEST_END, step_size=86400)
                og_plots = Plotter(base.test_data, base.models, num_ag)
                for alg in [whale, MRFO, hawk, SCA]:
                    start_time = time.perf_counter()
                    alg_instance = alg(base.score, days=days_bound, weights=weights_bound, alphas=alphas_bound, intervals = 86000, start = TRAIN_START, end = TRAIN_END, data=base.train_data)
                    base.train_model(alg_instance,  num_agents=num_ag, num_iterations=max_iter)
                    end_time = time.perf_counter() - start_time
                    print(f"{alg} Time: {end_time:.4f}")
                    csv_data.append(",".join(list(map(str, [alg_instance.name,
                                                            equation_name,
                                                            max_iter,
                                                            num_ag,
                                                            base.models[alg_instance.name].best_score,
                                                            base.score(base.test_data, equation, base.models[alg_instance.name].best_pos), 
                                                            end_time] + list(base.models[alg_instance.name].best_pos)))))
                og_plots.get_plotting_data()
                og_plots.plot_profit()
                og_plots.plot_scores_ot()

    return csv_data

csv_data = run(1, "Original")
#base = Train(TRAIN_START, TRAIN_END, TEST_START, TEST_END, step_size=86400)
#print(base.score(base.train_data, MACD, [12, 26, 9, 2/13, 2/27, 2/10]))

if CSV_OUTPUT:
    with open("results.csv", "w") as file:
        file.write("Alg,Equation,Iterations,Agents,Train Score,Test Score,Time,Params\n")
        for line in csv_data:
            file.write(line+"\n")
