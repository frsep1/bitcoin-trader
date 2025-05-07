from abc import ABC
import numpy as np
import pandas as pd
import data_reader as dr
from NatureBasedAlgorithm import NatureBasedAlgorithm
from hho import hawk
from MRO import MRFO
from sinecosine import point
from woa import whale

class BitcoinTradingApp:
    def __init__(self, models):
        self.models = models

    def create_algorithm(self, algorithm_name: str, scoring, days, weights, alphas, intervals, start, end, data) -> NatureBasedAlgorithm:
        if algorithm_name == "SineCosine":
            return point(scoring, days, weights, alphas, intervals, start, end, data)
        elif algorithm_name == "MRFO":
            return MRFO(scoring, days, weights, alphas, intervals, start, end, data)
        elif algorithm_name == "HHO":
            return hawk(scoring, days, weights, alphas, intervals, start, end, data)
        elif algorithm_name == "Whale":
            return whale(scoring, days, weights, alphas, intervals, start, end, data)
        else:
            raise ValueError(f"Unknown algorithm name: {algorithm_name}")

    def run(self, algorithm, scoring, days, weights, alphas, intervals, start, end, data):
        algorithm = self.create_algorithm(algorithm_name, scoring, days, weights, alphas, intervals, start, end, data)
        self.models.train_model(algorithm, num_agents=20, num_iterations=10)

# Define runtime parameters
step_size = 86400
models = dr.Train("01/01/2019", "30/07/2019", "01/08/2019", "30/12/2019", step_size)

# Initialize the app
app = BitcoinTradingApp(models)

# Runtime logic to run each algorithm
algorithms = ["HHO", "Whale", "MRFO", "SineCosine"]

for algorithm_name in algorithms:
    # define parameters for each algorithm
    scoring = models.score
    days = [1, 100, 6]
    weights = [0.1, 1, 6]
    alphas = [0.1, 1, 2]
    intervals = 10
    start = models.train_start
    end = models.train_end
    data = models.train_data
    
    # run algorithm
    print(f"\nRunning {algorithm_name} algorithm...")
    app.run(algorithm_name, models.score, days, weights, alphas, step_size,
            models.train_start, models.train_end, models.train_data)
    
models.compare_models()