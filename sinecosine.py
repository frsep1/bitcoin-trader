from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import data_reader as dr
from NatureBasedAlgorithm import NatureBasedAlgorithm

class SCA(NatureBasedAlgorithm):
    def __init__(self, scoring, days, weights, alphas, intervals, start, end, data):
        super().__init__(name="SineCosine", description="Sine Cosine Optimization",
                         scoring=scoring, days=days, weights=weights, alphas=alphas, intervals=intervals,
                         start=start, end=end, data=data)
    
    def change_pos(self, new_pos):
        super().change_pos(new_pos)
    
    def optimise(self, num_agents, iterations, constant=1):
        points = [SCA(self.scoring, self.days, self.weights, self.alphas, self.intervals, self.start, self.end, self.data) for i in range(num_agents)]
        best_pos = np.zeros(14)
        best_score = 0
        a = constant
        for i in range(num_agents):
            if points[i].score > best_score:
                best_score = points[i].score
                best_pos = points[i].pos
        for i in range(iterations):
            r1 = a - (a * i / iterations)
            for j in range(num_agents):
                r2 = 2 * np.pi * np.random.rand()
                r4 = np.random.rand()
                if r4 < 0.5:
                    new_pos = points[j].pos + r1 * np.sin(r2) * (best_pos - points[j].pos) # should be new_pos = points[j].pos + r1 * np.sin(r2) * (r3 * best_pos - points[j].pos) but cant figure out what r3 is
                    points[j].change_pos(new_pos)
                else:
                    new_pos = points[j].pos + r1 * np.cos(r2) * (best_pos - points[j].pos) #same as above
                    points[j].change_pos(new_pos)
                    
            for j in range(num_agents):
                if points[j].score > best_score:
                    best_score = points[j].score
                    best_pos = points[j].pos
        return best_pos


# === RUN SECTION ===

# Define parameters
days = [1, 100, 6]     # [min_value, max_value, number_of_values]
weights = [0.1, 1, 6]  # [min_value, max_value, number_of_values]
alphas = [0.1, 1, 2]   # [min_value, max_value, number_of_values]
step_size = 86400     # step_size in seconds (for x minutes use 60 * x, for x hours use 60 * 60 * x, etc.)

models = dr.Train("01/01/2019", "30/07/2019", "01/08/2019", "30/12/2019", step_size)

alg: NatureBasedAlgorithm = SCA(models.score, days, weights, alphas, step_size,
                                  models.train_start, models.train_end, models.train_data)

models.train_model(alg,  num_agents=10, num_iterations=10)
models.compare_models()
