from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import data_reader as dr
from NatureBasedAlgorithm import NatureBasedAlgorithm
from equations import MACD, original

class whale(NatureBasedAlgorithm):
    def __init__(self, scoring, days, weights, alphas, data):
        super().__init__(name="Whale", description="Whale Optimization",
                         scoring=scoring, days=days, weights=weights, alphas=alphas, data=data)
    
    def change_pos(self, new_pos):
        # super().change_pos(new_pos)
        new_pos[0:6] = np.clip(new_pos[0:6], 0.1, 1.0)  # weights
        new_pos[6:12] = np.clip(new_pos[6:12], 1, 100)  # days
        new_pos[12:14] = np.clip(new_pos[12:14], 0.0, 1.0)  # alphas
        # === ================= ===
        self.pos = new_pos
        self.score = self.scoring(self.pos[0:6], self.pos[6:12], self.pos[12:14], self.start, self.end, self.data)
    
    def optimise(self, num_agents, iterations, constant=1):
        whales = [whale(self.scoring, self.days, self.weights, self.alphas, self.intervals, self.start, self.end, self.data) for i in range(num_agents)]
        best_pos = np.zeros(14)
        best_score = 0
        for i in range(num_agents):
            if whales[i].score > best_score:
                best_score = whales[i].score
                best_pos = whales[i].pos
        for i in range(iterations):
            a = 2 * (1 - i / iterations)
            for j in range(num_agents):
                r = np.random.rand()
                A = 2 * a * r - a
                C = 2 * r
                l = (2 * np.random.rand()) - 1
                b = constant
                p = np.random.rand()
                if p < 0.5:
                    if abs(A) < 1:
                        D = abs(C * best_pos - whales[j].pos)
                        new_pos = best_pos - A * D
                        whales[j].change_pos(new_pos)

                    else:
                        random_whale = whales[np.random.randint(0, num_agents - 1)]
                        D = abs(C * random_whale.pos - whales[j].pos)
                        new_pos = random_whale.pos - A * D
                        whales[j].change_pos(new_pos)
                else:
                    d = abs(best_pos - whales[j].pos)
                    new_pos = d * np.exp(b * l) * np.cos(2 * np.pi * l) + best_pos
                    whales[j].change_pos(new_pos)
            for j in range(num_agents):
                if whales[j].score > best_score:
                    best_score = whales[j].score
                    best_pos = whales[j].pos
        return best_pos


# === RUN SECTION ===

# Define parameters
days = [1, 100, 6]     # [min_value, max_value, number_of_values]
weights = [0.1, 1, 6]  # [min_value, max_value, number_of_values]
alphas = [0.1, 1, 2]   # [min_value, max_value, number_of_values]
step_size = 86400    # step_size in seconds (for x minutes use 60 * x, for x hours use 60 * 60 * x, etc.)

models = dr.Train("01/01/2019", "30/07/2019", "01/08/2019", "30/12/2019", step_size)

alg: NatureBasedAlgorithm = whale(models.score, days, weights, alphas, step_size,
                                  models.train_start, models.train_end, models.train_data)

models.train_model(alg,  num_agents=10, num_iterations=10)
models.compare_models()

#10 whales, 10 iterations = $6.369556456832015 profit over baseline
#20 whales, 10 iterations = -$83.41078505604878 profit over baseline
#10 whales, 20 iterations = $6.369556456832015 profit over baseline