from abc import ABC
import numpy as np
import pandas as pd
import data_reader as dr
from NatureBasedAlgorithm import NatureBasedAlgorithm
from equations import original, MACD

class hawk(NatureBasedAlgorithm):
    def __init__(self, scoring, days, weights, alphas, intervals, start, end, data):
        super().__init__(name="HHO", description="Harris Hawks Optimization",
                         scoring=scoring, days=days, weights=weights, alphas=alphas, intervals=intervals,
                         start=start, end=end, data=data)
    
    def change_pos(self, new_pos):
        super().change_pos(new_pos)

    def optimise(self, num_agents, iterations, constant=1):
        hawks = [hawk(self.scoring, self.days, self.weights, self.alphas, self.intervals, self.start, self.end, self.data) for _ in range(num_agents)]
        best_pos = np.zeros(len(self.lower_bound))
        best_score = float('-inf')
        equation = original if len(self.lower_bound) == 14 else MACD
        
        for h in hawks:
            if h.score > best_score:
                best_score = h.score
                best_pos = h.pos

        for t in range(iterations):
            E0 = 2 * np.random.rand() - 1  # initial energy
            for i in range(num_agents):
                E = 2 * E0 * (1 - (t / iterations))  # energy decreases over time
                q = np.random.rand()
                r = np.random.rand()
                hawk_i = hawks[i]

                if abs(E) >= 1:
                    # Exploration
                    rand_hawk = hawks[np.random.randint(0, num_agents)]
                    new_pos = rand_hawk.pos - np.random.rand() * abs(rand_hawk.pos - 2 * np.random.rand() * hawk_i.pos)
                else:
                    # Exploitation
                    if r >= 0.5 and abs(E) < 1:
                        new_pos = best_pos - E * abs(best_pos - hawk_i.pos)
                    elif r >= 0.5 and abs(E) >= 0.5:
                        jump_strength = 2 * (1 - np.random.rand())
                        new_pos = best_pos - E * abs(jump_strength * best_pos - hawk_i.pos)
                    elif r < 0.5 and abs(E) >= 0.5:
                        jump_strength = 2 * (1 - np.random.rand())
                        X1 = best_pos - E * abs(jump_strength * best_pos - hawk_i.pos)
                        X1_score = self.scoring(self.data, equation, X1)
                        if X1_score > hawk_i.score:
                            new_pos = X1
                        else:
                            #new_pos = np.random.uniform(self.weights[0], self.weights[1], 6 + 6 + 2)
                            w = np.random.uniform(self.weights[0], self.weights[1], size=self.weights[2])
                            d = np.random.uniform(self.days[0], self.days[1], size=self.days[2])
                            a = np.random.uniform(self.alphas[0], self.alphas[1], size=self.alphas[2])
                            new_pos = np.concatenate((w, d, a))
                    else:
                        # Soft besiege with progressive rapid dives
                        Y = best_pos - E * abs(best_pos - hawk_i.pos)
                        Z = Y + np.random.normal(0, 1, size=Y.shape)  # Lévy flight could be added here
                        new_score_Y = self.scoring(self.data, equation, Y)
                        new_score_Z = self.scoring(self.data, equation, Z)
                        new_pos = Y if new_score_Y > new_score_Z else Z

                hawk_i.change_pos(new_pos)

            for h in hawks:
                if h.score > best_score:
                    best_score = h.score
                    best_pos = h.pos
            self.scores_over_time.append(best_score)
        return best_pos


# === RUN SECTION ===
if __name__ == "__main__":
    # Define parameters
    days = [1, 100, 6]     # [min_value, max_value, number_of_values]
    weights = [0.1, 1, 6]  # [min_value, max_value, number_of_values]
    alphas = [0.1, 1, 2]   # [min_value, max_value, number_of_values]
    step_size = 86400     # step_size in seconds (for x minutes use 60 * x, for x hours use 60 * 60 * x, etc.)

    models = dr.Train("01/01/2019", "30/07/2019", "01/08/2019", "30/12/2019", step_size)

    alg: NatureBasedAlgorithm = hawk(models.score, days, weights, alphas, step_size,
                                    models.train_start, models.train_end, models.train_data)

    models.train_model(alg,  num_agents=20, num_iterations=10)
    models.compare_models()

    #10 whales, 10 iterations = -$3.768671154374033 profit over baseline
    #20 whales, 10 iterations = -$83.41078505604878 profit over baseline
    #10 whales, 20 iterations = -4.478325905950442 profit over baseline