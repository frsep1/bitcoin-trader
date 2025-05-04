from abc import ABC, abstractmethod
import numpy as np
import data_reader as dr
from random import randint
from NatureBasedAlgorithm import NatureBasedAlgorithm

# code below based heavily on the provided pseudo code provided in
# "Manta ray foraging optimization:An effective bio-inspired optimizer for engineering applications"
# Article by Zhao, Zhang & Wang

class MRFO(NatureBasedAlgorithm):
    def __init__(self, scoring, days, weights, alphas, intervals, start, end, data):
        super().__init__(name="MRFO", description="Manta Ray Foraging Optimisation",
                         scoring=scoring, days=days, weights=weights, alphas=alphas, intervals=intervals,
                         start=start, end=end, data=data)
    
    def change_pos(self, new_pos):
        new_pos[0:6] = np.clip(new_pos[0:6], 0.1, 1.0)  # weights
        new_pos[6:12] = np.clip(new_pos[6:12], 1, 100)  # days
        new_pos[12:14] = np.clip(new_pos[12:14], 0.0, 1.0)  # alphas
        # === ================= ===
        self.pos = new_pos
        self.score = self.scoring(self.pos[0:6], self.pos[6:12], self.pos[12:14], self.start, self.end, self.data)

    def optimise(self, num_agents, iterations, constant=1):
        #S is the somersault factor that decides the somersault range of manta rays and 𝑆 = 2, 𝑟2 and 𝑟3 are two random numbers in [0, 1]    
        somersault = 2
        
        # make sure it is weights, days, alpha pattern. Tried to change a few of these parameters but didn't do much
        lower_bound = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1,1, 1, 1, 1, 1, 1, 0.1, 0.1])
        upper_bound = np.array([1, 1, 1, 1, 1, 1, 100, 100, 100, 100, 100, 100, 1, 1])
        
        points = [MRFO(self.scoring, self.days, self.weights,self.alphas, self.intervals, self.start, self.end, self.data) for _ in range(num_agents)]
        
        best_fitness = float("-inf")

        best_solution = None
        
        for p in points:
            if p.score > best_fitness:
                best_fitness = p.score
                best_solution = p.pos
        
        for iteration in range(iterations):
            current_best = best_solution
            for i in range(num_agents):
                rand = np.random.rand()
                current = points[i].pos
                
                # cyclone foraging
                if rand < 0.5:
                    beta = np.random.rand()
                    rand2 = np.random.rand()
                    x_rand = points[randint(0, num_agents-1)].pos
                    if iteration / iterations < rand:
                        if i == 0:
                            new_position = x_rand + rand2 * (x_rand - current) + beta * (x_rand - current)
                        else:
                            new_position = x_rand + rand2 * (points[i - 1].pos - current) + beta * (x_rand - current)

                    else:
                        if i == 0:
                            new_position = current_best + rand2 * (current_best - current) + beta * (current_best - current)
                        else:
                            new_position = current_best + rand2 * (points[i - 1].pos - current) + beta * (current_best - current)
                
                # chain foraging
                else:
                    alpha = np.random.rand()
                    if i == 0:
                        new_position = current + rand + alpha * (current_best - current)
                    else:
                        new_position = current + rand * (points[i - 1].pos - current) + alpha * (current_best - current)
                
                # adds some degree of somersaulting
                rand3 = np.random.rand()
                rand4 = np.random.rand()
                somersault_move = somersault * (rand3 * current_best - rand4 * current)
                new_position += somersault_move
                
                # this avoids it going out of the boundaries
                new_position = np.clip(new_position, lower_bound, upper_bound)

                points[i].change_pos(new_position)

            for p in points:
                if p.score > best_fitness:
                    best_fitness = p.score
                    best_solution = p.pos.copy()


        return best_solution

# === RUN SECTION ===

# Define parameters
days = [1, 100, 6]     # [min_value, max_value, number_of_values]
weights = [0.1, 1, 6]  # [min_value, max_value, number_of_values]
alphas = [0.1, 1, 2]   # [min_value, max_value, number_of_values]
step_size = 86400     # step_size in seconds (for x minutes use 60 * x, for x hours use 60 * 60 * x, etc.)

models = dr.Train("01/01/2020", "30/07/2020", "01/08/2020", "30/12/2020", step_size)

alg: NatureBasedAlgorithm = MRFO(models.score, days, weights, alphas, step_size,
                                  models.train_start, models.train_end, models.train_data)

models.train_model(alg,  num_agents=10, num_iterations=10)
models.compare_models()
