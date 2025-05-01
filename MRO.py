import numpy as np
import data_reader as dr
from random import randint

# code below based heavily on the provided pseudo code provided in
# "Manta ray foraging optimization:An effective bio-inspired optimizer for engineering applications"
# Article by Zhao, Zhang & Wang

class MRFO():
    def __init__(self, scoring, days, weights, alphas, intervals, start, end, data):
        self.scoring = scoring
        self.start = start
        self.end = end
        self.data = data
        self.intervals = intervals

        # same as WOA setup
        w = np.random.uniform(weights[0], weights[1], size=weights[2])
        d = np.random.uniform(days[0], days[1], size=days[2])
        a = np.random.uniform(alphas[0], alphas[1], size=alphas[2])
        self.pos = np.concatenate((w, d, a))
        self.score = scoring(self.pos[0:6], self.pos[6:12], self.pos[12:14], start, end, data)

    def change_pos(self, new_pos):
        self.pos = new_pos
        self.score = self.scoring(self.pos[0:6], self.pos[6:12], self.pos[12:14], self.start, self.end, self.data)

#max_iterations = 100

def manta_ray_algo(scoring, days, weights, alphas, num_pop, max_iterations, intervals, start, end, data, constant=1):
    #S is the somersault factor that decides the somersault range of manta rays and 𝑆 = 2, 𝑟2 and 𝑟3 are two random numbers in [0, 1]
    error_rate = []
    
    somersault = 2

    lower_bound = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1, 1, 1, 1, 1, 0.1, 0.1])
    upper_bound = np.array([100, 100, 100, 100, 100, 100, 30, 30, 30, 30, 30, 30, 0.99, 0.99])

    points = [MRFO(scoring, days, weights, alphas, intervals, start, end, data) for _ in range(num_pop)]

    best_fitness = -1
    best_solution = None

    for p in points:
        if p.score > best_fitness:
            best_fitness = p.score
            best_solution = p.pos.copy()

    for iteration in range(max_iterations):
        current_best = best_solution.copy()
        for i in range(num_pop):
            rand = np.random.rand()
            current = points[i].pos

            # cyclone foraging
            if rand < 0.5:
                beta = np.random.rand()
                rand2 = np.random.rand()
                x_rand = points[randint(1, num_pop-1)].pos
                if iteration / max_iterations < rand:
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
                #print("Found new leader")
                best_fitness = p.score
                best_solution = p.pos.copy()
        error_rate.append(best_fitness)
    return best_solution, error_rate

# training model copied from, not tested yet
#models = dr.Train(TRAIN_START, TRAIN_END, TEST_START, TEST_END, step_size=86400 )  # (train_start, train_end, test_start, test_end, step_size)
# the days, weights, alphas lists are in the form of [min_value, max_value, number_of_values]
# step_size is in seconds for x minutes use 60 * x for x hours use 60 * 60 * x and so on
# (model, days, weights, alphas, max_iter=1000, num_pop=10, constant=1)
#models.train_model(manta_ray_algo, [1, 100, 6], [0.001, 10, 6], [0.01, 5, 2], max_iter=50, num_pop=10, constant=1)  # (model, days, weights, alphas, max_iter, num_pop, constant)
#models.compare_models()

