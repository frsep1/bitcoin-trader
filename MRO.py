import numpy as np
import data_reader2 as dr
from random import randint

# code below based heavily on the provided pseudo code provided in
# "Manta ray foraging optimization:An effective bio-inspired optimizer for engineering applications"
# Article by Zhao, Zhang & Wang

class MRFO():
    def __init__(self, scoring, weights, days, alphas, intervals, start, end, data):
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

def manta_ray_algo(scoring, weights, days, alphas, num_pop, max_iterations, intervals, start, end, data, constant=1):

    #S is the somersault factor that decides the somersault range of manta rays and 𝑆 = 2, 𝑟2 and 𝑟3 are two random numbers in [0, 1]
    somersault = 2

    # make sure it is weights, days, alpha pattern. Tried to change a few of these parameters but didn't do much

    lower_bound = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1,1, 1, 1, 1, 1, 1, 0.1, 0.1])
    upper_bound = np.array([1, 1, 1, 1, 1, 1, 100, 100, 100, 100, 100, 100, 1, 1])

    points = [MRFO(scoring, weights, days, alphas, intervals, start, end, data) for _ in range(num_pop)]

    best_fitness = -1
    best_solution = None

    for p in points:
        if p.score > best_fitness:
            best_fitness = p.score
            best_solution = p.pos

    for iteration in range(max_iterations):

        current_best = best_solution
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
                best_fitness = p.score
                best_solution = p.pos

    return best_solution

#models = dr.train("01/03/2021", "15/06/2021", "01/10/2021", "01/12/2021",
#step_size=86400) always returns a negative. Not sure why.

models = dr.train("01/01/2021", "15/06/2021", "01/10/2021", "01/12/2021", step_size=86400)
# sometimes doesn't find a optima of 1093 and just returns 1000

#models = dr.train("01/07/2020", "15/06/2021", "01/07/2021", "01/09/2021",step_size=86400)

# constant use?
models.train_model(manta_ray_algo,[0.1, 1, 6],[1, 100, 6], [0.1, 1, 2], max_iter=15, num_pop=15,
                   constant=1)  # (model, weights, days, alphas, max_iter, num_pop, constant)
models.compare_models()

# if I want to try to make a high and low
weight = [1.0, 2.0] * 3 + [0.1, 0.9] * 3

days = [1, 100] * 3 + [1, 100] * 3

alpha = [0.1, 1, 0.1, 1]



