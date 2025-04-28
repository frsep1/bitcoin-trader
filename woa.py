import numpy as np
import pandas as pd
import data_reader as dr


class whale():
    def __init__(self, scoring, days, weights, alphas, intervals, start, end, data):
        self.scoring = scoring
        self.start = start
        self.end = end
        self.data = data
        self.intervals = intervals
        w = np.random.uniform(weights[0], weights[1], size=weights[2])
        d = np.random.uniform(days[0], days[1], size=days[2])
        a = np.random.uniform(alphas[0], alphas[1], size=alphas[2])
        self.pos = np.concatenate((w, d, a))       
        self.score = scoring(self.pos[0:6], self.pos[6:12], self.pos[12:14], start, end, data)
    def change_pos(self, new_pos):
        self.pos = new_pos
        self.score = self.scoring(self.pos[0:6], self.pos[6:12], self.pos[12:14], self.start, self.end, self.data)
        
#model(self.score, days, weights, alphas, num_whales, max_iter, step_size, constant, self.train_start, self.train_end, self.train_data)
def WOA(scoring, days, weights, alphas, num_whales, iterations, intervals, start, end, data, spiral_constant=1):
    whales = [whale(scoring, days, weights, alphas, intervals, start, end, data) for i in range(num_whales)]
    best_pos = np.zeros(14)
    best_score = 0
    for i in range(num_whales):
        if whales[i].score > best_score:
            best_score = whales[i].score
            best_pos = whales[i].pos
    for i in range(iterations):
        a = 2 * (1 - i / iterations)
        a2 = -1 + i * (-1 / iterations)
        for j in range(num_whales):
            r = np.random.rand()
            A = 2 * a * r - a
            C = 2 * r
            l = (2 * np.random.rand()) - 1
            b = spiral_constant
            p = np.random.rand()
            if p < 0.5:
                if abs(A) < 1:
                    D = abs(C * best_pos - whales[j].pos)
                    new_pos = best_pos - A * D
                    whales[j].change_pos(new_pos)

                else:
                    random_whale = whales[np.random.randint(0, num_whales - 1)]
                    D = abs(C * random_whale.pos - whales[j].pos)
                    new_pos = random_whale.pos - A * D
                    whales[j].change_pos(new_pos)
            else:
                d = abs(best_pos - whales[j].pos)
                new_pos = d * np.exp(b * l) * np.cos(2 * np.pi * l) + best_pos
                whales[j].change_pos(new_pos)
        for j in range(num_whales):
            if whales[j].score > best_score:
                best_score = whales[j].score
                best_pos = whales[j].pos
    return best_pos



models = dr.train("01/01/2023", "30/07/2023", "01/08/2023", "30/12/2023", step_size=60*100) #(train_start, train_end, test_start, test_end, step_size)
# the days, weights, alphas lists are in the form of [min_value, max_value, number_of_values]
# step_size is in seconds for x minutes use 60 * x for x hours use 60 * 60 * x and so on
models.train_model(WOA, [1, 100, 6], [0.1, 1, 6], [0.1, 1, 2], max_iter=120, num_pop=10, constant=1) #(model, days, weights, alphas, max_iter, num_pop, constant)
models.compare_models()

#10 whales, 10 iterations = $6.369556456832015 profit over baseline
#20 whales, 10 iterations = -$83.41078505604878 profit over baseline
#10 whales, 20 iterations = $6.369556456832015 profit over baseline



        
                

