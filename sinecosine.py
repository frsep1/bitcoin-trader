import numpy as np
import pandas as pd
import data_reader as dr


class point():
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
def sinecosine(scoring, days, weights, alphas, num_pop, iterations, intervals, start, end, data, constant=1):
    points = [point(scoring, days, weights, alphas, intervals, start, end, data) for i in range(num_pop)]
    best_pos = np.zeros(14)
    best_score = 0
    a = constant
    for i in range(num_pop):
        if points[i].score > best_score:
            best_score = points[i].score
            best_pos = points[i].pos
    for i in range(iterations):
        r1 = a - (a * i / iterations)
        for j in range(num_pop):
            r2 = 2 * np.pi * np.random.rand()
            r4 = np.random.rand()
            if r4 < 0.5:
                new_pos = points[j].pos + r1 * np.sin(r2) * (best_pos - points[j].pos) # should be new_pos = points[j].pos + r1 * np.sin(r2) * (r3 * best_pos - points[j].pos) but cant figure out what r3 is
                points[j].change_pos(new_pos)
            else:
                new_pos = points[j].pos + r1 * np.cos(r2) * (best_pos - points[j].pos) #same as above
                points[j].change_pos(new_pos)
                
        for j in range(num_pop):
            if points[j].score > best_score:
                best_score = points[j].score
                best_pos = points[j].pos
    return best_pos


#models = dr.train("01/01/2023", "30/07/2023", "01/08/2023", "30/12/2023", step_size=60*100) #(train_start, train_end, test_start, test_end, step_size)
# the days, weights, alphas lists are in the form of [min_value, max_value, number_of_values]
# step_size is in seconds for x minutes use 60 * x for x hours use 60 * 60 * x and so on
#models.train_model(sinecosine, [1, 100, 6], [0.1, 1, 6], [0.1, 1, 2], max_iter=10, num_pop=10, constant=1) #(model, days, weights, alphas, max_iter, num_pop, constant)
#models.compare_models()