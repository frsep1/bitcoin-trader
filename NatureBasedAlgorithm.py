from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import data_reader as dr
from equations import MACD, original

class NatureBasedAlgorithm:
    def __init__(self, name, description, scoring, days, weights, alphas, intervals, start, end, data):
        self.name = name
        self.description = description
        
        self.start = start
        self.end = end
        self.data = data
        self.intervals = intervals

        self.days = days
        self.weights = weights
        self.alphas = alphas
        
        self.scoring = scoring
        self.pos = None
        self.score = None
        self.lower_bound, self.upper_bound = self.clip_bounds()
        self.initial_position()
        
        self.best_pos = None
        self.best_score = None
        self.scores_over_time = []
        
    def initial_position(self):
        """Initialize the position based on weights, days, and alphas."""
        total_params = self.weights[2] + self.days[2] + self.alphas[2]
        if total_params == 6:
            d = np.random.uniform(self.lower_bound[:3], self.upper_bound[:3], size=self.days[2])
            a = np.random.uniform(self.lower_bound[3:], self.upper_bound[3:], size=self.alphas[2])
            self.pos = np.concatenate((d, a))
            self.score = self.scoring(self.data, MACD, self.pos)
        else:
            w = np.random.uniform(self.lower_bound[:6], self.upper_bound[:6], size=self.days[2])
            d = np.random.uniform(self.lower_bound[6:12], self.upper_bound[6:12], size=self.days[2])
            a = np.random.uniform(self.lower_bound[12:14], self.upper_bound[12:14], size=self.alphas[2])
            self.pos = np.concatenate((w, d, a))
            self.score = self.scoring(self.data, original, self.pos)


    def change_pos(self, new_pos):
        # this avoids it going out of the boundaries
        new_pos = np.clip(new_pos, self.lower_bound, self.upper_bound)
        self.pos = new_pos
        if len(self.pos) == 6:
            self.score = self.scoring(self.data, MACD, self.pos)
        else:
            self.score = self.scoring(self.data, original, self.pos)

    def clip_bounds(self):
        lower_bound = []
        upper_bound = []
        
        for i in range(self.weights[2]):
            lower_bound.append(self.weights[0])
            upper_bound.append(self.weights[1])
        for i in range(self.days[2]):
            lower_bound.append(self.days[0])
            upper_bound.append(self.days[1])
        for i in range(self.alphas[2]):
            lower_bound.append(self.alphas[0])
            upper_bound.append(self.alphas[1])
        
        if len(lower_bound) == 14:
            lower_bound[6:12] = [5,5,5,15,15,15]
            upper_bound[6:12] = [15,15,15,40,40,40]
        lower_bound = np.array(lower_bound)
        upper_bound = np.array(upper_bound)
        return lower_bound, upper_bound

    
    @abstractmethod
    def optimise(self, num_agents, num_iterations, constant=1):
        """Abstract method for the optimization process."""
        pass
    
    def __str__(self):
        return f"{self.name}: {self.description}"