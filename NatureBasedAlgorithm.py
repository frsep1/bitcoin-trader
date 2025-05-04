from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import data_reader as dr

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
        self.initial_position()
        
        self.best_pos = None
        self.best_score = None
        
    def initial_position(self):
        """Initialize the position based on weights, days, and alphas."""
        w = np.random.uniform(self.weights[0], self.weights[1], size=self.weights[2])
        d = np.random.uniform(self.days[0], self.days[1], size=self.days[2])
        a = np.random.uniform(self.alphas[0], self.alphas[1], size=self.alphas[2])
        self.pos = np.concatenate((w, d, a))
        self.score = self.scoring(self.pos[0:6], self.pos[6:12], self.pos[12:14], self.start, self.end, self.data)
    
    def change_position(self, new_pos):
        self.pos = new_pos
        self.score = self.scoring(self.pos[0:6], self.pos[6:12], self.pos[12:14], self.start, self.end, self.data)
    
    @abstractmethod
    def optimise(self, num_agents, num_iterations, constant=1):
        """Abstract method for the optimization process."""
        pass
    
    def __str__(self):
        return f"{self.name}: {self.description}"