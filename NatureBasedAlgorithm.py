from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import data_reader as dr
from equations import MACD, original

class NatureBasedAlgorithm:
    def __init__(self, name, description, scoring, days, weights, alphas, data):
        self.name = name
        self.description = description
        
        self.data = data

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
        if len(self.pos) == 6:
            self.pos = np.concatenate((d, a))
            self.score = self.scoring(self.data, MACD, days=self.pos[:3], alphas=self.pos[3:])
        else:
            self.score = self.scoring(self.data, original, self.pos[0:6], self.pos[6:12], self.pos[12:14])
    
    def change_pos(self, new_pos):
        self.pos = new_pos
        if len(self.pos) == 6:
            self.score = self.scoring(self.data, MACD, days=self.pos[:3], alphas=self.pos[3:])
        else:
            self.score = self.scoring(self.data, original, self.pos[0:6], self.pos[6:12], self.pos[12:14])
    
    @abstractmethod
    def optimize(self, num_agents, num_iterations, constant=1):
        """Abstract method for the optimization process."""
        pass
    
    def __str__(self):
        return f"{self.name}: {self.description}"