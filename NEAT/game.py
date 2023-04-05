'''
game.py
Description: Application and training for the NEAT algorithm.
Author: Drew Curran
'''

# Methods - tournament, round robin, against RL bot
# Hyperparameters - rounds, cull rate

import numpy as np

from population import Population

class Game:
    def __init__(self, population):
        self.population = population
    
    ### Play a game
    def play_game(self):
        for species in self.population.species:
            for player in species.players:
                won = 0
                for _ in range(10):
                    inputs = [1] + np.random.randint(2, size=2).tolist()
                    decision = player.decide(inputs)
                    if inputs[1] == inputs[2] and decision == 0:
                        won += 1
                    elif inputs[1] != inputs[2] and decision == 1:
                        won += 1
                player.fitness = won