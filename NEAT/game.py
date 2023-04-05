'''
game.py
Description: Application and training for the NEAT algorithm.
Author: Drew Curran
'''

# Methods - tournament, round robin, against RL bot
# Hyperparameters - rounds, cull rate

import numpy as np

from player import Player
from population import Population

class Game:
    def __init__(self, population):
        self.population = population
    
    ### Play a game
    def play_game(self):
        for species in self.population.species:
            for player in species.players:
                for _ in range(10):
                    bias = [1]
                    inputs = bias + np.random.randint(0, 1, size=2).tolist()
                    decision = player.decide(inputs)
                    if inputs[0] == inputs[1] and decision == 0:
                        player.fitness += 1
                    elif inputs[0] != inputs[1] and decision == 1:
                        player.fitness += 1

def main():
    population = Population(1000, 2, 2)
    game = Game(population)
    game.play_game()
    population.update_generation()
    


if __name__ == '__main__':
    main()
