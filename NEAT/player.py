'''
player.py
Description: A player that has an objective function based on application.
Author: Drew Curran
'''

import numpy as np

from genome import Genome

class Player:
    def __init__(self, num_inputs, num_outputs):
        self.nn = Genome(num_inputs, num_outputs)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.fitness = 0
        self.lifespan = 0
        self.generation = 0

    ### Make decision based on neural network
    def decide(self, inputs):
        # Sort decisions by output from neural network
        output = self.nn.forward_pass(inputs)
        
        # Choose output with the highest value
        decision = np.argmax(output)

        return decision
    
    ### Calculate fitness of the player
    def evaluate(self, evaluation_function):
        # Use given evaluation function
        self.fitness = evaluation_function(self)

        return self.fitness
    
    ### Mutate genome
    def mutate(self, history):
        self.nn.mutate_genome(history)

        return self.nn

    ### Create child player with self and another player as parents
    def crossover(self, player):
        child = Player(self.num_inputs, self.num_outputs)
        child.nn = self.nn.crossover(player.nn)

        return child
    
    ### Return a copy
    def clone(self):
        clone = Player(self.num_inputs, self.num_outputs)
        clone.nn = self.nn.clone()
        clone.fitness = self.fitness
        clone.generation = self.generation

        return clone
