'''
population.py
Description: Population of players that changes with training.
Author: Drew Curran
'''

from player import Player
from species import Species

class Population:
    def __init__(self, size, num_inputs, num_outputs, evaluation_function):
        self.population = {}
        self.population_size = size
        self.generation = 0
        self.generation_players = []
        self.innovation_history = []
        self.new_stage = False
        self.evaluation_function = evaluation_function
    
    ### Enforce natural selection on the players
    def new_generation(self):
        # Separate players into species
        

        # Calculate fitness of each player
        

        # Sort players in each species according to fitness
        

        # Cull each species with regard to fitness
        for species in self.population:
            species.cull(0.5)

        # Kill bad and unimproved species


        # Create next generation

        pass
    
    ###
    def speciate(self):
        pass
