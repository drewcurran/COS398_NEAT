'''
species.py
Description: Species of genomes within a population.
Author: Drew Curran
'''

import numpy as np

from player import Player

class Species:
    def __init__(self, player):
        self.players = [player]
        self.representative_player = player
        self.max_fitness = player.fitness
        self.average_fitness = player.fitness
        self.staleness = 0
    
    ### Add player to species
    def add_player(self, player):
        self.players.append(player)

        # Test if new player is max fitness
        if player.fitness > self.max_fitness:
            self.representative_player = player
            self.max_fitness = player.fitness
            self.staleness = 0

        # Adjust average fitness
        self.average_fitness = (self.average_fitness * (len(self.players) - 1) + player.fitness) / len(self.players)
        
        return player
    
    ### Test whether player is in species
    def is_species(self, player):
        # Get parameters
        unmatching_coefficient = 1
        weight_coefficient = 0.5
        incompatibility_threshold = 3

        # Get values related to differences in genomes
        num_unmatching_genes, average_weight_difference = self.representative_player.nn.genome_difference(player.nn)
        
        # Get normalization for number of genes
        normalizer = len(player.nn.genes) - 20
        if normalizer < 1:
            normalizer = 1

        # Define incompatibility based on differences in genomes
        incompatibility = num_unmatching_genes * unmatching_coefficient / normalizer + average_weight_difference * weight_coefficient

        return incompatibility < incompatibility_threshold

    ### Sort the species according to fitness
    def sort(self):
        self.players.sort(key=lambda k: k.fitness)
        
        return self.players

    ### Cull the species
    def cull(self, proportion):
        assert proportion >=0 and proportion <= 1

        # Truncate species to proportion given
        desired_size = int(np.floor(len(self.players) * proportion))
        self.players = self.players[0:desired_size+1]

        self.staleness += 1
        self.average_fitness = 0
        for player in self.players:
            self.average_fitness += player.fitness
        self.average_fitness /= len(self.players)

        return self.players