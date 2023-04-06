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
        self.players.sort(key=lambda k: k.fitness, reverse=True)
        
        fittest_player = self.players[0]
        self.representative_player = fittest_player
        self.max_fitness = fittest_player.fitness
        
        return self.players

    ### Cull the species
    def cull(self, proportion):
        assert proportion >= 0 and proportion <= 1

        # Truncate species to proportion given
        desired_size = int((len(self.players) * proportion))
        self.players = self.players[0:desired_size+1]

        self.staleness += 1
        self.average_fitness = 0
        for player in self.players:
            self.average_fitness += player.fitness
        self.average_fitness /= len(self.players)

        return self.players
    
    ### Make a child player in the next generation
    def make_child(self, history):
        if np.random.uniform() < 0.25:
            parent = self.select_player()
            child = parent.clone()
        else:
            parent1 = self.select_player()
            parent2 = self.select_player()
            child = self.crossover(parent1, parent2)

        child.mutate(history)

        return child
    
    ### Crossover between two players in the species
    def crossover(self, parent1, parent2):
        # Fitter parent is the base
        if parent1.fitness > parent2.fitness:
            base = parent1
            mate = parent2
        else:
            base = parent2
            mate = parent1

        # Clone from self
        child = base.crossover(mate)
        
        return child
    
    ### Select player in species
    def select_player(self):
        fitness_sum = 0
        for player in self.players:
            fitness_sum += player.fitness

        target_sum = np.random.uniform(fitness_sum)
        for player in self.players:
            target_sum -= player.fitness
            if target_sum < 0:
                return player
                
        return self.players[0]