'''
species.py
Description: Species of genomes within a population.
Author: Drew Curran
'''

from player import Player

class Species:
    def __init__(self, player):
        self.players = [player]
        self.representative_player = player
    
    ### Add player to species
    def add_player(self, player):
        self.players.append(player)
        
        return player
    
    ## TODO: use kwargs
    ### Test whether player is in species
    def is_species(self, player, unmatching_coefficient, weight_coefficient, incompatibility_threshold):
        # Get values related to differences in genomes
        num_unmatching_genes, average_weight_difference = self.representative_genome.genome_difference(player.nn)
        
        # Get normalization for number of genes
        normalizer = player.nn.num_genes - 20
        if normalizer < 1:
            normalizer = 1

        # Define incompatibility based on differences in genomes
        incompatibility = num_unmatching_genes * unmatching_coefficient / normalizer + average_weight_difference * weight_coefficient

        return incompatibility < incompatibility_threshold

    ### Sort the species according to fitness
    def sort(self):
        self.players.sort(key=lambda k: k.fitness)

    ### Cull the species
    def cull(self, proportion):
        assert proportion >=0 and proportion <= 1

        # Sort species according to fitness
        self.sort()

        # Truncate species to proportion given
        desired_size = len(self.players) * proportion
        self.players = self.players[:desired_size]

        return self.players
