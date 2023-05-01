'''
species.py
Description: Species of genomes within a population.
Author: Drew Curran
'''

import numpy as np

from NEAT.organism import Organism
from NEAT.parameters import EXCESS_COEFF, DISJOINT_COEFF, WEIGHT_DIFF_COEFF
from NEAT.parameters import SPECIES_CULL_RATE

class Species:
    def __init__(self):
        self.players = []
        self.max_fitness = 0
        self.staleness = 0

    ### Reset species keeping the fittest player 
    def reset_species(self):
        self.players = [self.fittest_player]
    
    ### Add player to species
    def add_player(self, player):
        if len(self.players) == 0:
            self.fittest_player = player
        self.players.append(player)
    
    def cull(self) -> list[Organism]:
        # Sort the species by fitness
        self.players.sort(key=lambda k: k.fitness, reverse=True)

        
        
        # Truncate species to proportion given
        desired_size = int(len(self.players) * (1 - SPECIES_CULL_RATE))

        return self.players[desired_size:]
    
    ### Update species
    def update(self):
        # Average fitness over the species
        self.average_fitness = np.mean(list(map(lambda p: p.fitness, self.players)))
        
        # Fittest player
        self.fittest_player = self.players[0]

        # Update staleness
        if self.fittest_player.fitness > self.max_fitness:
            self.max_fitness = self.fittest_player.fitness
            self.staleness = 0
        else:
            self.staleness += 1
    
    ### Get number of unmatching genes with another genome
    def genome_distance(self, player:Organism) -> float:
        # Initialize values
        excess = 0
        disjoint = 0
        weight_difference = []

        # Find excess, disjoint, and weight difference for all given player genes
        for gene in player.genes:
            corresponding_gene = self.fittest_player.get_gene(gene.innovation_label)
            if corresponding_gene is None:
                if len(self.fittest_player.innovation_labels) == 0 or gene.innovation_label > np.max(self.fittest_player.innovation_labels):
                    excess += 1
                else:
                    disjoint += 1
            else:
                weight_difference.append(abs(gene.weight - corresponding_gene.weight))

        # Find excess, disjoint, and weight difference for all species player genes
        for gene in self.fittest_player.genes:
            corresponding_gene = player.get_gene(gene.innovation_label)
            if corresponding_gene is None:
                if len(player.innovation_labels) == 0 or gene.innovation_label > np.max(player.innovation_labels):
                    excess += 1
                else:
                    disjoint += 1

        # Find number of genes in the larger genome
        if len(player.genes) > len(self.fittest_player.genes):
            num_genes = len(player.genes)
        else:
            num_genes = len(self.fittest_player.genes)
        
        # Get average of weight differences
        if len(weight_difference) > 0:
            avg_weight_difference = np.mean(weight_difference)
        else:
            avg_weight_difference = 0

        return EXCESS_COEFF * excess / num_genes + DISJOINT_COEFF * disjoint / num_genes + WEIGHT_DIFF_COEFF * avg_weight_difference