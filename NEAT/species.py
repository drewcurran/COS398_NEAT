'''
species.py
Description: Species of genomes within a population.
Author: Drew Curran
'''

import numpy as np

from NEAT.organism import Organism
from NEAT.parameters import EXCESS_COEFF, DISJOINT_COEFF, WEIGHT_DIFF_COEFF, NORMALIZE_OFFSET
from NEAT.parameters import CULL_RATE

class Species:
    def __init__(self):
        self.players = []
        self.max_fitness = 0
        self.staleness = 0
    
    ### Add player to species
    def add_player(self, player):
        self.players.append(player)
    
    ### Sort the species by fitness and get the fittest player
    def sort(self):
        self.players.sort(key=lambda k: k.fitness, reverse=True)
        self.fittest_player = self.players[0]
        if (self.fittest_player.fitness > self.max_fitness):
            self.max_fitness = self.fittest_player.fitness
            self.staleness = 0
        else:
            self.staleness += 1
    
    def cull(self):
        # Truncate species to proportion given
        desired_size = int((len(self.players) * CULL_RATE))
        self.players = self.players[0:desired_size+1]
    
    ### Get number of unmatching genes with another genome
    def genome_distance(self, player:Organism) -> float:
        # Choose random player in species
        species_player = self.players[np.random.randint(len(self.players))]

        # Initialize values
        excess = 0
        disjoint = 0
        weight_difference = []

        # Find excess, disjoint, and weight difference for all given player genes
        for gene in player.genes:
            corresponding_gene = species_player.get_gene(gene.innovation_label)
            if corresponding_gene is None:
                if len(species_player.innovation_labels) > 0 and gene.innovation_label > np.max(species_player.innovation_labels):
                    excess += 1
                else:
                    disjoint += 1
            else:
                weight_difference.append(abs(gene.weight - corresponding_gene.weight))

        # Find excess, disjoint, and weight difference for all species player genes
        for gene in species_player.genes:
            corresponding_gene = player.get_gene(gene.innovation_label)
            if corresponding_gene is None:
                if len(player.innovation_labels) > 0 and gene.innovation_label > np.max(player.innovation_labels):
                    excess += 1
                else:
                    disjoint += 1

        # Find number of genes in the larger genome
        if len(player.genes) > len(species_player.genes):
            num_genes = len(player.genes)
        else:
            num_genes = len(species_player.genes)
        
        # Apply offset and normalize
        num_genes -= NORMALIZE_OFFSET
        if num_genes <= 0:
            num_genes = 1
        
        # Get average of weight differences
        if len(weight_difference) > 0:
            avg_weight_difference = np.mean(weight_difference)
        else:
            avg_weight_difference = 0

        return EXCESS_COEFF * excess / num_genes + DISJOINT_COEFF * disjoint / num_genes + WEIGHT_DIFF_COEFF * avg_weight_difference
    
    ### Get the average species fitness
    def average_fitness(self) -> float:
        fitness = 0
        for player in self.players:
            fitness += player.fitness
        return fitness / len(self.players)