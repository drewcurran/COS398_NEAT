'''
species.py
Description: Species of genomes within a population.
Author: Drew Curran
'''

import numpy as np

from NEAT.organism import Organism
from NEAT.history import InnovationMarker
from NEAT.parameters import MAX_WEIGHT
from NEAT.parameters import PR_CLONE
from NEAT.parameters import GENE_OFFSET, EXCESS_DISJOINT_COEFF, WEIGHT_DIFF_COEFF
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
        desired_size = int(len(self.players) * (1 - SPECIES_CULL_RATE)) + 1
        culled_players = self.players[desired_size:]
        self.players = self.players[:desired_size]

        return culled_players
    
    ### Update species
    def update(self):
        if len(self.players) == 0:
            self.average_fitness = 0
            return 0
        
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
        
        return self.average_fitness
    
    ### Reproduce a new child
    def reproduce(self, innovations: list[InnovationMarker]) -> Organism:
        if np.random.uniform() < PR_CLONE:
            return self.clone(innovations)
        else:
            return self.crossover(innovations)

    ### Crossover between two parent genomes
    def crossover(self, innovations: list[InnovationMarker]) -> Organism:
        # Get parent
        parent1 = np.random.choice(self.players)
        parent2 = np.random.choice(self.players)

        # Fitter parent is the base
        if parent1.fitness > parent2.fitness:
            base_mate = parent1
            partner_mate = parent2
        else:
            base_mate = parent2
            partner_mate = parent1

        return base_mate.crossover(innovations, partner_mate)
    
    ### Clone from one parent genome
    def clone(self, innovations: list[InnovationMarker]) -> Organism:
        # Get parent
        parent = np.random.choice(self.players)

        return parent.clone(innovations)

    ### Get number of unmatching genes with another genome
    def genome_distance(self, player:Organism) -> float:
        # Initialize values
        excess_disjoint = 0
        weight_difference = []

        # Find excess, disjoint, and weight difference for all given player genes
        for gene in self.fittest_player.genes:
            corresponding_gene = player.get_gene(gene.label)
            if corresponding_gene is None:
                excess_disjoint += 1
            else:
                weight_difference.append(abs(gene.weight - corresponding_gene.weight))

        # Find number of genes in the larger genome
        if len(player.genes) > len(self.fittest_player.genes):
            num_genes = len(player.genes)
        else:
            num_genes = len(self.fittest_player.genes)

        normalizer = num_genes - GENE_OFFSET
        if normalizer <= 0:
            normalizer = 1
        
        # Get average of weight differences
        if len(weight_difference) > 0:
            avg_weight_difference = np.mean(weight_difference)
        else:
            avg_weight_difference = MAX_WEIGHT

        return EXCESS_DISJOINT_COEFF * excess_disjoint / num_genes + WEIGHT_DIFF_COEFF * avg_weight_difference
