'''
population.py
Description: Population of players that changes with training.
Author: Drew Curran
'''

import numpy as np

from NEAT.organism import Organism
from NEAT.species import Species
from NEAT.history import InnovationHistory
from NEAT.parameters import MAX_WEIGHT
from NEAT.parameters import PR_CLONE, PR_INTERSPECIES, PR_INHERIT_FITTER, PR_ENABLE
from NEAT.parameters import PR_MUTATE_NEURON, PR_MUTATE_GENE, PR_MUTATE_WEIGHTS
from NEAT.parameters import GENETIC_DISTANCE, SPECIES_WANTED, DISTANCE_MODIFIER
from NEAT.parameters import MAX_STALENESS

class Population:
    def __init__(self, num_players:int, num_inputs:int, num_outputs:int):
        self.population_size = num_players
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.generation = 0
        self.players = []
        self.species = []
        self.species_threshold = GENETIC_DISTANCE

    ### Create a new generation
    def new_generation(self) -> list[Organism]:
        if self.generation == 0:
            self.initialize_population()
        else:
            self.reproduce()
        self.mutate()
        self.speciate()
        self.generation += 1

    ### Enforce natural selection on the players that have played
    def update_generation(self):
        for species in self.species:
            species.sort()
            species.cull()

        self.update_species()

    ### Initialize the population
    def initialize_population(self) -> list[Organism]:
        self.innovation_history = InnovationHistory(self.num_inputs + self.num_outputs + 2)
        self.players = [Organism(self.num_inputs, self.num_outputs) for _ in range(self.population_size)]

    ### Reproduce using surviving players
    def reproduce(self) -> list[Organism]:
        # Initialize player list
        players = []

        # Add fittest player from each species to population
        for species in self.species:
            players.append(species.fittest_player)

        # Generate players from the previous population
        while len(players) < self.population_size:
            # Clone from a single parent
            if np.random.uniform() < PR_CLONE:
                parent = self.players[np.random.randint(len(self.players))]
                child = self.clone(parent)

            # Crossover between two parents
            else:
                if np.random.uniform() < PR_INTERSPECIES:
                    parent1 = self.players[np.random.randint(len(self.players))]
                    parent2 = self.players[np.random.randint(len(self.players))]
                else:
                    species = self.species[np.random.randint(len(self.species))]
                    parent1 = species.players[np.random.randint(len(species.players))]
                    parent2 = species.players[np.random.randint(len(species.players))]
                child = self.crossover(parent1, parent2)
            
            # Add child to player list
            players.append(child)
        
        # Set population players to new generation
        self.players = players
    
    ### Mutate player genomes
    def mutate(self):
        for player in self.players:
            self.mutate_genome(player)

    ### Separate players into species
    def speciate(self):
        # Reset species
        for species in self.species:
            species.reset_species()
        
        # Determine closest species
        for player in self.players:
            min_distance = float('inf')
            for species in self.species:
                distance = species.genome_distance(player)
                if distance < self.species_threshold:
                    if distance < min_distance:
                        min_distance = distance
                        closest_species = species
            
            # Add new species if no species is close enough
            if min_distance > self.species_threshold:
                closest_species = Species()
                self.species.append(closest_species)

            # Assign player to closest species
            closest_species.add_player(player)
        
        # Adjust threshold distance based on number of species
        if len(self.species) > SPECIES_WANTED:
            self.species_threshold /= DISTANCE_MODIFIER
        elif len(self.species) < SPECIES_WANTED:
            self.species_threshold *= DISTANCE_MODIFIER

    ### Update species based on performance
    def update_species(self):
        # Sort species according to fitness
        self.species.sort(key=lambda k: k.fittest_player.fitness, reverse=True)
        self.max_fitness = self.species[0].fittest_player.fitness

        # Find stale species
        stale_species = []
        for species in self.species:
            if species.staleness >= MAX_STALENESS:
                stale_species.append(species)
        
        # Remove worst-performing stale species
        if len(stale_species) > 0:
            stale_species.sort(key=lambda k: k.fittest_player.fitness)
            for player in stale_species[0].players:
                if player in self.players:
                    self.players.remove(player)
    
    ### Mutate genome
    def mutate_genome(self, player:Organism):
        # Mutate neuron
        if np.random.uniform() < PR_MUTATE_NEURON:
            self.mutate_neuron(player)
        
        # Mutate gene
        if np.random.uniform() < PR_MUTATE_GENE or len(player.genes) == 0:
            self.mutate_gene(player)
        
        # Mutate weights
        if np.random.uniform() < PR_MUTATE_WEIGHTS:
            self.mutate_weights(player)
    
    ### Mutate the genome by adding a new neuron
    def mutate_neuron(self, player:Organism):
        # Find a gene to split
        gene = player.find_split_gene()
        if gene is None:
            return

        # Adjust layers if applicable
        neuron_layer = gene.from_node.layer + 1
        if neuron_layer == gene.to_node.layer:
            player.adjust_layers(neuron_layer)

        # Add a new neuron
        label = self.innovation_history.new_node()
        player.add_neuron(label, neuron_layer)
        neuron = player.get_neuron(label)

        # Make connection between from neuron and new neuron
        label = self.innovation_history.add_innovation(gene.from_node.label, neuron.label)
        player.add_gene(label, gene.from_node, neuron, 1)

        # Make connection between new neuron and to neuron
        label = self.innovation_history.add_innovation(neuron.label, gene.to_node.label)
        player.add_gene(label, neuron, gene.to_node, gene.weight)

        # Make connection between bias neuron and new neuron
        label = self.innovation_history.add_innovation(player.bias.label, neuron.label)
        player.add_gene(label, player.bias, neuron, 0)

    ### Mutate genome by adding a new connection
    def mutate_gene(self, player:Organism):
        # Find a gene to mutate
        gene = player.find_mutate_gene()
        if gene is None:
            return
        else:
            from_neuron, to_neuron = gene
        
        # Make connection between from neuron and to neuron
        label = self.innovation_history.find_innovation(from_neuron.label, to_neuron.label)
        if label == -1:
            label = self.innovation_history.add_innovation(from_neuron.label, to_neuron.label)
        player.add_gene(label, from_neuron, to_neuron, np.random.uniform(-MAX_WEIGHT, MAX_WEIGHT))
    
    ### Mutate genome by modifying weights
    def mutate_weights(self, player:Organism):
        for gene in player.genes:
            gene.mutate_weight()

    ### Crossover between two parent genomes
    def crossover(self, parent1:Organism, parent2:Organism) -> Organism:
        # Create skeleton for child genome
        child = Organism(self.num_inputs, self.num_outputs)

        # Fitter parent is the base
        if parent1.fitness > parent2.fitness:
            base_mate = parent1
            partner_mate = parent2
        else:
            base_mate = parent2
            partner_mate = parent1

        # Use neurons of the base
        for layer_neurons in base_mate.neurons.values():
            for neuron in layer_neurons:
                child.add_neuron(neuron.label, neuron.layer)

        # Crossover for connection genes
        for base_gene in base_mate.genes:
            if base_gene.innovation_label in partner_mate.innovation_labels:
                partner_gene = partner_mate.get_gene(base_gene.innovation_label)

                # Weight inheritance
                if np.random.uniform() < PR_INHERIT_FITTER:
                    weight = base_gene.weight
                else:
                    weight = partner_gene.weight

                # Determine whether the gene is enabled
                if not base_gene.enabled or not partner_gene.enabled:
                    enabled = np.random.uniform() < PR_ENABLE
                else:
                    enabled = True
            else:
                weight = base_gene.weight
            
            from_node = child.get_neuron(base_gene.from_node.label)
            to_node = child.get_neuron(base_gene.to_node.label)
            child.add_gene(base_gene.innovation_label, from_node, to_node, weight, enabled=enabled)
        
        return child
    
    ### Clone from one parent genome
    def clone(self, parent:Organism) -> Organism:
        # Create skeleton for child genome
        child = Organism(self.num_inputs, self.num_outputs)

        # Use neurons of the parent
        for layer, layer_neurons in sorted(parent.neurons.items(), reverse=True):
            if layer != 0 and layer != parent.num_layers - 1:
                for neuron in layer_neurons:
                    child.add_neuron(neuron.label, neuron.layer)

        # Add genes
        for gene in parent.genes:
            from_node = child.get_neuron(gene.from_node.label)
            to_node = child.get_neuron(gene.to_node.label)
            child.add_gene(gene.innovation_label, from_node, to_node, gene.weight, enabled=gene.enabled)
        
        return child
        