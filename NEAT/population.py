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
from NEAT.parameters import GENETIC_DISTANCE, SPECIES_WANTED, SPECIES_BUFFER, GENERATION_GRACE, DISTANCE_MODIFIER
from NEAT.parameters import POPULATION_CULL_RATE, MAX_STALENESS

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
        self.cull()
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
                parent = np.random.choice(self.players)
                child = self.clone(parent)

            # Crossover between two parents
            else:
                if np.random.uniform() < PR_INTERSPECIES:
                    parent1 = np.random.choice(self.players)
                    parent2 = np.random.choice(self.players)
                else:
                    avg_fitness_sum = np.sum(list(map(lambda s: s.average_fitness, self.species)))
                    species = np.random.choice(self.species, p=list(map(lambda s: s.average_fitness / avg_fitness_sum, self.species)))
                    parent1 = np.random.choice(species.players)
                    parent2 = np.random.choice(species.players)
                child = self.crossover(parent1, parent2)
            
            # Add child to player list
            players.append(child)
        
        # Set population players to new generation
        self.players = players
    
    ### Mutate player genomes
    def mutate(self):
        for i in range(len(self.players)):
            player = self.players[i]
            if i >= len(self.species):
                self.mutate_genome(player)

    ### Separate players into species
    def speciate(self):
        # Reset species
        for species in self.species:
            species.reset_species()
        
        # Add each player to a close species
        for i in range(len(self.players)):
            player = self.players[i]
            if i >= len(self.species):
                self.find_species(player).add_player(player)

        # If species only contains one player, attempt to find a different species
        for species in self.species[:]:
            if len(species.players) == 1:
                player = species.players[0]
                self.species.remove(species)
                self.find_species(player).add_player(player)
        
        # Adjust threshold distance based on number of species
        if self.generation > GENERATION_GRACE:
            if len(self.species) > SPECIES_WANTED + SPECIES_BUFFER:
                self.species_threshold += GENETIC_DISTANCE * DISTANCE_MODIFIER
            elif len(self.species) < SPECIES_WANTED - SPECIES_BUFFER:
                self.species_threshold -= GENETIC_DISTANCE * DISTANCE_MODIFIER

    def cull(self):
        # Cull population lightly
        self.players.sort(key=lambda k: k.fitness, reverse=True)
        desired_size = int((len(self.players) * (1 - POPULATION_CULL_RATE)))
        self.players = self.players[0:desired_size+1]

        # Cull within species
        for species in self.species:
            culled_players = species.cull()
            for player in culled_players:
                if player in self.players:
                    self.players.remove(player)
    
    ### Update species based on performance
    def update_species(self):
        # Update properties of species
        for species in self.species:
            species.update()

        # Sort species according to max fitness
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
        # Fitter parent is the base
        if parent1.fitness > parent2.fitness:
            base_mate = parent1
            partner_mate = parent2
        else:
            base_mate = parent2
            partner_mate = parent1

        # Create skeleton for child genome
        child = Organism(base_mate.num_inputs, base_mate.num_outputs, num_layers=base_mate.num_layers)

        # Use neurons of the parent
        for layer, layer_neurons in sorted(base_mate.neurons.items(), reverse=True):
            if layer != 0 and layer != base_mate.num_layers - 1:
                for neuron in layer_neurons:
                    child.add_neuron(neuron.label, neuron.layer)

        # Crossover for connection genes
        for base_gene in base_mate.genes:
            # Shared innovation
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
            
            # Excess or disjoint innovation
            else:
                weight = base_gene.weight
                enabled = True
            
            from_node = child.get_neuron(base_gene.from_node.label)
            to_node = child.get_neuron(base_gene.to_node.label)
            child.add_gene(base_gene.innovation_label, from_node, to_node, weight, enabled=enabled)
        
        return child
    
    ### Clone from one parent genome
    def clone(self, parent:Organism) -> Organism:
        # Create skeleton for child genome
        child = Organism(parent.num_inputs, parent.num_outputs, num_layers=parent.num_layers)

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
    
    ### Find close species for a player
    def find_species(self, player:Organism) -> Species:
        # Find close species
        for species in self.species:
            if species.genome_distance(player) < self.species_threshold:
                return species
        
        # Add new species if no species is close enough
        species = Species()
        self.species.append(species)
        return species
