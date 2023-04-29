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
from NEAT.parameters import GENETIC_DISTANCE
from NEAT.parameters import MAX_STALENESS

class Population:
    def __init__(self, num_players:int, num_inputs:int, num_outputs:int):
        self.population_size = num_players
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.generation = 0
        self.players = []
        self.species = []

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

        self.species.sort(key=lambda k: k.fittest_player.fitness, reverse=True)
        self.max_fitness = self.species[0].fittest_player.fitness

        stale_species = []
        for species in self.species:
            if self.generation - species.fittest_player.generation >= MAX_STALENESS:
                stale_species.append(species)
        if len(stale_species) > 0:
            for player in stale_species[0].players:
                if player in self.players:
                    self.players.remove(player)

    ### Initialize the population
    def initialize_population(self) -> list[Organism]:
        self.innovation_history = InnovationHistory(self.num_inputs + self.num_outputs + 2)
        self.players = [Organism(self.num_inputs, self.num_outputs, self.generation) for _ in range(self.population_size)]

    ### Reproduce using surviving players
    def reproduce(self) -> list[Organism]:
        players = []
        for species in self.species:
            fittest_player = species.fittest_player
            species.players = [fittest_player]
            players.append(fittest_player)
        while len(players) < self.population_size:
            if np.random.uniform() < PR_CLONE:
                parent = self.players[np.random.randint(len(self.players))]
                child = self.clone(parent)
            else:
                if np.random.uniform() < PR_INTERSPECIES:
                    parent1 = self.players[np.random.randint(len(self.players))]
                    parent2 = self.players[np.random.randint(len(self.players))]
                else:
                    species = self.species[np.random.randint(len(self.species))]
                    parent1 = species.players[np.random.randint(len(species.players))]
                    parent2 = species.players[np.random.randint(len(species.players))]
                child = self.crossover(parent1, parent2)
            players.append(child)
        self.players = players
    
    ### Mutate player genomes
    def mutate(self):
        for player in self.players:
            # Mutate neuron
            if np.random.uniform() < PR_MUTATE_NEURON:
                self.mutate_neuron(player)
            
            # Mutate connection
            if np.random.uniform() < PR_MUTATE_GENE or len(player.genes) == 0:
                self.mutate_connection(player)
            
            # Mutate weights
            if np.random.uniform() < PR_MUTATE_WEIGHTS:
                self.mutate_weights(player)

    ### Separate players into species
    def speciate(self):
        # Determine species match
        for player in self.players:
            new_species = True
            for species in self.species:
                distance = species.genome_distance(player)
                if distance < GENETIC_DISTANCE:
                    species.add_player(player)
                    new_species = False
                    break
            if new_species:
                # Add new species if none match
                self.species.append(Species(player))
    
    ### Mutate the genome by adding a new neuron
    def mutate_neuron(self, player:Organism):
        # Find a connection without the bias neuron (bias should not be disconnected)
        if len(player.genes) == 0:
            return
        gene = player.genes[np.random.randint(len(player.genes))]
        if gene.from_node == player.bias:
            return
        gene.enabled = False

        # Adjust neuron layers
        neuron_layer = gene.from_node.layer + 1
        if neuron_layer == gene.to_node.layer:
            for layer, layer_neurons in sorted(player.neurons.items(), reverse=True):
                if layer < gene.to_node.layer:
                    break
                for neuron in layer_neurons:
                    neuron.layer += 1
                player.neurons[layer + 1] = player.neurons.pop(layer)
            player.neurons[neuron_layer] = []
            player.neurons = dict(sorted(player.neurons.items()))
            player.num_layers += 1

        # Add a new neuron
        label = self.innovation_history.new_node()
        player.add_neuron(label, neuron_layer)
        neuron = player.get_neuron(label)

        # Make connection between from neuron and new neuron
        label = self.innovation_history.add_innovation(gene.from_node, neuron)
        player.add_gene(label, gene.from_node, neuron, 1)

        # Make connection between new neuron and to neuron
        label = self.innovation_history.add_innovation(neuron, gene.to_node)
        player.add_gene(label, neuron, gene.to_node, gene.weight)

        # Make connection between bias neuron and new neuron
        label = self.innovation_history.add_innovation(player.bias, neuron)
        player.add_gene(label, player.bias, neuron, 0)

    ### Mutate genome by adding a new connection
    def mutate_connection(self, player:Organism):
        # Find a new connection
        neuron_list = []
        for neurons in player.neurons.values():
            neuron_list.extend(neurons)
        neuron1 = np.random.choice(neuron_list)
        neuron2 = np.random.choice(neuron_list)
        if neuron1.layer == neuron2.layer or player.is_gene(neuron1, neuron2):
            return
        if neuron1.layer < neuron2.layer:
            from_neuron = neuron1
            to_neuron = neuron2
        else:
            from_neuron = neuron2
            to_neuron = neuron1
        
        # Make connection between from neuron and to neuron
        label = self.innovation_history.find_innovation(from_neuron, to_neuron)
        if label == -1:
            label = self.innovation_history.add_innovation(from_neuron, to_neuron)
        player.add_gene(label, from_neuron, to_neuron, np.random.uniform(-MAX_WEIGHT, MAX_WEIGHT))
    
    ### Mutate genome by modifying weights
    def mutate_weights(self, player:Organism):
        for gene in player.genes:
            gene.mutate_weight()

    ### Crossover between two parent genomes
    def crossover(self, parent1:Organism, parent2:Organism) -> Organism:
        # Create skeleton for child genome
        child = Organism(self.num_inputs, self.num_outputs, self.generation)

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
                
                child.add_gene(base_gene.innovation_label, base_gene.from_node, base_gene.to_node, weight, enabled=enabled)
            else:
                child.add_gene(base_gene.innovation_label, base_gene.from_node, base_gene.to_node, base_gene.weight, enabled=base_gene.enabled)
        
        return child
    
    ### Clone from one parent genome
    def clone(self, parent:Organism) -> Organism:
        # Create skeleton for child genome
        child = Organism(self.num_inputs, self.num_outputs, self.generation)

        # Use neurons of the parent
        for layer_neurons in parent.neurons.values():
            for neuron in layer_neurons:
                child.add_neuron(neuron.label, neuron.layer)

        # Add genes
        for gene in parent.genes:
            child.add_gene(gene.innovation_label, gene.from_node, gene.to_node, gene.weight, enabled=gene.enabled)
        
        return child
        