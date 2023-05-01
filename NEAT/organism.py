'''
organism.py
Description: An individual with a genome and fitness.
Author: Drew Curran
'''

from __future__ import annotations
import numpy as np
from matplotlib import pyplot as plt

from NEAT.node import Node
from NEAT.connection import Connection
from NEAT.history import InnovationMarker
from NEAT.parameters import MAX_WEIGHT
from NEAT.parameters import PR_INHERIT_FITTER, PR_ENABLE
from NEAT.parameters import PR_MUTATE_NEURON, PR_MUTATE_GENE, PR_MUTATE_WEIGHTS
from NEAT.parameters import SIGMOID_SCALE

class Organism:
    def __init__(self, num_inputs:int, num_outputs:int, num_layers:int=2):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.initialize_genome()

    ### Initialize the genome
    def initialize_genome(self):
        self.neurons = {}
        self.genes = []
        self.innovation_labels = []
        self.num_neurons = 0
        
        # Add input neurons
        self.neurons[0] = []
        for _ in range(self.num_inputs + 1):
            self.add_neuron(self.num_neurons, 0)
        self.bias = self.neurons[0][0]

        # Add output neurons
        self.neurons[self.num_layers - 1] = []
        for _ in range(self.num_outputs):
            self.add_neuron(self.num_neurons, self.num_layers - 1)
        self.neurons = dict(sorted(self.neurons.items()))

    ### Crossover with another parent
    def crossover(self, innovations: list[InnovationMarker], partner: Organism) -> Organism:
        # Create skeleton for child genome
        child = Organism(self.num_inputs, self.num_outputs, num_layers=self.num_layers)

        # Use neurons of the parent
        for layer, layer_neurons in sorted(self.neurons.items(), reverse=True):
            if layer != 0 and layer != self.num_layers - 1:
                for neuron in layer_neurons:
                    child.add_neuron(neuron.label, neuron.layer)

        # Crossover for connection genes
        for gene in self.genes:
            label = gene.label
            from_node = child.get_neuron(gene.from_node.label)
            to_node = child.get_neuron(gene.to_node.label)

            partner_gene = partner.get_gene(label)

            # Shared innovation
            if partner_gene is not None:
                # Weight inheritance
                if np.random.uniform() < PR_INHERIT_FITTER:
                    weight = gene.weight
                else:
                    weight = partner_gene.weight

                # Determine whether the gene is enabled
                if not gene.enabled or not partner_gene.enabled:
                    enabled = np.random.uniform() < PR_ENABLE
                else:
                    enabled = True
            
            # Excess or disjoint innovation
            else:
                weight = gene.weight
                enabled = True
            
            child.add_gene(innovations, from_node, to_node, weight, enabled=enabled)
        
        return child
    
    def clone(self, innovations: list[InnovationMarker]) -> Organism:
        # Create skeleton for child genome
        child = Organism(self.num_inputs, self.num_outputs, num_layers=self.num_layers)

        # Use neurons of the parent
        for layer, layer_neurons in sorted(self.neurons.items(), reverse=True):
            if layer != 0 and layer != self.num_layers - 1:
                for neuron in layer_neurons:
                    child.add_neuron(neuron.label, neuron.layer)

        # Add genes
        for gene in self.genes:
            from_node = child.get_neuron(gene.from_node.label)
            to_node = child.get_neuron(gene.to_node.label)
            child.add_gene(innovations, from_node, to_node, gene.weight, enabled=gene.enabled)
        
        return child
    
    ### Forward pass through the network
    def output(self, input_values:list[float]) -> list[float]:
        # Reset neuron values
        for layer_neurons in self.neurons.values():
            for neuron in layer_neurons:
                neuron.input_value = 0

        # Give values to the neurons in the input layer
        for neuron in self.neurons[0]:
            neuron.input_value = input_values[neuron.label]

        # Sort the genes according to from neuron layer
        self.genes.sort(key=lambda k: k.from_node.layer)

        # Propagate through the network
        for gene in self.genes:
            gene.send_value()
        
        # Retrieve output values
        out = []
        for neuron in self.neurons[self.num_layers - 1]:
            sigmoid = 1 / (1 + np.exp(-SIGMOID_SCALE * neuron.input_value))
            out.append(sigmoid)
        
        return out
    
    ### Mutate genome
    def mutate_genome(self, innovations: list[InnovationMarker]):
        # Mutate neuron
        if np.random.uniform() < PR_MUTATE_NEURON:
            self.mutate_neuron(innovations)
        
        # Mutate gene
        if np.random.uniform() < PR_MUTATE_GENE or len(self.genes) == 0:
            self.mutate_gene(innovations)
        
        # Mutate weights
        if np.random.uniform() < PR_MUTATE_WEIGHTS:
            self.mutate_weights()
    
    ### Mutate the genome by adding a new neuron
    def mutate_neuron(self, innovations: list[InnovationMarker]):
        # Find a gene to split
        gene = self.find_split_gene()
        if gene is None:
            return

        # Adjust layers if applicable
        neuron_layer = gene.from_node.layer + 1
        if neuron_layer == gene.to_node.layer:
            self.adjust_layers(neuron_layer)

        # Add a new neuron
        neuron = self.add_neuron(self.num_neurons, neuron_layer)

        # Make connection between from neuron and new neuron
        self.add_gene(innovations, gene.from_node, neuron, 1)

        # Make connection between new neuron and to neuron
        self.add_gene(innovations, neuron, gene.to_node, gene.weight)

        # Make connection between bias neuron and new neuron
        self.add_gene(innovations, self.bias, neuron, 0)

    ### Mutate genome by adding a new connection
    def mutate_gene(self, innovations: list[InnovationMarker]):
        # Find a gene to mutate
        gene = self.find_mutate_gene()
        if gene is None:
            return
        else:
            from_neuron, to_neuron = gene
        
        # Make connection between from neuron and to neuron
        self.add_gene(innovations, from_neuron, to_neuron, np.random.uniform(-MAX_WEIGHT, MAX_WEIGHT))
    
    ### Mutate genome by modifying weights
    def mutate_weights(self):
        for gene in self.genes:
            gene.mutate_weight()
    
    ### Find a gene to split
    def find_split_gene(self) -> tuple[Node, Node] | None:
        # Escape if there are only bias connections
        all_bias = True
        for gene in self.genes:
            if gene.from_node != self.bias:
                all_bias = False
                break
        if all_bias == True:
            return None

        # Find a connection without the bias node (bias should not be disconnected)
        gene = np.random.choice(self.genes)
        while gene.from_node == self.bias:
            gene = np.random.choice(self.genes)
        
        # Disable the connection
        gene.enabled = False
        
        return gene

    ### Find a new gene to mutate
    def find_mutate_gene(self) -> tuple[Node, Node] | None:
        # Escape if fully connected
        nodes_before = 0
        max_genes = 0
        for layer_nodes in self.neurons.values():
            max_genes += nodes_before * len(layer_nodes)
            nodes_before += len(layer_nodes)
        if len(self.genes) == max_genes:
            return None
        
        # Find a new connection
        list_neurons = []
        for layer_neurons in self.neurons.values():
            list_neurons.extend(layer_neurons)
        neuron1 = np.random.choice(list_neurons)
        neuron2 = np.random.choice(list_neurons)
        while neuron1.layer == neuron2.layer or self.is_gene(neuron1, neuron2):
            neuron1 = np.random.choice(list_neurons)
            neuron2 = np.random.choice(list_neurons)
        
        # Set from and to neurons based on layers
        if neuron1.layer < neuron2.layer:
            return neuron1, neuron2
        else:
            return neuron2, neuron1
    
    ### Adjust layers for new neuron
    def adjust_layers(self, neuron_layer):
        # Shift layers
        for layer, layer_neurons in sorted(self.neurons.items(), reverse=True):
            if layer < neuron_layer:
                break
            for neuron in layer_neurons:
                neuron.layer += 1
            self.neurons[layer + 1] = self.neurons.pop(layer)

        # Add new layer
        self.neurons[neuron_layer] = []

        # Sort layers
        self.neurons = dict(sorted(self.neurons.items()))
        self.num_layers += 1
    
    ### Add a neuron to the genome
    def add_neuron(self, label:int, layer:int):
        if layer not in self.neurons:
            self.neurons[layer] = []
        neuron = Node(label, layer)
        self.neurons[layer].append(neuron)
        self.num_neurons += 1
        return neuron
    
    ### Add a gene to the genome
    def add_gene(self, innovations: list[InnovationMarker], from_neuron:Node, to_neuron:Node, weight:float, enabled:bool=True):
        # Find mutation if it exists
        for mutation in innovations:
            if mutation.matches(self, from_neuron.label, to_neuron.label):
                self.innovation_labels.append(mutation.innovation_label)
                self.genes.append(Connection(mutation.innovation_label, from_neuron, to_neuron, weight))
                return
        
        # Add to history if mutation doesn't exist already
        innovations.append(InnovationMarker(len(innovations), from_neuron.label, to_neuron.label, self.innovation_labels))
        self.innovation_labels.append(len(innovations))
        gene = Connection(len(innovations), from_neuron, to_neuron, weight, enabled=enabled)
        self.genes.append(gene)
    
    ### Determine if two neurons are connected
    def is_gene(self, neuron1:Node, neuron2:Node) -> bool:
        # Search neuron1 output genes
        if neuron1.layer < neuron2.layer:
            for gene in self.genes:
                if gene.from_node == neuron1 and gene.to_node == neuron2:
                    return True
        # Search neuron2 output genes
        elif neuron2.layer < neuron1.layer:
            for gene in self.genes:
                if gene.from_node == neuron2 and gene.to_node == neuron1:
                    return True
        return False
    
    ### Give fitness value to organism
    def set_fitness(self, fitness:float):
        self.fitness = fitness

    ### Get neuron with matching label
    def get_neuron(self, label:int) -> Node:
        for layer_neurons in self.neurons.values():
            for neuron in layer_neurons:
                if neuron.label == label:
                    return neuron
        return None
    
    ### Get gene with matching neurons
    def get_gene(self, label:int):
        for gene in self.genes:
            if label == gene.label:
                return gene
        return None

    ### Print state of the genome
    def print_state(self):
        print("State\nNeurons: %d (%d), %s\nConnections: %d, %s\n" % (self.num_neurons, self.num_layers, self.neurons, len(self.genes), self.genes))

    ### Draw the state of the genome
    def draw_state(self, numbers:bool=True):
        # Draw neurons
        for layer_neurons in self.neurons.values():
            for i in range(len(layer_neurons)):
                neuron = layer_neurons[i]
                neuron.draw_location = (len(layer_neurons) - 1) / 2 - i
                circle = plt.Circle((neuron.layer, neuron.draw_location), radius=0.1, label=neuron.label, color="black", zorder=2)
                plt.gca().add_patch(circle)
                if numbers:
                    plt.text(neuron.layer, neuron.draw_location, neuron.label, fontsize=12, ha="center", va="center", color="white", zorder=3)

        # Draw genes
        for gene in self.genes:
            if gene.enabled:
                line = plt.Line2D((gene.from_node.layer, gene.to_node.layer), (gene.from_node.draw_location, gene.to_node.draw_location), color="red" if gene.weight > 0 else "blue", alpha = abs(gene.weight), zorder=1)
                plt.gca().add_line(line)
        
        # Config
        plt.axis('scaled')
        plt.axis('off')
        plt.show()
