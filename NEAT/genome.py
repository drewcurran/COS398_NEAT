'''
genome.py
Description: The genome is the blueprint for the neural network.
Author: Drew Curran
'''

import numpy as np
from matplotlib import pyplot as plt

from NEAT.node import Node
from NEAT.connection_gene import ConnectionGene
from NEAT.connection_history import ConnectionHistory
from NEAT.helper_functions import sigmoid

class Genome:
    def __init__(self, num_inputs, num_outputs, num_layers = 2):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.nodes = {}
        self.num_nodes = 0
        self.genes = []
        
        # Add input nodes (including bias node)
        self.nodes[0] = []
        self.add_node(0)
        self.bias_node = self.nodes[0][0]
        self.non_bias_connection = False
        for _ in range(self.num_inputs):
            self.add_node(0)

        # Add output nodes
        self.nodes[self.num_layers - 1] = []
        for _ in range(self.num_outputs):
            self.add_node(self.num_layers - 1)

    ### Get node with matching label
    def get_node(self, label):
        for layer_nodes in self.nodes.values():
            for node in layer_nodes:
                if node.label == label:
                    return node
        return None
    
    ### Add a node to the genome
    def add_node(self, layer):
        node = Node(self.num_nodes, layer=layer)
        self.nodes[layer].append(node)
        self.num_nodes += 1
        return node
    
    ### Add a connection gene to the genome
    def add_connection(self, history, from_node, to_node, weight=1):
        # Only proceed if the connection does not exist
        label = None
        if self.is_connection(from_node, to_node):
            return None

        # Retrieve the genome label if mutation already exists
        for gene in history:
            if gene.matches(self, from_node, to_node):
                label = gene.innovation_label
        
        # Make a new mutation
        gene_labels = []
        for gene in self.genes:
            gene_labels.append(gene.innovation_label)
        if label is None:
            label = len(history)
            mutation = ConnectionHistory(from_node.label, to_node.label, label, gene_labels)
            history.append(mutation)

        # Add connection to genome
        connection = ConnectionGene(from_node, to_node, weight, label)
        self.genes.append(connection)
        if not self.non_bias_connection and connection.from_node != self.bias_node:
            self.non_bias_connection = True

        return connection

    ### Refresh constant values
    def refresh_constants(self):
        # Number of nodes
        self.num_nodes = 0
        for layer_nodes in self.nodes.values():
            self.num_nodes += len(layer_nodes)

        # Number of layers
        self.num_layers = len(self.nodes.keys())

        # Bias node
        self.bias_node = self.nodes[0][0]


    ### Forward pass
    def forward_pass(self, input_values):
        # Input values must be same size and inputs and have bias value 1
        assert len(self.nodes[0]) == len(input_values), "NN Inputs: %d, Values Given: %d" % (len(self.nodes[0]), len(input_values))
        assert input_values[0] == 1

        # Give values to the nodes in the input layer
        for node in self.nodes[0]:
            node.input_value = input_values[node.label]

        # Sort the genes according to from node layer
        self.genes.sort(key=lambda k: k.from_node.layer)

        # Propagate through the network
        for gene in self.genes:
            gene.send_value()
        
        out = []

        for node in self.nodes[self.num_layers - 1]:
            # Get output node values
            out.append(sigmoid(node.input_value))
        
        # Reset node values
        for layer_nodes in self.nodes.values():
            for node in layer_nodes:
                node.input_value = 0
        
        return out

    ### Mutate the genome weights
    def mutate_weights(self):
        for gene in self.genes:
            gene.mutate_weight()

    ### Mutate the genome by adding a new node
    def mutate_node(self, history):
        # Escape if there are only bias connections
        if not self.non_bias_connection:
            return None, None

        # Find a connection without the bias node (bias should not be disconnected)
        connection = self.genes[np.random.randint(len(self.genes))]
        while connection.from_node == self.bias_node:
            connection = self.genes[np.random.randint(len(self.genes))]

        # Disable the connection between the two nodes
        connection.enabled = False

        # Adjust layers to consider the new node
        node_layer = connection.from_node.layer + 1
        if node_layer == connection.to_node.layer:
            for layer, layer_nodes in sorted(self.nodes.items(), reverse=True):
                if layer < connection.to_node.layer:
                    break
                for node in layer_nodes:
                    node.layer += 1
                self.nodes[layer + 1] = self.nodes.pop(layer)
            self.nodes[node_layer] = []
            self.nodes = dict(sorted(self.nodes.items()))

        # List of nodes made
        nodes = []

        # Add a new node with layer one more than the from node
        node = self.add_node(connection.from_node.layer + 1)
        nodes.append(node)

        # List of connections made
        connections = []

        # Make connection between from node and new node
        new_connection = self.add_connection(history, connection.from_node, node, weight=1)
        connections.append(new_connection)

        # Make connection between new node and to node
        new_connection = self.add_connection(history, node, connection.to_node, weight=connection.weight)
        connections.append(new_connection)

        # Make connection between bias node and new node
        new_connection = self.add_connection(history, self.bias_node, node, weight=0)
        connections.append(new_connection)

        # Refresh
        self.refresh_constants()

        return nodes, connections

    ### Mutate the genome by adding a new connection
    def mutate_connection(self, history):
        # Test if fully connected
        nodes_before = 0
        max_connections = 0
        for layer_nodes in self.nodes.values():
            max_connections += nodes_before * len(layer_nodes)
            nodes_before += len(layer_nodes)
        if len(self.genes) == max_connections:
            return None
        
        # Find a new connection
        node1 = self.get_node(np.random.randint(self.num_nodes))
        node2 = self.get_node(np.random.randint(self.num_nodes))
        while self.is_connection(node1, node2, layer_check=True):
            node1 = self.get_node(np.random.randint(self.num_nodes))
            node2 = self.get_node(np.random.randint(self.num_nodes))

        # Designate nodes as from and to nodes
        if node1.layer < node2.layer:
            from_node = node1
            to_node = node2
        else:
            from_node = node2
            to_node = node1
        
        # Make connection between from node and to node
        connection = self.add_connection(history, from_node, to_node, weight=np.random.uniform(-1, 1))

        return connection
    
    ### General mutation for the genome
    def mutate_genome(self, history):
        # List of nodes made
        nodes = []

        # List of connections made
        connections = []

        # Mutate connection
        if np.random.uniform() < 0.05 or len(self.genes) == 0:
            connection = self.mutate_connection(history)
            connections.append(connection)
        
        # Mutate weights
        if np.random.uniform() < 0.8:
            for gene in self.genes:
                gene.mutate_weight()
        
        # Mutate node
        if np.random.uniform() < 0.01:
            node, node_connections = self.mutate_node(history)
            nodes.append(node)
            connections.append(node_connections)

        return nodes, connections

    ### Determine if two nodes are connected
    def is_connection(self, node1, node2, layer_check=False):
        # Cannot be in the same layer
        if node1.layer == node2.layer:
            return layer_check
        
        # Search node1 output connections
        elif node1.layer < node2.layer:
            for gene in self.genes:
                if gene.from_node == node1 and gene.to_node == node2:
                    return True
        # Search node2 output connections
        elif node2.layer < node1.layer:
            for gene in self.genes:
                if gene.from_node == node2 and gene.to_node == node1:
                    return True
        return False

    ### Crossover with another parent genome
    def crossover(self, mate):
        # Clone from self
        child = self.clone()

        # Crossover for connection genes
        for gene in child.genes:
            for mate_gene in mate.genes:
                if mate_gene.innovation_label == gene.innovation_label:
                    # Determine whether the gene is enabled
                    if not gene.enabled or not mate_gene.enabled:
                        enabled = np.random.uniform() < 0.25
                    else:
                        enabled = True

                    # Inheritance from mate
                    if np.random.uniform() < 0.5:
                        weight = gene.weight
                    else:
                        weight = mate_gene.weight
                    
                    gene.modify(weight, enabled)
        
        return child
    
    ### Get number of unmatching genes with another genome
    def genome_difference(self, genome):
        # Count matching genes
        num_matching_genes = 0
        weight_difference = 0
        for gene1 in self.genes:
            for gene2 in genome.genes:
                if gene1.innovation_label == gene2.innovation_label:
                    num_matching_genes += 1
                    weight_difference += abs(gene1.weight - gene2.weight)
                    break

        # Calculate excess and disjoint genes
        num_unmatching_genes = len(self.genes) + len(genome.genes) - 2 * num_matching_genes
        
        # Calculate average weight difference
        if len(self.genes) == 0 or len(genome.genes) == 0:
            average_weight_difference = 0
        elif num_matching_genes == 0:
            average_weight_difference = float('inf')
        else:
            average_weight_difference = weight_difference / num_matching_genes

        return num_unmatching_genes, average_weight_difference


    ### Return a copy
    def clone(self):
        clone = Genome(self.num_inputs, self.num_outputs, num_layers=self.num_layers)
        
        # Copy the nodes
        nodes = {}
        for layer, layer_nodes in self.nodes.items():
            nodes[layer] = []
            for node in layer_nodes:
                nodes[layer].append(node.clone())
        clone.nodes = nodes

        # Copy the connections
        genes = []
        for gene in self.genes:
            genes.append(gene.clone(clone.get_node(gene.from_node.label), clone.get_node(gene.to_node.label)))
        clone.genes = genes

        # Refresh
        clone.refresh_constants()

        return clone
    
    ### Print state of the genome
    def print_state(self, history = [], nodes = [], connections = []):
        print("\tState")
        print("\tNodes: %d (%d), %s" % (self.num_nodes, self.num_layers, self.nodes))
        print("\tConnections: %d, %s" % (len(self.genes), self.genes))
        if history:
            print("\tHistory: %s" % history)
        if nodes or connections:
            print("\n\tMutations")
        if nodes:
            print("\tNodes: %s" % nodes)
        if connections:
            print("\tConnections: %s" % connections)
        print("\n")

    ### Draw the state of the genome
    def draw_state(self, numbers=True):
        _, ax = plt.subplots()

        # Draw nodes
        for layer_nodes in self.nodes.values():
            for i in range(len(layer_nodes)):
                node = layer_nodes[i]
                node.draw_location = (len(layer_nodes) - 1) / 2 - i
                circle = plt.Circle((node.layer, node.draw_location), radius=0.1, label=node.label, color="black", zorder=2)
                plt.gca().add_patch(circle)
                if numbers:
                    plt.text(node.layer, node.draw_location, node.label, fontsize=12, ha="center", va="center", color="white", zorder=3)

        # Draw genes
        for gene in self.genes:
            if gene.enabled:
                line = plt.Line2D((gene.from_node.layer, gene.to_node.layer), (gene.from_node.draw_location, gene.to_node.draw_location), color="red" if gene.weight > 0 else "blue", alpha = abs(gene.weight), zorder=1)
                plt.gca().add_line(line)
        
        # Config
        plt.axis('scaled')
        plt.axis('off')
