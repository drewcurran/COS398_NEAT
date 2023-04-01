'''
genome.py
Description: The genome is the blueprint for the neural network.
Author: Drew Curran
'''

import numpy as np

from node import Node
from connection_gene import ConnectionGene
from connection_history import ConnectionHistory

class Genome:
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = 2
        self.nodes = {}
        self.num_nodes = 0
        self.genes = []
        self.num_genes = 0
        
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
        if self.is_connection(from_node, to_node):
            return None
        
        # Retrieve the genome label if mutation already exists
        for gene in history:
            if gene.matches(self, from_node, to_node):
                return gene.innovation_label
        
        # Make a new mutation
        gene_labels = []
        for gene in self.genes:
            gene_labels.append(gene.innovation_label)
        mutation = ConnectionHistory(from_node.label, to_node.label, self.num_genes, gene_labels)
        history.append(mutation)

        # Add connection to genome
        connection = ConnectionGene(from_node, to_node, weight, self.num_genes)
        self.genes.append(connection)
        self.num_genes += 1
        if not self.non_bias_connection and connection.from_node != self.bias_node:
            self.non_bias_connection = True

        return connection, history

    ### Refresh constant values
    def refresh_constants(self):
        # Number of nodes
        self.num_nodes = 0
        for layer_nodes in self.nodes.values():
            self.num_nodes += len(layer_nodes)

        # Number of connections
        self.num_genes = len(self.genes)

        # Number of layers
        self.num_layers = len(self.nodes.keys())

        # Bias node
        self.bias_node = self.nodes[0][0]


    ### Forward pass
    def forward_pass(self, input_values):
        # Input values must be same size and inputs and have bias value 1
        assert len(self.nodes[0]) == len(input_values)
        assert input_values[0] == 1

        # Give values to the nodes in the input layer
        for node in self.nodes[0]:
            node.input_value = input_values[node.label]

        # Sort the genes according to from node layer
        self.genes.sort(self.genes, key=lambda k: k.from_node.layer)

        # Propagate through the network
        for gene in self.genes:
            gene.send_value()
        
        out = []

        for node in self.nodes[self.num_layers - 1]:
            # Get output node values
            out.append(node.get_output_value())
        
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
            return None

        # Find a connection without the bias node (bias should not be disconnected)
        connection = self.genes[np.random.randint(len(self.genes))]
        while connection.from_node == self.bias_node:
            connection = self.genes[np.random.randint(len(self.genes))]

        # Disable the connection between the two nodes
        connection.enabled = False

        # Add a new node with layer one more than the from node
        new_node = self.add_node(connection.from_node.layer + 1)

        # Make connection between from node and new node
        self.add_connection(history, connection.from_node, new_node, weight=1)

        # Make connection between new node and to node
        self.add_connection(history, new_node, connection.to_node, weight=connection.weight)

        # Make connection between bias node and new node
        self.add_connection(history, self.bias_node, new_node, weight=0)

        # Adjust layers to consider the new node
        if new_node.layer == connection.to_node.layer:
            self.nodes[new_node.layer].remove(new_node)
            for layer, layer_nodes in sorted(self.nodes.items(), reverse=True):
                if layer < new_node.layer:
                    break
                for node in layer_nodes:
                    node.layer += 1
                self.nodes[layer + 1] = self.nodes.pop(layer)
            self.nodes[new_node.layer] = []
            self.nodes[new_node.layer].append(new_node)
            

        # Refresh
        self.refresh_constants()

        return new_node

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
        connection, history = self.add_connection(history, from_node, to_node, weight=np.random.uniform(-1, 1))

        return connection, history
    
    ### General mutation for the genome
    def mutate_genome(self, history):
        # Mutate connection
        connection = None
        if np.random.uniform() < 0.05 or len(self.genes) == 0:
            connection, history = self.mutate_connection(history)
        
        # Mutate weights
        if np.random.uniform() < 0.8:
            for gene in self.genes:
                gene.mutate_weight()
        
        # Mutate node
        node = None
        if np.random.uniform() < 0.01:
            node, history = self.mutate_node(history)

        return connection, history, node

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
                    # Disable gene
                    enabled = True
                    if not gene.enabled or not mate_gene.enabled:
                        enabled = np.random.uniform() < 0.75

                    # Inheritance from mate
                    if np.random.uniform() < 0.5:
                        gene = mate_gene.copy()

                    gene.enabled = enabled
    
    ### Return a copy
    def clone(self):
        clone = Genome(self.num_inputs, self.num_outputs, num_layers=self.num_layers)
        
        # Copy the nodes
        nodes = {}
        for layer, layer_nodes in self.nodes.items():
            for node in layer_nodes:
                nodes[layer].append(node.clone())
        clone.nodes = nodes

        # Copy the connections
        genes = []
        for gene in self.genes:
            genes.append(gene.clone())
        clone.genes = genes

        # Refresh
        clone.refresh_constants()

        return clone
    
def main():
    history = []

    print("Initializing genome")
    genome = Genome(2, 3)
    print("\tNodes: %d, %s" % (genome.num_nodes, genome.nodes))
    print("\tConnections: %d, %s" % (genome.num_genes, genome.genes))
    print("\n")

    print("General mutation")
    connection, history, node = genome.mutate_genome(history)
    print("\tNodes: %d, %s" % (genome.num_nodes, genome.nodes))
    print("\tConnections: %d, %s" % (genome.num_genes, genome.genes))
    print()
    print("\tConnection: %s" % connection)
    print("\tHistory: %s" % history)
    print("\tNode: %s" % node)
    print("\n")    

    print("Node mutation")
    node = genome.mutate_node(history)
    print("\tNodes: %d, %s" % (genome.num_nodes, genome.nodes))
    print("\tConnections: %d, %s" % (genome.num_genes, genome.genes))
    print()
    print("\tNode: %s" % node)
    print("\n")

    print("Gene mutation")
    connection, history = genome.mutate_connection(history)
    print("\tNodes: %d, %s" % (genome.num_nodes, genome.nodes))
    print("\tConnections: %d, %s" % (genome.num_genes, genome.genes))
    print()
    print("\tConnection: %s" % connection)
    print("\tMutation: %s" % history)
    print("\n")

    print("Node mutation")
    node = genome.mutate_node(history)
    print("\tNodes: %d, %s" % (genome.num_nodes, genome.nodes))
    print("\tConnections: %d, %s" % (genome.num_genes, genome.genes))
    print()
    print("\tNode: %s" % node)
    print("\n")

if __name__ == '__main__':
    main()