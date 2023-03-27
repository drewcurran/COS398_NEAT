import numpy as np

from node import Node

class Genome:
    def __init__(self, inputs, outputs, layers=2):
        self.inputs = inputs
        self.outputs = outputs
        self.layers = layers

        self.next_node = 0
        self.nodes = []
        for i in range(self.inputs):
            node = Node(i)
            self.nodes.append(node)
            node.layer = 0
            self.next_node += 1

        for o in range(self.outputs):
            node = Node(o + self.inputs)
            self.nodes.append(node)
            node.layer = 1
            self.next_node += 1
        
        self.bias_node = Node(self.next_node)
        
        self.genes = None
        
        
        
        self.network = None

    # Get node with matching label
    def get_node(self, label):
        for i in range(self.nodes.size()):
            if self.nodes.get(i).label == label:
                return self.nodes.get(i)
        return None
    
    # Connects nodes
    def connect_nodes(self):
        for node in self.nodes:
            node.output_connections = None
        for connection_gene in self.genes:
            connection_gene.from_node.output_connections.add(connection_gene)

    # Forward pass
    def forward_pass(self, input_values):
        for i in range(self.inputs):
            self.inputs[i].output_value = input_values[i]
        self.nodes[self.bias_node].output_value = 1

        for node in self.network:
            node.get_value()
        
        out = []
        for i in range(self.outputs):
            out[i] = self.nodes.get(self.inputs + i)
        
        for node in self.nodes:
            node.input_value = 0
        
        return out
    
    # Set up neural network
    def generate_network(self):
        self.connect_nodes()
        self.network = []
        for l in range(self.layers):
            for node in self.nodes:
                if (node.layer == l):
                    self.network.append(node)

    # Mutate the genome
    def mutate(self):
        if np.random.uniform() < 0.8:
            for gene in self.genes:
                gene.mutate_weight()


    # Add a new node

    # Add a new connection



        
        