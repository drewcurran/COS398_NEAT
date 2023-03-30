'''
genome.py
Description: The genome is the blueprint for the neural network.
Author: Drew Curran
'''

from node import Node
from connection_gene import ConnectionGene

class Genome:
    def __init__(self, num_inputs, num_outputs, num_layers=2):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.genes = []
        self.nodes = {}
        self.num_nodes = 0

        # Add input nodes (including bias node)
        self.nodes[0] = []
        self.add_node(0)
        for _ in range(self.num_inputs):
            self.add_node(0)

        # Add output nodes
        self.nodes[self.num_layers - 1] = []
        for _ in range(self.outputs):
            self.add_node(self.num_layers - 1)

    ### Get node with matching label
    def get_node(self, label):
        for i in range(len(self.nodes)):
            if self.nodes[i].label == label:
                return self.nodes[i]
        return None
    
    ### Add a node to the genome
    def add_node(self, layer):
        node = Node(self.num_nodes, layer = layer)
        self.nodes[layer].append(node)
        self.num_nodes += 1
    
    ### Connects nodes
    def connect_nodes(self):
        # Delete output connections for all nodes
        for _, layer_nodes in self.nodes:
            for node in layer_nodes:
                node.output_connections = []

        # Add connection for every connection gene
        for connection_gene in self.genes:
            connection_gene.from_node.output_connections.append(connection_gene)

    ### Forward pass
    def forward_pass(self, input_values):
        # Set bias node to 1
        self.nodes[self.layer_indices[0]].output_value = 1

        # Set input node values to given values
        for i in range(self.inputs):
            self.inputs[self.layer_indices[1] + i].output_value = input_values[i]

        # Send value across connections for all the nodes in the genome by layer
        for node in self.nodes:
            node.send_value()
        
        # Get output node values
        out = []
        for _ in range(self.outputs):
            out[i] = self.nodes[self.inputs + i].output_value
        
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



        
        