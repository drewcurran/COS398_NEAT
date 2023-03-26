import numpy as np

from connection_gene import ConnectionGene

class Node:
    def __init__(self, label, layer=0):
        self.label = label
        self.input_value = 0
        self.output_value = 0
        self.output_connections = None
        self.layer = layer

    # Engage the node by producing output from input
    def get_value(self):
        if self.layer != 0:
            self.output_value = self.sigmoid(self.input_value)
        for i in range(self.output_connections.size()):
            connection = self.output_connections.get(i)
            if connection.enabled:
                connection.to_node.input_value += connection.weight * self.output_value

    # Sigmoid function
    def sigmoid(value):
        return 1 / (1 + np.exp(-value))
    
    # Determine if connected to node
    def connected_to(self, node):
        if node.layer == self.layer:
            return False
        elif node.layer < self.layer:
            for i in range(node.output_connections.size()):
                connection = node.output_connections.get(i)
                if connection.to_node == self:
                    return True
        elif node.layer > self.layer:
            for i in range(self.output_connections.size()):
                connection = self.output_connections.get(i)
                if connection.to_node == node:
                    return True
        return False
                