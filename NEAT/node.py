import numpy as np

from connection_gene import ConnectionGene
from helper_functions import sigmoid

class Node:
    def __init__(self, label, layer=0):
        self.label = label
        self.input_value = 0
        self.output_value = 0
        self.output_connections = None
        self.layer = layer

    ### Set the output value sent across connections
    def set_output_value(self, bias = False):
        # Use sigmoid activation function only if node is not an input
        if self.layer != 0:
            self.output_value = sigmoid(self.input_value)
        if bias:
            self.output_value = 1

    ### Engage the node by producing output from input
    def send_value(self):
        # Add linear combination of weights and inputs
        for i in range(self.output_connections.size()):
            connection = self.output_connections.get(i)
            if connection.enabled:
                connection.to_node.input_value += connection.weight * self.output_value

    ### Determine if connected to node
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
                