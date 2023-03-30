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
    
    ### Engage the node by producing output from input
    def send_value(self, value = None):
        # If input, then set to given input value
        if self.layer == 0:
            self.output_value = value
        # If not an input, use sigmoid activation function on linear input from previous layer
        else:
            self.output_value = sigmoid(self.input_value)
        
        # For each connection, add output times respective weight to the respective node input
        for connection in self.output_connections:
            if connection.enabled:
                connection.to_node.input_value += connection.weight * self.output_value

    ### Determine if connected to node
    def is_connected_to(self, node):
        # Cannot be in the same layer
        if node.layer == self.layer:
            return False
        # Search node output connections
        elif node.layer < self.layer:
            for connection in node.output_connections:
                if connection.to_node == self:
                    return True
        # Search self output connections
        elif node.layer > self.layer:
            for connection in self.output_connections:
                if connection.to_node == node:
                    return True
        return False

    ### Return a copy
    def clone(self):
        clone = Node(self.label, layer = self.layer)
        return clone
                