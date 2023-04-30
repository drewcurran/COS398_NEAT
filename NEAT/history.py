'''
history.py
Description: Innovation marker to track genetic history.
Author: Drew Curran
'''

from NEAT.node import Node

class InnovationHistory:
    def __init__(self, node_label:int):
        self.innovations = []
        self.node_label = node_label
    
    ### Register a new node
    def new_node(self):
        label = self.node_label
        self.node_label += 1
        return label

    ### Add an innovation to the history
    def add_innovation(self, from_node_label:int, to_node_label:int) -> int:
        self.innovations.append((from_node_label, to_node_label))
        return len(self.innovations)
    
    ### Find same connection mutation
    def find_innovation(self, from_node_label:int, to_node_label:int) -> int:
        for label in range(len(self.innovations)):
            mutation_from, mutation_to = self.innovations[label]
            if from_node_label == mutation_from and to_node_label == mutation_to:
                return label
        return -1