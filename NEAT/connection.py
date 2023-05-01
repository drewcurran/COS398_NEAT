'''
connection.py
Description: Links two nodes as input and output due to a weight.
Author: Drew Curran
'''

import numpy as np

from NEAT.node import Node
from NEAT.parameters import MAX_WEIGHT
from NEAT.parameters import PR_WEIGHT_RANDOM, WEIGHT_PERTURB

class Connection:
    def __init__(self, label:int, from_node:Node, to_node:Node, weight:float, enabled:bool=True):
      self.label = label
      self.from_node = from_node
      self.to_node = to_node
      self.weight = weight
      self.enabled = enabled
    
    ### Change the weight
    def mutate_weight(self):
        # Mutate weight randomly
        if np.random.uniform() < PR_WEIGHT_RANDOM:
            self.weight = np.random.uniform(-MAX_WEIGHT, MAX_WEIGHT)
        # Slight change if not mutated
        else:
            self.weight += np.random.normal(0, WEIGHT_PERTURB)
        # Keep within bounds
        if self.weight > MAX_WEIGHT:
           self.weight = MAX_WEIGHT
        elif self.weight < -MAX_WEIGHT:
           self.weight = -MAX_WEIGHT

    ### Send output of a node to input of a second node
    def send_value(self):
        # For each connection, add output times respective weight to the respective node input
        if self.enabled:
            if self.from_node.layer == 0:
                self.to_node.input_value += self.from_node.input_value * self.weight
            else:
                sigmoid = 1 / (1 + np.exp(-self.from_node.input_value))
                self.to_node.input_value += sigmoid * self.weight

    ### To string
    def __str__(self):
        return "C(%s->%s,W=%.4f,I=%d,E=%d)" % (self.from_node, self.to_node, self.weight, self.label, self.enabled)
    def __repr__(self):
        return "C(%s->%s,W=%.4f,I=%d,E=%d)" % (self.from_node, self.to_node, self.weight, self.label, self.enabled)