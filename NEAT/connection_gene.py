'''
connection_gene.py
Description: The connection gene links two nodes as input and output due to a weight.
Author: Drew Curran
'''

import numpy as np

from node import Node

class ConnectionGene:
    def __init__(self, from_node, to_node, weight, innovation_label, enabled=True):
      self.from_node = from_node
      self.to_node = to_node
      self.weight = weight
      self.innovation_label = innovation_label
      self.enabled = enabled
    
    ### Change the weight
    def mutate_weight(self):
        if (np.random.uniform() < 0.1): # Mutate 10% of the time
            self.weight = np.random.uniform(-1, 1)
        else: # Slight change if not mutated
            self.weight += np.random.normal() / 50
        if self.weight > 1:
           self.weight = 1
        elif self.weight < -1:
           self.weight = -1

    ### Return a copy
    def clone(self, from_node, to_node):
        clone = ConnectionGene(from_node, to_node, self.weight, self.innovation_label, enabled=self.enabled)
        clone.enabled = self.enabled
        return clone