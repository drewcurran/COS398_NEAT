'''
node.py
Description: An instance of a node that connects to other nodes in the network.
Author: Drew Curran
'''

from helper_functions import sigmoid

class Node:
    def __init__(self, label, layer=0):
        self.label = label
        self.input_value = 0
        self.layer = layer
        
    ### Return a copy
    def clone(self):
        clone = Node(self.label, layer = self.layer)
        return clone
                