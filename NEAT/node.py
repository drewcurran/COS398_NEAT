'''
node.py
Description: An instance of a node that connects to other nodes in the network.
Author: Drew Curran
'''

class Node:
    def __init__(self, label, layer=0):
        self.label = label
        self.input_value = 0
        self.layer = layer
        self.draw_location = None
        
    ### Return a copy
    def clone(self):
        clone = Node(self.label, layer = self.layer)
        return clone
    
    ### To string
    def __str__(self):
        return "N(%d)" % self.label
    def __repr__(self):
        return "N(%d)" % self.label     