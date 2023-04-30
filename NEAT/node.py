'''
node.py
Description: An instance of a node that connects to other nodes in the network.
Author: Drew Curran
'''

class Node:
    def __init__(self, label:int, layer:int):
        self.label = label
        self.layer = layer
        self.input_value = 0
        self.draw_location = None
    
    ### To string
    def __str__(self):
        return "N(%d)" % self.label
    def __repr__(self):
        return "N(%d)" % self.label     