'''
connection_history.py
Description: The connection history allows to test if a mutation is in the same history when a new connection is added.
Author: Drew Curran
'''

from node import Node
from metaclass import MetaClass

class ConnectionHistory(metaclass=MetaClass):
    def __init__(self, from_node, to_node, innovation_label, gene_labels):
        self.from_node = from_node
        self.to_node = to_node
        self.innovation_label = innovation_label
        self.initial_genome = gene_labels.copy()
    
    ### Determine if mutation is on the same genome and makes the same connection as an innovation
    def matches(self, genome, from_node, to_node):
        if len(genome.genes) == len(self.initial_genome):
            if from_node.label == self.from_node and to_node.label == self.to_node:
                for gene in genome.genes:
                    if gene.innovation_label not in self.initial_genome:
                        return False
                return True
        return False
    
    ### To string
    def __str__(self):
        return "History(%s->%s,I=%d,L=%s)" % (self.from_node, self.to_node, self.innovation_label, self.initial_genome)
    def __repr__(self):
        return "History(%s->%s,I=%d,L=%s)" % (self.from_node, self.to_node, self.innovation_label, self.initial_genome)