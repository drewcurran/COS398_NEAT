'''
connection_history.py
Description: The connection history allows to test if a mutation is in the same history when a new connection is added.
Author: Drew Curran
'''

from node import Node

class ConnectionHistory:
    def __init__(self, from_node, to_node, innovation_label, gene_labels):
        self.from_node = from_node
        self.to_node = to_node
        self.innovation_label = innovation_label
        self.initial_genome = gene_labels.copy()
    
    ### Determine if matches original mutation genome and connection is between the same nodes
    def matches(self, genome, from_node, to_node):
        if len(genome.genes) == len(self.initial_genome):
            if from_node.label == self.from_node and to_node.label == self.to_node:
                for gene in genome.genes:
                    if gene.innovation_label not in self.initial_genome:
                        return False
                return True
        return False