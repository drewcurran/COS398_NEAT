'''
connection_history.py
Description: The connection history allows to test if a genome is the same as a pre-existing genome.
Author: Drew Curran
'''

from node import Node

class ConnectionHistory:
    def __init__(self, from_node, to_node, innovation_label, genome_innovation_labels):
        self.from_node = from_node
        self.to_node = to_node
        self.innovation_label = innovation_label
        self.genome_innovation_labels = genome_innovation_labels.copy()
    
    ### Determine if matches genome
    def matches_genome(self, genome, from_node, to_node):
        if len(genome.genes) == len(self.genome_innovation_labels):
            if from_node.label == self.from_node and to_node.label == self.to_node:
                for i in range(len(genome.genes)):
                    if genome.genes[i].innovation_label not in self.genome_innovation_labels:
                        return False
                return True
        return False