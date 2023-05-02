'''
history.py
Description: Innovation marker to track genetic history.
Author: Drew Curran
'''

class InnovationMarker:
    def __init__(self, innovation_label:int, from_node_label:int, to_node_label:int, gene_labels: list[int]):
        self.innovation_label = innovation_label
        self.from_node_label = from_node_label
        self.to_node_label = to_node_label
        self.initial_genome = gene_labels.copy()

    ### Find same connection mutation
    def matches(self, genome, from_node_label:int, to_node_label:int) -> bool:
        if len(genome.genes) == len(self.initial_genome):
            if from_node_label == self.from_node_label and to_node_label == self.to_node_label:
                for gene in genome.genes:
                    if gene.label not in self.initial_genome:
                        return False
                return True
        return False
    
    ### To string
    def __str__(self):
        return "H(%s->%s,I=%d,L=%s)" % (self.from_node_label, self.to_node_label, self.innovation_label, self.initial_genome)
    def __repr__(self):
        return "H(%s->%s,I=%d,L=%s)" % (self.from_node_label, self.to_node_label, self.innovation_label, self.initial_genome)
