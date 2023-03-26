import numpy as np

class ConnectionGene:
    def __init__(self, from_node, to_node, weight, innovation_label, enabled=True):
      self.from_node = from_node
      self.to_node = to_node
      self.weight = weight
      self.innovation_label = innovation_label
      self.enabled = enabled
    
    # Change the weight
    def mutate_weight(self):
        if (np.random.uniform() < 0.1): # Mutate 10% of the time
            self.weight = np.random.uniform(-1, 1)
        else: # Slight change if not mutated
            self.weight += np.random.normal() / 50
        if self.weight > 1:
           self.weight = 1
        elif self.weight < -1:
           self.weight = -1

class ConnectionHistory:
    def __init__(self, from_node, to_node, innovation_label, genome_innovation_labels):
        self.from_node = from_node
        self.to_node = to_node
        self.innovation_label = innovation_label
        self.genome_innovation_labels = genome_innovation_labels
    
    # Determine if matches genome
    def matches_genome(self, genome, from_node, to_node):
        if genome.genes.size() == self.genome_innovation_labels.size():
            if from_node.label == self.from_node and to_node.label == self.to_node:
                for i in range(genome.genes.size()):
                    if not self.genome_innovation_labels.contains(genome.genes[i].label):
                        return False
                return True
        return False