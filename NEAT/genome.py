class Genome:
    def __init__(self, inputs, outputs, layers=2):
        self.inputs = inputs
        self.outputs = outputs
        self.layers = layers
        self.genes = None
        self.nodes = None
        self.next_node = None
        self.bias_node = None
        self.network = None
        