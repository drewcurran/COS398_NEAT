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

    # Get node with matching label
    def get_node(self, label):
        for i in range(self.nodes.size()):
            if self.nodes.get(i).label == label:
                return self.nodes.get(i)
        return None
    
    # Connects nodes
    def connect_nodes():
        
        