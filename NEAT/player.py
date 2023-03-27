from genome import Genome

class Player:
    def __init__(self, inputs, outputs):
        self.nn = Genome(inputs, outputs)
        self.inputs = inputs
        self.outputs = outputs
        self.fitness = 0
        self.lifespan = 0
    
    # Make decision based on neural network
    def decide(self):
        output = self.nn.forward_pass(self.input)
        # Sort decisions
        for decision in output:
            if decision: # if it can be played
                return decision
    
    # Calculate fitness of the player
    def evaluate(self, victory_points, opponents_victory_points):
        self.fitness = 0
        for vp in opponents_victory_points:
            self.fitness += victory_points - vp

    # Create child player with self and another player as parents
    def crossover(self, player):
        child = Player(self.inputs, self.outputs)
        child.nn = self.nn.crossover(player.nn)
        child.nn.generate_network()
        return child
