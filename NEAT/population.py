class Population:
    def __init__(self, size):
        self.players = None
        self.generation = 0
        self.innovation_history = None
        self.generation_players = None
        self.species = None
    
    # Enforce natural selection on the players
    def new_generation(self):
        pass