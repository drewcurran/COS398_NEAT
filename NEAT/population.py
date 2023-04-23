'''
population.py
Description: Population of players that changes with training.
Author: Drew Curran
'''

from NEAT.player import Player
from NEAT.species import Species

class Population:
    def __init__(self, num_players, num_inputs, num_outputs, speciation=True):
        self.population_size = num_players
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.speciation = speciation
        self.generation = 0
        self.player_history = []
        self.innovation_history = []
        self.species = []
        self.sum_average_fitness = 0
        self.max_fitness = 0

    ### Create new generation
    def new_generation(self):
        # First generation
        if self.generation == 0:
            players = [Player(self.num_inputs, self.num_outputs) for _ in range(self.population_size)]
            for player in players:
                player.mutate(self.innovation_history)
        # Progeny
        else:
            players = []

            for species in self.species:
                # Number of children allocated to species
                num_children = int(species.average_fitness / self.sum_average_fitness * self.population_size)

                # Add representative player
                players.append(species.representative_player.clone())
                num_children -= 1

                # Add children
                for _ in range(num_children):
                    players.append(species.make_child(self.innovation_history))
        
        # Ensure population has correct number of players
        while len(players) < self.population_size:
            players.append(self.species[0].make_child(self.innovation_history))
        while len(players) > self.population_size:
            players.pop()

        self.speciate(players)
        self.generation += 1

        return players

    ### Enforce natural selection on the players that have played
    def update_generation(self):
        staleness_coefficient = 15

        # Sort each species with regard to fitness and cull
        for species in self.species:
            species.sort()
            species.cull(0.5)

        # Sort species by best fitness
        self.species.sort(key=lambda k: k.max_fitness, reverse=True)
        self.max_fitness = self.species[0].players[0].fitness

        # Kill unimproved species
        self.sum_average_fitness = 0
        for species in self.species:
            if species.staleness > staleness_coefficient and len(self.species) > 1:
                self.species.remove(species)
            else:
                self.sum_average_fitness += species.average_fitness

        return self.species

    ### Separate players into species
    def speciate(self, players):
        # Empty species lists
        self.species = []

        # Determine species match
        if self.speciation:
            for player in players:
                new_species = True
                for species in self.species:
                    if species.is_species(player):
                        species.add_player(player)
                        new_species = False
                        break
                if new_species:
                    # Add new species if none match
                    self.species.append(Species([player]))
        else:
            self.species.append(Species(players))
        
        return self.species
