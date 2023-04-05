'''
population.py
Description: Population of players that changes with training.
Author: Drew Curran
'''

import numpy as np

from player import Player
from species import Species

class Population:
    def __init__(self, num_players, num_inputs, num_outputs):
        self.population_size = num_players
        self.generation = 0
        self.player_history = []
        self.innovation_history = []
        players = [Player(num_inputs, num_outputs) for _ in range(num_players)]
        for player in players:
            player.mutate(self.innovation_history)
        self.species = []
        self.speciate(players)

    ### Create new generation
    def new_generation(self):
        children = []
        
        for species in self.species:
            # Add representative player
            children.append(species.representative_player.clone())

        return children

    ### Enforce natural selection on the players that have played
    def update_generation(self):
        staleness_coefficient = 15

        # Sort each species with regard to fitness and cull
        sum_average_fitness = 0
        population_size = 0
        for species in self.species:
            species.sort()
            species.cull(0.5)
            sum_average_fitness += species.average_fitness
            population_size += len(species.players)

        # Sort species by best fitness
        self.species.sort(key=lambda k: k.max_fitness)

        # Kill unimproved species
        if sum_average_fitness == 0:
            return self.species
        for species in self.species:
            if species.staleness > staleness_coefficient and len(self.species) > 1:
                self.species.remove(species)
            elif species.average_fitness / sum_average_fitness * population_size < 1:
                self.species.remove(species)

        return self.species

    ### Separate players into species
    def speciate(self, players):
        # Empty species lists
        for species in self.species:
            species.players = []

        # Determine species match
        for player in players:
            new_species = True
            for species in self.species:
                if species.is_species(player):
                    species.add_player(player)
                    new_species = False
                    break
            if new_species:        
                # Add new species if none match
                self.species.append(Species(player))
        
        return self.species
