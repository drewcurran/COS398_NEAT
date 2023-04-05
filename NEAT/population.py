'''
population.py
Description: Population of players that changes with training.
Author: Drew Curran
'''

from player import Player
from species import Species

class Population:
    def __init__(self, size, evaluation_function):
        self.population = []
        self.population_size = size
        self.generation = 0
        self.player_history = []
        self.innovation_history = []
        self.evaluation_function = evaluation_function

    ### Enforce natural selection on the players that have played
    def update_generation(self, players):
        staleness_coefficient = 15

        # Empty species lists
        for species in self.population:
            species.players = []

        # Separate players into species
        for player in players:
            for species in self.population:
                if species.is_species(player):
                    species.add_player(player)
                    break
            # Add new species if none match
            self.population.append(Species(player))
            self.num_species += 1

        # Sort each species with regard to fitness and cull
        sum_average_fitness = 0
        population_size = 0
        for species in self.population:
            species.sort()
            species.cull(0.5)
            sum_average_fitness += species.average_fitness
            population_size += len(species.players)

        # Sort species by best fitness
        self.population.sort(key=lambda k: k.max_fitness)

        # Kill bad and unimproved species
        for species in self.population:
            if species.staleness > staleness_coefficient and self.num_species > 1:
                self.population.remove(species)
                self.num_species -= 1
            elif species.average_fitness / sum_average_fitness * population_size < 1:
                self.population.remove(species)
                self.num_species -= 1

        return self.population
