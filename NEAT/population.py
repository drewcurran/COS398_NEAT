'''
population.py
Description: Population of players that changes with training.
Author: Drew Curran
'''

from NEAT.organism import Organism
from NEAT.species import Species
from NEAT.history import InnovationMarker
from NEAT.parameters import SPECIES_WANTED, SPECIES_BUFFER, GENERATION_GRACE, GENETIC_DISTANCE, DISTANCE_MODIFIER
from NEAT.parameters import MAX_STALENESS

class Population:
    def __init__(self, num_players:int, num_inputs:int, num_outputs:int):
        self.population_size = num_players
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.generation = 0
        self.players = []
        
        self.species_threshold = GENETIC_DISTANCE

    ### Create a new generation
    def new_generation(self) -> list[Organism]:
        if self.generation == 0:
            self.initialize_population()
        else:
            self.reproduce()
        self.mutate()
        self.speciate()
        self.generation += 1

    ### Enforce natural selection on the players that have played
    def update_generation(self):
        self.cull()
        self.update_species()

    ### Initialize the population
    def initialize_population(self) -> list[Organism]:
        self.players = [Organism(self.num_inputs, self.num_outputs) for _ in range(self.population_size)]
        self.species = []
        self.innovations = []

    ### Reproduce using surviving players
    def reproduce(self) -> list[Organism]:
        # Initialize player list
        self.players = []

        # Remove empty species
        for species in self.species[:]:
            if len(species.players) == 0:
                self.species.remove(species)

        # Add fittest player from each species to population
        for species in self.species:
            self.players.append(species.fittest_player)

        for species in self.species:
            # Number of children allocated to species
            num_children = int(species.average_fitness / self.sum_average_fitness * self.population_size)

            # Add representative player
            if (num_children >= 1):
                self.players.append(species.fittest_player.clone(self.innovations))
                num_children -= 1

            # Add children
            for _ in range(num_children):
                self.players.append(species.reproduce(self.innovations))
    
        # Ensure population has correct number of players
        while len(self.players) < self.population_size:
            self.players.append(self.species[0].reproduce())
        while len(self.players) > self.population_size:
            self.players.pop()
    
    ### Mutate player genomes
    def mutate(self):
        for i in range(len(self.players)):
            player = self.players[i]
            if i >= len(self.species):
                player.mutate_genome(self.innovations)

    ### Separate players into species
    def speciate(self):
        # Reset species
        for species in self.species:
            species.reset_species()
        
        # Add each player to a close species
        for i in range(len(self.players)):
            player = self.players[i]
            if i >= len(self.species):
                self.find_species(player).add_player(player)
        
        # Adjust threshold distance based on number of species
        if self.generation > GENERATION_GRACE:
            if len(self.species) > SPECIES_WANTED + SPECIES_BUFFER:
                self.species_threshold += GENETIC_DISTANCE * DISTANCE_MODIFIER
            elif len(self.species) < SPECIES_WANTED - SPECIES_BUFFER:
                self.species_threshold -= GENETIC_DISTANCE * DISTANCE_MODIFIER

    def cull(self):
        # Cull within species
        for species in self.species:
            culled_players = species.cull()
            for player in culled_players:
                if player in self.players:
                    self.players.remove(player)
    
    ### Update species based on performance
    def update_species(self):
        # Update properties of species
        self.sum_average_fitness = 0
        for species in self.species:
            self.sum_average_fitness += species.update()

        # Sort species according to max fitness
        self.species.sort(key=lambda k: k.fittest_player.fitness, reverse=True)
        self.max_fitness = self.species[0].fittest_player.fitness

        # Find stale species
        stale_species = []
        for species in self.species:
            if species.staleness >= MAX_STALENESS:
                stale_species.append(species)
        
        # Remove worst-performing stale species
        if len(stale_species) > 0:
            stale_species.sort(key=lambda k: k.fittest_player.fitness)
            for player in stale_species[0].players:
                if player in self.players:
                    self.players.remove(player)
    
    ### Find close species for a player
    def find_species(self, player:Organism) -> Species:
        # Find close species
        for species in self.species:
            if species.genome_distance(player) < self.species_threshold:
                return species
        
        # Add new species if no species is close enough
        species = Species()
        self.species.append(species)
        return species
