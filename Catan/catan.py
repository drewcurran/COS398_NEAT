'''
catan.py
Description: Application of the NEAT algorithm to Settlers of Catan.
Author: Drew Curran
'''

import sys
sys.path.append('..')

from matplotlib import pyplot as plt

from catanatron import Game, RandomPlayer, Color
from catanatron_gym.envs.catanatron_env import ACTIONS_ARRAY

from NEAT.population import Population
from catan_players import NEATPlayer

def main():
    players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE), RandomPlayer(Color.WHITE), NEATPlayer(Color.ORANGE)]
    game = Game(players)
    print(game.play())

    num_iters = 1
    print_step = 1
    population_size = 10

    num_inputs = 20

    population = Population(population_size, num_inputs, len(ACTIONS_ARRAY))

    for iteration in range(num_iters):
        players = population.new_generation()
        
        population.update_generation()

        if iteration % print_step == 0:
            print("Iteration: %d, Innovations: %s, Species: %d, Average Fitness: %.4f, Max Fitness: %.4f" % (iteration, len(population.innovation_history), len(population.species), population.sum_average_fitness / len(population.species), population.max_fitness))
    
    max_player = population.species[0].players[0]
    max_player.nn.draw_state()
    plt.show()

if __name__ == '__main__':
    main()
