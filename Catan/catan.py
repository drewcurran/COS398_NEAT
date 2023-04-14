'''
catan.py
Description: Application of the NEAT algorithm to Settlers of Catan.
Author: Drew Curran
'''

import sys
sys.path.append('C:\\Users\\drewc\\Documents\\GitHub\\COS398_CatanAI\\')

from matplotlib import pyplot as plt

from catanatron import Game, RandomPlayer, Color
from catanatron_gym.envs.catanatron_env import ACTIONS_ARRAY

from NEAT.population import Population
from Catan.catan_players import NEATPlayer

def main():
    num_iters = 1
    print_step = 1
    population_size = 10

    num_inputs = 52

    population = Population(population_size, num_inputs, len(ACTIONS_ARRAY))
    players = population.new_generation()
    player = population.species[0].players[0]

    for player in players:
        game_players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE), RandomPlayer(Color.WHITE), NEATPlayer(Color.ORANGE, player, [0] * 20)]
        game = Game(game_players)
        print(game.play())

if __name__ == '__main__':
    main()
