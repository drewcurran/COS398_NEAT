'''
catan.py
Description: Application of the NEAT algorithm to Settlers of Catan.
Author: Drew Curran
'''

import sys
sys.path.append('C:\\Users\\drewc\\Documents\\GitHub\\COS398_CatanAI\\')

from matplotlib import pyplot as plt
import pickle
import argparse

from catanatron import Game, RandomPlayer, Color
from catanatron_gym.envs.catanatron_env import ACTIONS_ARRAY

from NEAT.population import Population
from Catan.catan_players import NEATPlayer
from Catan.catan_training import play_batch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--num-iters', help='number of iterations', required=True, type=int)
    parser.add_argument('-p', '--population-size', help='number of players in population', required=True, type=int)
    parser.add_argument('-g', '--games', help='games per player in population', required=True, type=int)
    parser.add_argument('-n', '--new', help='create new population', action='store_true')
    parser.add_argument('-q', '--quiet', help='hide round progress', action='store_true')
    args = parser.parse_args()
    return args

def print_stats(iteration, num_wins, num_innovations, num_species, avg_fitness, max_fitness):
    print("Iteration: %d, Wins: %d, Innovations: %s, Species: %d, Average Fitness: %.4f, Max Fitness: %.4f" % (iteration, num_wins, num_innovations, num_species, avg_fitness, max_fitness))

def train(num_iters, population_size, games_per_player, new, quiet):
    game_agents = [NEATPlayer(Color.ORANGE), RandomPlayer(Color.RED), RandomPlayer(Color.BLUE), RandomPlayer(Color.WHITE)]
    agent = game_agents[0]

    num_features = len(agent.features.get_feature_values(Game(game_agents))) - 1
    num_actions = len(ACTIONS_ARRAY)

    if new:
        population = Population(population_size, num_features, num_actions)
        agent_wins = []
        num_innovations = []
        num_species = []
        avg_fitness = []
        max_fitness = []
        config = (population_size, games_per_player)
    else:
        try:
            with open('config.pickle', 'rb') as handle:
                config = pickle.load(handle)
            with open('population.pickle', 'rb') as handle:
                population = pickle.load(handle)
            with open('agent_wins.pickle', 'rb') as handle:
                agent_wins = pickle.load(handle)
            with open('num_innovations.pickle', 'rb') as handle:
                num_innovations = pickle.load(handle)
            with open('num_species.pickle', 'rb') as handle:
                num_species = pickle.load(handle)
            with open('avg_fitness.pickle', 'rb') as handle:
                avg_fitness = pickle.load(handle)
            with open('max_fitness.pickle', 'rb') as handle:
                max_fitness = pickle.load(handle)
            assert population.population_size == population_size
        except AssertionError:
            raise Exception("Saved population size is different from that requested.")
        except:
            raise Exception("Cannot load pickle data.")
    
    i = len(agent_wins)
    num_total_iters = i + num_iters
    while i < num_total_iters:
        players = population.new_generation()

        wins, _, _ = play_batch(games_per_player, game_agents, population_players=players, quiet=quiet)

        population.update_generation()

        try:
            agent_wins.append(wins[agent.color])
            num_innovations.append(len(population.innovation_history))
            num_species.append(len(population.species))
            avg_fitness.append(population.sum_average_fitness / len(population.species))
            max_fitness.append(population.max_fitness)
        except:
            continue

        print_stats(i + 1, agent_wins[i], num_innovations[i], num_species[i], avg_fitness[i], max_fitness[i])

        i += 1
    
    with open('config.pickle', 'wb') as handle:
        pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('population.pickle', 'wb') as handle:
        pickle.dump(population, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('agent_wins.pickle', 'wb') as handle:
        pickle.dump(agent_wins, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('num_innovations.pickle', 'wb') as handle:
        pickle.dump(num_innovations, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('num_species.pickle', 'wb') as handle:
        pickle.dump(num_species, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('avg_fitness.pickle', 'wb') as handle:
        pickle.dump(avg_fitness, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('max_fitness.pickle', 'wb') as handle:
        pickle.dump(max_fitness, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    args = parse_args()
    num_iters = args.num_iters
    population_size = args.population_size
    games_per_player = args.games
    new = args.new
    quiet = args.quiet

    train(num_iters, population_size, games_per_player, new, quiet)
    
if __name__ == '__main__':
    main()
