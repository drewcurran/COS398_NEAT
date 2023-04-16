'''
circuit_training.py
Description: Circuit design application and training for the NEAT algorithm.
Author: Drew Curran
'''

import sys
sys.path.append('C:\\Users\\drewc\\Documents\\GitHub\\COS398_CatanAI\\')

import numpy as np
from matplotlib import pyplot as plt
import argparse
import pickle

from NEAT.population import Population
from Circuit_Design.circuit_games import XORGame, Add4Game, Add8Game

### Parse the command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--game-type', help='type of circuit to train', required=True)
    parser.add_argument('-i', '--num-iters', help='number of iterations', required=True, type=int)
    parser.add_argument('-p', '--population-size', help='number of players in population', required=True, type=int)
    parser.add_argument('-g', '--games', help='games per player in population', required=True, type=int)
    parser.add_argument('-n', '--new', help='create new population', action='store_true')
    args = parser.parse_args()
    return args

### Print the stats of an iteration
def print_stats(iteration, num_wins, num_innovations, num_species, avg_fitness, max_fitness):
    print("Iteration: %d, Wins: %d, Innovations: %s, Species: %d, Average Fitness: %.4f, Max Fitness: %.4f" % (iteration, num_wins, num_innovations, num_species, avg_fitness, max_fitness))

### Save training data
def save(game_label, config, population, agent_wins, num_innovations, num_species, avg_fitness, max_fitness):
    try:
        with open(f'{game_label}/config.pickle', 'wb') as handle:
            pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{game_label}/population.pickle', 'wb') as handle:
            pickle.dump(population, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{game_label}/agent_wins.pickle', 'wb') as handle:
            pickle.dump(agent_wins, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{game_label}/num_innovations.pickle', 'wb') as handle:
            pickle.dump(num_innovations, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{game_label}/num_species.pickle', 'wb') as handle:
            pickle.dump(num_species, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{game_label}/avg_fitness.pickle', 'wb') as handle:
            pickle.dump(avg_fitness, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{game_label}/max_fitness.pickle', 'wb') as handle:
            pickle.dump(max_fitness, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Progress saved.")
    except:
        print("Failed to save.")

### Load training data
def load(game_label, population_size):
    try:
        with open(f'{game_label}/config.pickle', 'rb') as handle:
            config = pickle.load(handle)
        with open(f'{game_label}/population.pickle', 'rb') as handle:
            population = pickle.load(handle)
        with open(f'{game_label}/agent_wins.pickle', 'rb') as handle:
            agent_wins = pickle.load(handle)
        with open(f'{game_label}/num_innovations.pickle', 'rb') as handle:
            num_innovations = pickle.load(handle)
        with open(f'{game_label}/num_species.pickle', 'rb') as handle:
            num_species = pickle.load(handle)
        with open(f'{game_label}/avg_fitness.pickle', 'rb') as handle:
            avg_fitness = pickle.load(handle)
        with open(f'{game_label}/max_fitness.pickle', 'rb') as handle:
            max_fitness = pickle.load(handle)
        assert population.population_size == population_size
    except AssertionError:
        raise Exception("Saved population size is different from that requested.")
    except:
        raise Exception("Cannot load pickle data.")
    return config, population, agent_wins, num_innovations, num_species, avg_fitness, max_fitness

### Train the neural network
def train(game_label, num_iters, population_size, games_per_player, new):
    if game_label == 'xor':
        game = XORGame(games_per_player)
    if game_label == 'add4':
        game = Add4Game(games_per_player)
    if game_label == 'add8':
        game = Add8Game(games_per_player)
    
    if new:
        config = (population_size, games_per_player)
        population = Population(population_size, game.num_inputs, game.num_outputs)
        agent_wins = []
        num_innovations = []
        num_species = []
        avg_fitness = []
        max_fitness = []
    else:
        config, population, agent_wins, num_innovations, num_species, avg_fitness, max_fitness = load(game_label, population_size)
    
    i = len(agent_wins)
    num_total_iters = i + num_iters
    while i < num_total_iters:
        # Generate new players
        players = population.new_generation()

        # Play games
        wins = game.play_game(players)

        # Enforce natural selection
        population.update_generation()
        
        # Get the stats from the training iteration
        try:
            agent_wins.append(np.sum(wins))
            num_innovations.append(len(population.innovation_history))
            num_species.append(len(population.species))
            avg_fitness.append(population.sum_average_fitness / len(population.species))
            max_fitness.append(population.max_fitness)
        except:
            continue
        print_stats(i + 1, agent_wins[i], num_innovations[i], num_species[i], avg_fitness[i], max_fitness[i])

        # Break from loop if maximum fitness threshold has been reached
        if game.maximum_fitness_break == 0:
            save(game_label, config, population, agent_wins, num_innovations, num_species, avg_fitness, max_fitness)
            break

        # Save progress
        if i % 5 == 0:
            save(game_label, config, population, agent_wins, num_innovations, num_species, avg_fitness, max_fitness)
            
        i += 1

def main():
    args = parse_args()
    game = args.game_type
    num_iters = args.num_iters
    population_size = args.population_size
    games_per_player = args.games
    new = args.new

    train(game, num_iters, population_size, games_per_player, new)
    
if __name__ == '__main__':
    main()
