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

### Parse the command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--game', help='type of circuit to train', required=True, type=int)
    parser.add_argument('-p', '--population-size', help='number of players in population', required=True, type=int)
    parser.add_argument('-g', '--games', help='games per player in population', required=True, type=int)
    parser.add_argument('-n', '--new', help='create new population', action='store_true')
    args = parser.parse_args()
    return args

### Print the stats of an iteration
def print_stats(iteration, num_wins, num_innovations, num_species, avg_fitness, max_fitness):
    print("Iteration: %d, Wins: %d, Innovations: %s, Species: %d, Average Fitness: %.4f, Max Fitness: %.4f" % (iteration, num_wins, num_innovations, num_species, avg_fitness, max_fitness))

### Play the specified game
def play_game(players, games_per_player):
    won = []
    for player in players:
        won.append(0)
        for _ in range(games_per_player):
            inputs = [1] + np.random.randint(2, size=2).tolist()
            decision = player.decide(inputs)
            if inputs[1] == inputs[2] and decision == 0:
                won[len(won) - 1] += 1
            elif inputs[1] != inputs[2] and decision == 1:
                won[len(won) - 1] += 1
    return won

### Train the neural network
def train(game, num_iters, population_size, games_per_player, new):
    if new:
        config = (population_size, games_per_player)
        population = Population(population_size, game.num_inputs, game.num_outputs)
        agent_wins = []
        num_innovations = []
        num_species = []
        avg_fitness = []
        max_fitness = []
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
        # Generate new players
        players = population.new_generation()

        # Save progress
        if i % 5 == 0:
            try:
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
                print("Progress saved.")
            except:
                print("Failed to save.")

        # Play games
        wins, _, _ = play_game(players, games_per_player)

        # Enforce natural selection
        population.update_generation()
        
        # Get the stats from the training iteration
        try:
            agent_wins.append(wins)
            num_innovations.append(len(population.innovation_history))
            num_species.append(len(population.species))
            avg_fitness.append(population.sum_average_fitness / len(population.species))
            max_fitness.append(population.max_fitness)
        except:
            continue
        print_stats(i + 1, agent_wins[i], num_innovations[i], num_species[i], avg_fitness[i], max_fitness[i])

        i += 1

def main():
    args = parse_args()
    num_iters = args.num_iters
    population_size = args.population_size
    games_per_player = args.games
    new = args.new

    train(num_iters, population_size, games_per_player, new)
    
if __name__ == '__main__':
    main()


def main():
    num_iters = 100
    print_step = 1
    population_size = 1000
    num_inputs = 2
    num_outputs = 2
    max_hits_threshold = 5

    population = Population(population_size, num_inputs, num_outputs)
    max_hits = 0

    for iteration in range(num_iters):
        players = population.new_generation()
        won = play_game(players)
        for i in range(len(players)):
            players[i].fitness = won[i]
        population.update_generation()
        if population.max_fitness == 100:
            print("Iteration: %d, Innovations: %s, Species: %d, Average Fitness: %.4f, Max Fitness: %.4f" % (iteration, len(population.innovation_history), len(population.species), population.sum_average_fitness / len(population.species), population.max_fitness))
            max_hits += 1
        elif iteration % print_step == 0:
            print("Iteration: %d, Innovations: %s, Species: %d, Average Fitness: %.4f, Max Fitness: %.4f" % (iteration, len(population.innovation_history), len(population.species), population.sum_average_fitness / len(population.species), population.max_fitness))
            max_hits = 0
        else:
            max_hits = 0
        if max_hits == max_hits_threshold:
            break
    
    max_player = population.species[0].players[0]
    for i in range(4):
        inputs = [1, 0 if i < 2 else 1, 0 if i % 2 == 0 else 1]
        decision = max_player.decide(inputs)
        print(inputs, max_player.nn.forward_pass(inputs), decision)
    max_player.nn.print_state()
    max_player.nn.draw_state()
    plt.show()

if __name__ == '__main__':
    main()
