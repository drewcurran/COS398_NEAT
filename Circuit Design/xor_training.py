'''
xor_training.py
Description: XOR application and training for the NEAT algorithm.
Author: Drew Curran
'''

import sys
sys.path.append('C:\\Users\\drewc\\Documents\\GitHub\\COS398_CatanAI\\')

import numpy as np
from matplotlib import pyplot as plt
import argparse

from NEAT.population import Population

### Parse the command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--num-iters', help='number of iterations', required=True, type=int)
    parser.add_argument('-p', '--population-size', help='number of players in population', required=True, type=int)
    parser.add_argument('-g', '--games', help='games per player in population', required=True, type=int)
    parser.add_argument('-n', '--new', help='create new population', action='store_true')
    parser.add_argument('-q', '--quiet', help='hide round progress', action='store_true')
    args = parser.parse_args()
    return args

### Play the specified game
def play_game(players):
    won = []
    for player in players:
        won.append(0)
        for _ in range(100):
            inputs = [1] + np.random.randint(2, size=2).tolist()
            decision = player.decide(inputs)
            if inputs[1] == inputs[2] and decision == 0:
                won[len(won) - 1] += 1
            elif inputs[1] != inputs[2] and decision == 1:
                won[len(won) - 1] += 1
    return won

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
