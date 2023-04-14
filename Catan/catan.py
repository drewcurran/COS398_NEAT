'''
catan.py
Description: Application of the NEAT algorithm to Settlers of Catan.
Author: Drew Curran
'''

import sys
sys.path.append('C:\\Users\\drewc\\Documents\\GitHub\\COS398_CatanAI\\')

from matplotlib import pyplot as plt
import pickle

from catanatron import Game, RandomPlayer, Color
from catanatron_gym.envs.catanatron_env import ACTIONS_ARRAY

from NEAT.population import Population
from Catan.catan_players import NEATPlayer
from Catan.catan_stats import play_batch

def main():
    game_agents = [NEATPlayer(Color.ORANGE), RandomPlayer(Color.RED), RandomPlayer(Color.BLUE), RandomPlayer(Color.WHITE)]
    agent = game_agents[0]

    num_iters = 50
    games_per_player = 2
    population_size = 1000
    num_features = len(agent.features.get_feature_values(Game(game_agents))) - 1
    num_actions = len(ACTIONS_ARRAY)

    try:
        with open('population.pickle', 'rb') as handle:
            population = pickle.load(handle)
        with open('agent_wins.pickle', 'rb') as handle:
            agent_wins = pickle.load(handle)
        num_total_iters = len(agent_wins) + num_iters
    except:
        population = Population(population_size, num_features, num_actions)
        agent_wins = []

    for _ in range(num_iters):
        players = population.new_generation()

        wins, vps_by_player, games = play_batch(games_per_player, game_agents, population_players=players, quiet=True)
        agent_wins.append(wins[agent.color])
        
        for p in range(len(players)):
            players[p].fitness = 0
            for g in range(games_per_player):
                players[p].fitness += vps_by_player[agent.color][p * games_per_player + g]

        population.update_generation()
    
    with open('population.pickle', 'wb') as handle:
        pickle.dump(population, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('agent_wins.pickle', 'wb') as handle:
        pickle.dump(agent_wins, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

    plt.plot(range(1, num_total_iters + 1), agent_wins, color='Orange')
    plt.ylim([0, population_size * games_per_player])
    plt.xticks(list(range(25, num_total_iters, 25)) + [num_total_iters])
    plt.yticks(list(range(0, population_size * games_per_player, population_size)) + [population_size * games_per_player])
    plt.show()
    
if __name__ == '__main__':
    main()
