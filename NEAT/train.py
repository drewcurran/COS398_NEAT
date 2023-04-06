'''
game.py
Description: Application and training for the NEAT algorithm.
Author: Drew Curran
'''

import numpy as np


### Play a game
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
