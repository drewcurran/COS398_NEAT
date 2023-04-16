'''
circuit_games.py
Description: Circuit design game definitions for the NEAT algorithm.
Author: Drew Curran
'''

import numpy as np

class Game:
    def __init__(self, num_inputs, num_outputs, games_per_player):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.games_per_player = games_per_player
        self.maximum_fitness = games_per_player
        self.maximum_fitness_break = 5

    def play_game(self, players, print_game=False):
        wins = []
        max_hit = False
        for player in players:
            player_wins = 0
            player_fitness = 0
            for _ in range(self.games_per_player):
                inputs = self.get_inputs()
                outputs = player.output(inputs)
                won, fitness = self.evaluate(inputs, outputs)
                player_wins += won
                player_fitness += fitness
                if print_game:
                    print(inputs, outputs, won, fitness)
            wins.append(player_wins)
            if not max_hit and player_fitness == self.maximum_fitness:
                max_hit = True
                self.maximum_fitness_break -= 1
            player.fitness = player_fitness
        if not max_hit:
            self.maximum_fitness_break = 5
        return wins

    def get_inputs(self):
        pass

    def evaluate(self):
        pass

class XORGame(Game):
    def __init__(self, games_per_player):
        super().__init__(2, 2, games_per_player)

    def get_inputs(self):
        bias = [1]
        input1 = np.random.randint(2, size=1)
        input2 = np.random.randint(2, size=1)
        return bias + input1.tolist() + input2.tolist()
    
    def evaluate(self, inputs, outputs):
        decision = np.argmax(outputs)
        if inputs[1] == inputs[2] and decision == 0:
            won = 1
        elif inputs[1] != inputs[2] and decision == 1:
            won = 1
        else:
            won = 0
        fitness = won
        return won, fitness
    

class Add4Game(Game):
    def __init__(self, games_per_player):
        super().__init__(8, 5, games_per_player)

    def get_inputs(self):
        bias = [1]
        input1 = np.random.randint(2, size=4)
        input2 = np.random.randint(2, size=4)
        return bias + input1.tolist() + input2.tolist()
    
    def evaluate(self, inputs, outputs):
        value = []
        for output in outputs:
            if output > 0.5:
                value.append(1)
            else:
                value.append(0)
        decision = int("".join(str(x) for x in value), 2)
        input1 = int("".join(str(x) for x in inputs[1:5]), 2)
        input2 = int("".join(str(x) for x in inputs[5:9]), 2)
        fitness = 1 - abs((input1 + input2 - decision) / 2 ** 5)
        if fitness == 1:
            won = 1
        else:
            won = 0
        return won, fitness

class Add8Game(Game):
    def __init__(self, games_per_player):
        super().__init__(16, 9, games_per_player)

    def get_inputs(self):
        bias = [1]
        input1 = np.random.randint(2, size=8)
        input2 = np.random.randint(2, size=8)
        return bias + input1.tolist() + input2.tolist()
    
    def evaluate(self, inputs, outputs):
        value = []
        for output in outputs:
            if output > 0.5:
                value.append(1)
            else:
                value.append(0)
        decision = int("".join(str(x) for x in value), 2)
        input1 = int("".join(str(x) for x in inputs[1:9]), 2)
        input2 = int("".join(str(x) for x in inputs[9:17]), 2)
        fitness = 1 - abs((input1 + input2 - decision) / 2 ** 9)
        if fitness == 1:
            won = 1
        else:
            won = 0
        return won, fitness