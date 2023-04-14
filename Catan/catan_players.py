'''
catan_players.py
Description: Defining a player using the NEAT neural network.
Author: Drew Curran
'''

import numpy as np

from catanatron.models.player import Player
from catanatron.models.enums import Action, ActionType
from catanatron_gym.envs.catanatron_env import ACTIONS_ARRAY

from Catan.catan_features import CatanFeatures

class NEATPlayer(Player):
    def __init__(self, color, is_bot=True):
        self.color = color
        self.agent = None
        self.features = CatanFeatures(color)
        self.actions = []
        self.robber_actions = {}
        for action in ACTIONS_ARRAY:
            if action[0] != ActionType.MOVE_ROBBER:
                self.actions.append(Action(self.color, action[0], action[1]))
            else:
                self.robber_actions[action[1]] = []
        self.is_bot = is_bot
    
    ### Decide action based on the game state
    def decide(self, game, playable_actions):
        # For efficiency, return only playable action if applicable
        if len(playable_actions) == 1:
            return playable_actions[0]
        
        # If robber is in play, only possible actions are robber moves
        if playable_actions[0][1] == ActionType.MOVE_ROBBER:
            robber_moves_only = True
        else:
            robber_moves_only = False
        
        # Find all the possible robber moves if applicable
        if robber_moves_only:
            for tile in self.robber_actions:
                self.robber_actions[tile].clear()
            for action in playable_actions:
                tile = action[2][0]
                self.robber_actions[tile].append(action)

        # Get the features of the current game state
        inputs = self.features.get_feature_values(game)

        # Get output from the neural network
        outputs = self.agent.output(inputs)
        
        # Apply mask to output
        robber_actions = list(self.robber_actions.values())
        for i in range(len(self.actions) + len(self.robber_actions)):
            if i < len(self.actions):
                if not self.actions[i] in playable_actions:
                    outputs[i] = 0
            else:
                possible_robber_actions = robber_actions[i - len(self.actions)]
                if len(possible_robber_actions) == 0 or not possible_robber_actions[0] in playable_actions:
                    outputs[i] = 0
        
        # Choose output with the highest value
        decision = np.argmax(outputs)
        if decision < len(self.actions):
            action = self.actions[decision]
        else:
            possible_actions = list(self.robber_actions.values())[decision - len(self.actions)]
            action = possible_actions[np.random.randint(len(possible_actions))]
        
        return action