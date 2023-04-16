'''
catan_players.py
Description: Defining a player using the NEAT neural network.
Author: Drew Curran
'''

import numpy as np

from catanatron.models.player import Player
from catanatron_gym.envs.catanatron_env import ACTION_TYPES

from Catan.catan_features import CatanFeatures

class NEATPlayer(Player):
    def __init__(self, color, is_bot=True):
        self.color = color
        self.is_bot = is_bot
        self.agent = None
        self.features = CatanFeatures(color)
        self.actions = {}
        self.reset_actions()
    
    ### Reset the actions based on the game state
    def reset_actions(self, playable_actions=[]):
        self.actions = {}
        for type in ACTION_TYPES:
            self.actions[type] = []
        for action in playable_actions:
            self.actions[action.action_type].append(action)
        self.actions = list(self.actions.values())

    ### Decide action based on the game state
    def decide(self, game, playable_actions):
        # For efficiency, return only playable action if applicable
        if len(playable_actions) == 1:
            return playable_actions[0]

        # Get the features of the current game state
        inputs = self.features.get_feature_values(game)

        # Get output from the neural network
        outputs = self.agent.output(inputs)
        
        # Sort possible actions by their type
        self.reset_actions(playable_actions)
        
        # Choose output with the highest value
        decision = None
        while decision is None:
            action_type = np.random.choice(np.flatnonzero(outputs == np.max(outputs)))
            if len(self.actions[action_type]) == 0:
                outputs[action_type] = 0
            else:
                decision = self.actions[action_type][np.random.choice(len(self.actions[action_type]))]
        
        # Failsafe
        if decision is None:
            decision = playable_actions[0]
        
        return decision