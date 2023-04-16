'''
catan_features.py
Description: Features of the game Settlers of Catan.
Author: Drew Curran
'''

from catanatron_gym.features import iter_players
from catanatron.state_functions import player_key
from catanatron.models.enums import RESOURCES

class CatanFeatures():
    def __init__(self, color):
        self.color = color
    
    ### Get the features of the current game state
    def get_feature_values(self, game):
        features = {}
        pkey = player_key(game.state, self.color)
        
        # Number of victory points for each player
        for i, color in iter_players(game.state.colors, self.color):
            key = player_key(game.state, color)
            if color == self.color:
                features["P0_ACTUAL_VPS"] = game.state.player_state[key + "_ACTUAL_VICTORY_POINTS"]
            else:
                features[f"P{i}_PUBLIC_VPS"] = game.state.player_state[key + "_VICTORY_POINTS"]

        # Each resource amount in hand
        for resource in RESOURCES:
            features[f"P0_{resource}_IN_HAND"] = game.state.player_state[pkey + f"_{resource}_IN_HAND"]

        return [1] + list(features.values())