'''
catan_features.py
Description: Features of the game Settlers of Catan.
Author: Drew Curran
'''

from catanatron_gym.features import iter_players
from catanatron.state_functions import player_key, player_num_dev_cards, player_num_resource_cards
from catanatron.models.map import number_probability
from catanatron.models.enums import RESOURCES, DEVELOPMENT_CARDS, VICTORY_POINT

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
            features[f"P{i}_PUBLIC_VPS"] = game.state.player_state[key + "_VICTORY_POINTS"]

        # Each resource amount in hand
        for resource in RESOURCES:
            features[f"P0_{resource}_IN_HAND"] = game.state.player_state[pkey + f"_{resource}_IN_HAND"]

        # Total resource amount in hand
        features[f"P0_NUM_RESOURCES_IN_HAND"] = player_num_resource_cards(game.state, self.color)

        # Each development card amount in hand
        for card in DEVELOPMENT_CARDS:
            features[f"P0_{card}_IN_HAND"] = game.state.player_state[pkey + f"_{card}_IN_HAND"]

        # Total development card amount in hand
        features[f"P0_NUM_DEVS_IN_HAND"] = player_num_dev_cards(game.state, self.color)

        # Development cards played for each player
        for i, color in iter_players(game.state.colors, self.color):
            key = player_key(game.state, self.color)
            for card in DEVELOPMENT_CARDS:
                if card == VICTORY_POINT:
                    continue
                features[f"P{i}_{card}_PLAYED"] = game.state.player_state[key + f"_PLAYED_{card}"]

        # Tile production values
        for tile_id, tile in game.state.board.map.tiles_by_id.items():
            features[f"TILE{tile_id}_PROBA"] = 0 if tile.resource is None else number_probability(tile.number)

        return [1] + list(features.values())