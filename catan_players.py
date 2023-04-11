import numpy as np

from catanatron.models.player import Player
from catanatron_gym.envs.catanatron_env import ACTIONS_ARRAY, RESOURCES
from catanatron_gym.features import iter_players
from catanatron.state_functions import player_key

class NEATPlayer(Player):
    def __init__(self, features, player, color, is_bot=True):
        assert player.nn.num_inputs == len(features)
        assert player.nn.num_outputs == len(ACTIONS_ARRAY)

        self.features = features
        self.all_actions = ACTIONS_ARRAY
        self.player = player
        self.color = color
        self.is_bot = is_bot
    
    ### Decide action based on the game state
    def decide(self, game, playable_actions):
        # Get the features of the current game state
        input = self.get_feature_values(game)

        # Get output from the neural network
        output = self.player.output(input)

        # Apply mask to output
        for i in range(len(ACTIONS_ARRAY)):
            if not ACTIONS_ARRAY[i] in playable_actions:
                output[i] = 0
        
        # Choose output with the highest value
        decision = np.argmax(output)

        return ACTIONS_ARRAY[decision]
    
        ### Get the features of the current game state
    def get_feature_values(self, game):
        features = {}
        pkey = player_key(game.state, self.color)
        

        # Feature for each resource amount in hand
        for resource in RESOURCES:
            features[f"PLAYER_{resource}_IN_HAND"] = game.state.player_state[key + f"_{resource}_IN_HAND"]

        # Feature for longest road and max longest road
        max_longest = 0
        for i, color in iter_players(game.state.colors, self.color):
          key = player_key(game.state, color)
          longest = game.state.player_state[key + "_LONGEST_ROAD_LENGTH"]
          if color == self.color:
              features["PLAYER_LONGEST_ROAD_DIFFERENCE"] = longest
          elif longest > max_longest:
              max_longest = longest
        features["MAX_LONGEST_ROAD_DIFFERENCE"] = max_longest

        # Feature for knights and max knights
        max_longest = 0
        for i, color in iter_players(game.state.colors, self.color):
          key = player_key(game.state, color)
          longest = game.state.player_state[key + "_LONGEST_ROAD_LENGTH"]
          if color == self.color:
              features["PLAYER_LONGEST_ROAD_DIFFERENCE"] = longest
          elif longest > max_longest:
              max_longest = longest
        features["MAX_LONGEST_ROAD_DIFFERENCE"] = max_longest