import numpy as np

from catanatron.models.player import Player
from catanatron_gym.envs.catanatron_env import ACTIONS_ARRAY

class NEATPlayer(Player):
    def __init__(self, features, player, color, is_bot=True):
        assert player.nn.num_inputs == len(features)
        assert player.nn.num_outputs == len(ACTIONS_ARRAY)

        self.features = features
        self.all_actions = ACTIONS_ARRAY
        self.player = player
        self.color = color
        self.is_bot = is_bot

    ### Get the features of the current game state
    def get_feature_values(self, game):
        pass
    
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
