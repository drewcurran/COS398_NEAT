'''
helper_functions.py
Description: The helper functions used to run NEAT.
Author: Drew Curran
'''

import numpy as np

### Sigmoid function
def sigmoid(value):
    return 1 / (1 + np.exp(-value))