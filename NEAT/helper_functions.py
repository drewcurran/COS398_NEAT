'''
helper_functions.py
Description: The helper functions used to run NEAT.
Author: Drew Curran
'''

import numpy as np

### Sigmoid activation function
def sigmoid(value):
    return 1 / (1 + np.exp(-value))

### ReLU activation function
def relu(value):
    return value if value > 0 else 0