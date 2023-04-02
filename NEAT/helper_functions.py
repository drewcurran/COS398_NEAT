'''
helper_functions.py
Description: The helper functions used to run NEAT.
Author: Drew Curran
'''

import numpy as np

### ReLU function
def relu(value):
    return value if value > 0 else 0