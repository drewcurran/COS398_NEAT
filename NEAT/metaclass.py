'''
metaclass.py
Description: Allows for descriptor binding.
Author: Drew Curran
'''

class MetaClass(type):
    def __str__(cls):
        return "__str__ on the metaclass"
    def __repr__(cls):
        return "__repr__ on the metaclass"