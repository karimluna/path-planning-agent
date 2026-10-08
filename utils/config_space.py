import numpy as np

def _wrap_deg(theta):
    '''Return angle in degrees rather than radians.'''
    return np.degrees(np.arctan2(np.sin(theta), np.cos(theta)))