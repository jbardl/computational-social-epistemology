import numpy as np
from scipy import signal

class GameOfLife:

    @staticmethod
    def update(grid, kernel):

        neighbors = signal.convolve2d(grid, kernel, mode='same')

        new_life_condition = (grid==0) & (neighbors==3)
        keep_life_condition = (grid==1) & np.isin(neighbors, [2, 3])

        return np.where(new_life_condition | keep_life_condition, 1, 0)