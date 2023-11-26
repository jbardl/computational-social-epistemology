from .spatial import Neighborhood
import numpy as np
from scipy import signal

class SchellingSegregation:

    def __init__(self,
                 epochs,
                 side,
                 threshold,
                 empty,
                 percent_agents,
                 neighborhood,
                 r = 1,
                 v = True):
        self.epochs = epochs
        self.side = side
        self.threshold = threshold
        self.empty = empty
        self.percent_agents = percent_agents
        self.neighborhood = neighborhood
        self.r = r
        self.v = v

        self.percent_1, self.percent_2 = self.percent_agents
        self.kernel = Neighborhood._get_kernel(
            neighborhood = self.neighborhood,
            r            = self.r
        )

    def init_lattice(self):
        total_agents = self.side * self.side
        empty_cells = total_agents * self.empty
        populated_cells = total_agents - empty_cells

        agents_1 = int(populated_cells * self.percent_1)
        agents_2 = int(populated_cells - agents_1)

        lattice = np.zeros(total_agents)
        lattice[:agents_1] = 1
        lattice[-agents_2:] = 2
        np.random.shuffle(lattice)
        lattice = lattice.reshape(self.side, self.side)

        return lattice

    def update(self, lattice):

        total_neighbors = signal.convolve2d(lattice != 0, self.kernel, mode='same')
        neighbors_1 = signal.convolve2d(lattice == 1, self.kernel, mode='same')
        neighbors_2 = signal.convolve2d(lattice == 2, self.kernel, mode='same')

        dissatisfied_1 = (lattice == 1) & (neighbors_1 / total_neighbors < self.threshold)
        dissatisfied_2 = (lattice == 2) & (neighbors_2 / total_neighbors < self.threshold)

        dissatisfied = np.any([dissatisfied_1, dissatisfied_2], axis=0)
        lattice[dissatisfied] = 0
        vacant = np.sum(lattice == 0)

        n_dissatisfied_1, n_dissatisfied_2 = dissatisfied_1.sum(), dissatisfied_2.sum()
        filling = np.zeros(vacant, dtype=np.int8)
        filling[:n_dissatisfied_1] = 1
        filling[-n_dissatisfied_2:] = 2
        np.random.shuffle(filling)
        lattice[lattice==0] = filling

        return lattice