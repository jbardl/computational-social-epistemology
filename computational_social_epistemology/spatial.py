from itertools import product
import numpy as np
from scipy import spatial
import networkx as nx


class Neighborhood:
    """Funcionalidades básicas para la obtención de data de vecinos en un retículo cuadrado
    """

    metric_dict = {
        'moore': 'chebyshev',
        'von_neumann': 'cityblock'
    }

    @classmethod
    def get_idxs(cls, lattice, r=1):
        return NotImplementedError

    @classmethod
    def get_random_neighbor(cls, lattice, point, r=1):
        neighbors = list(cls.get_idxs(lattice, point, r))
        random_neighbor_idx = np.random.choice(np.arange(len(neighbors)))
        return neighbors[random_neighbor_idx]

    @classmethod
    def get_values(cls, lattice, point, r=1):
        neighbor_idxs = cls.get_idxs(lattice, point, r)
        return [lattice[idx] for idx in neighbor_idxs]

    @classmethod
    def _get_kernel(cls, neighborhood, r=1):
        side = 2*r + 1
        middle_idx = np.floor(side/2)
        idxs = list(map(list, product(np.arange(side), repeat=2)))
        distances = spatial.distance.cdist(XA     = idxs,
                                           XB     = [[middle_idx, middle_idx]],
                                           metric = cls.metric_dict[neighborhood])\
                                           .reshape(side, side)
        return np.where((distances<=r) & (distances>0), 1, 0)

    @classmethod
    def make_graph(cls, lattice, r=1):

        x_max, y_max = lattice.shape
        points = list(product(np.arange(x_max),
                              np.arange(y_max)))
        hashes = dict(zip(points, np.arange(len(points))))
        edges = []

        for point in points:
            point_idx = hashes[point]
            point_edges = [
                (point_idx, hashes[neighbor])
                for neighbor in cls.get_idxs(lattice, point, r)
            ]
            edges.extend(point_edges)

        graph = nx.Graph()
        graph.add_edges_from(edges)

        return graph


class MooreNeighborhood(Neighborhood):

    def __init__(self):
        super().__init__()

    @classmethod
    def get_idxs(cls, lattice, point, r=1):
        x, y = point
        x_max, y_max = lattice.shape
        x_range = np.arange(max(0, x-r), min(x_max, x+r+1))
        y_range = np.arange(max(0, y-r), min(y_max, y+r+1))
        neighbor_idxs = set(product(x_range, y_range))
        neighbor_idxs.discard(point)
        return neighbor_idxs

    @classmethod
    def get_kernel(cls, r=1):
        return cls._get_kernel(neighborhood='moore', r=r)


class VonNeumannNeighborhood(Neighborhood):

    def __init__(self):
        super().__init__()

    @classmethod
    def get_idxs(cls, lattice, point, r=1):
        x, y = point
        x_max, y_max = lattice.shape
        x_max -= 1
        y_max -= 1
        neighbors = set([
            (max(x-1, 0), y),
            (min(x+1, x_max), y),
            (x, max(y-1, 0)),
            (x, min(y+1, y_max))
        ])
        neighbors.discard(point)
        return neighbors

    @classmethod
    def get_kernel(cls, r=1):
        return cls._get_kernel(neighborhood='von_neumann', r=r)