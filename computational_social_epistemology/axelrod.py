import numpy as np

class AxelrodModel:

    def __init__(self,
                 f: int,
                 q: int,
                 side: int,
                 neighborhood: object,
                 v = True):
        self.F = f
        self.q = q
        self.side = side
        self.neighborhood = neighborhood
        self.v = v

        self.logs = []


    def init_lattice(self):
        return np.random.randint(low=0,
                                 high=self.q-1,
                                 size=(self.F, self.side, self.side))


    def update(self, lattice):
        point = np.random.randint(low=0, high=self.side, size=2)
        neighbor = self.neighborhood.get_random_neighbor(lattice[0], tuple(point))
        data_point, data_neighbor = (
            lattice[:, point[0], point[1]],
            lattice[:, neighbor[0], neighbor[1]]
        )
        coincidences = data_point == data_neighbor
        interaction_proba = sum(coincidences) / self.F

        if interaction_proba < 1.0:
            do_interact = np.random.choice(a=[True, False],
                                           p=[interaction_proba,
                                              1-interaction_proba])
            if do_interact:
                random_trait_idx = np.random.choice([idx for idx, coincidence
                                                     in enumerate(coincidences)
                                                     if not coincidence])
                lattice[random_trait_idx, point[0], point[1]] = data_neighbor[random_trait_idx]
        return lattice


    def get_plot_lattice(self, lattice):
        str_lattice = lattice.astype(str)
        return np.array([
            "".join(str_lattice[:, idx[0], idx[1]])
            for idx, _ in np.ndenumerate(lattice[0])
        ]).reshape(self.side, self.side).astype(int)