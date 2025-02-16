from typing import Tuple
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class RelativeAgreementModel:

    def __init__(self,
                 n: int,
                 mu: float,
                 epochs: int,
                 opinion_range: Tuple[float] = (-1., 1.),
                 uncertainty: float = .4,
                 v: bool = False):
        self.n = n
        self.mu = mu
        self.epochs = epochs
        self.opinion_range = opinion_range
        self.uncertainty = uncertainty
        self.v = v

        self.agents = None
        self.history = None


    def run(self):
        self.init_params()
        iterator = range(self.epochs)
        for epoch in tqdm(iterator) if self.v else iterator:
            random_pairs_idxs = np.random.choice(a=self.n, replace=False,
                                                 size=(int(self.n/2), 2))
            random_pairs = self.agents[random_pairs_idxs]
            new_profile = np.empty((self.n, 2))
            new_profile[random_pairs_idxs] = np.array(list(map(self.update, random_pairs)))
            self.agents = new_profile.reshape(self.n, 2)
            self.history = np.concatenate([self.history, new_profile[:, 0].reshape(-1, 1)], axis=1)


    def init_params(self):
        low, high = self.opinion_range
        opinions = np.random.uniform(low=low, high=high, size=(self.n, 1))
        # uncertainties = np.random.uniform(low=0., high=2., size=(self.n, 1))
        uncertainties = np.full(shape=(self.n, 1), fill_value=self.uncertainty)
        self.agents = np.concatenate([opinions, uncertainties], axis=1)
        self.agents.sort(axis=0)
        self.history = self.agents[:, 0].reshape(-1, 1)


    def update(self, pair):
        agent_1, agent_2 = tuple(map(tuple, pair))
        agent_1_updated = self.pairwise_update(agent_2, agent_1)
        agent_2_updated = self.pairwise_update(agent_1, agent_2)
        return np.array([agent_1_updated, agent_2_updated])


    def pairwise_update(self, agent_1, agent_2):
        opinion_1, uncertainty_1 = agent_1
        opinion_2, uncertainty_2 = agent_2

        overlap = min(opinion_1+uncertainty_1, opinion_2+uncertainty_2) - \
                  max(opinion_1-uncertainty_1, opinion_2-uncertainty_2)

        if overlap > uncertainty_1:
            relative_agreement = (overlap / uncertainty_1) - 1
            new_opinion     = opinion_2     + self.mu * relative_agreement * (opinion_1-opinion_2)
            new_uncertainty = uncertainty_2 + self.mu * relative_agreement * (uncertainty_1-uncertainty_2)
            return np.array([new_opinion, new_uncertainty])

        else:
            return np.array([opinion_2, uncertainty_2])


    def plot_results(self):
        fig, ax = plt.subplots(figsize=(13,6))
        cmap = plt.get_cmap('hsv', self.n)
        x = np.arange(self.history.shape[1])

        for agent in range(self.n):
            ax.plot(x, self.history[agent, :], color=cmap(agent))