from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class SocialJudgementModel:
    def __init__(self, n, mu, u, t, epochs, v=True):
        self.n = n
        self.mu = mu
        self.u = u
        self.t = t
        self.epochs = epochs
        self.v = v

        self.opinion_history = None

    @property
    def last_opinion_profile(self):
        return self.opinion_history[:, -1]

    def run(self):
        self.init_params()
        iterator = range(self.epochs)
        for epoch in tqdm(iterator) if self.v else iterator:
            # correr update sobre el últio perfil de opinión
            new_opinion_profile = self.update(self.last_opinion_profile)
            # agregar el resultado a la historia
            self.opinion_history = np.concatenate([self.opinion_history, new_opinion_profile], axis=1)
            # chequear convergencia
            convergence = self.check_convergence()
            if convergence: break

    def init_params(self):
        opinions = np.random.uniform(low=-1, high=1, size=self.n)
        opinions.sort()
        self.opinion_history = opinions.reshape(-1, 1)

    def update(self, opinion_profile):
        random_pairs_idxs = np.random.choice(a=self.n, replace=False,
                                             size=(int(self.n / 2), 2))
        random_pairs = opinion_profile[random_pairs_idxs]
        update_result = np.array(list(map(self.pairwise_update, random_pairs)))
        new_profile = np.empty((self.n))
        new_profile[random_pairs_idxs.reshape(-1)] = update_result.reshape(-1)
        return new_profile.reshape(-1, 1)

    def check_convergence(self):
        return np.all(self.opinion_history[:, -2] == self.last_opinion_profile)

    def pairwise_update(self, pair):
        opinion_1, opinion_2 = tuple(pair)
        new_opinion_1 = self.individual_update(opinion_1, opinion_2)
        new_opinion_2 = self.individual_update(opinion_2, opinion_1)
        return np.array([new_opinion_1, new_opinion_2])

    def individual_update(self, opinion_1, opinion_2):
        difference = np.abs(opinion_1 - opinion_2)
        # assimilation
        if difference < self.u:
            influence = self.mu * (opinion_2 - opinion_1)
        # contrast
        elif difference > self.t:
            influence = self.mu * (opinion_1 - opinion_2)
        # non-commitment
        else:
            influence = 0
        new_opinion = opinion_1 + influence
        return min(1,  new_opinion) if new_opinion>0 else \
               max(-1, new_opinion)

    def plot_results(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(13,6))

        cmap = plt.get_cmap('hsv', self.n)
        x = np.arange(self.opinion_history.shape[1])

        for agent in range(self.n):
            ax.plot(x, self.opinion_history[agent, :], color=cmap(agent))