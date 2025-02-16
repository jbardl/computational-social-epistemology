"""En un futuro, este archivo podría convertirse en un directorio.
Cada simulación que hay en este archivo podría tener su archivo separado,
de manera que en cada archivo quede la simulación principal junto con sus extensiones"""

from typing import List, Tuple
from tqdm import tqdm
from itertools import groupby
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go


class BoundedConfidenceModel:

    def __init__(self,
                 n: int,
                 epsilon: float,
                 epochs: int,
                 early_stopping: int = 5,
                 v: bool = False):
        self.n = n
        self.epsilon = epsilon
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.v = v

        self.history = None # np.ndarray
        self.convergence_epochs = 0


    def run(self) -> None:
        """Update history with updated opinion profile and check convergence"""
        self.history = self.init_params()
        # para cada época de la simulación
        iterator = range(self.epochs-1)
        for _ in (tqdm(iterator) if self.v else iterator):
            # correr update sobre el últio perfil de opinión
            new_opinion_profile = self.update(self.last_opinion_profile)
            # agregar el resultado a la historia
            self.history = np.concatenate([self.history, new_opinion_profile], axis=1)
            # chequear convergencia
            convergence = self.check_convergence()
            if convergence: break


    @property
    def last_opinion_profile(self) -> np.ndarray:
        """Último paso de la simulación"""
        return self.history[:, -1]


    def init_params(self) -> np.ndarray:
        """Inicializa un perfil de opinión y la historia"""
        opinions = np.random.random(size=self.n)
        opinions.sort()
        history = opinions.reshape(-1, 1)
        return history


    def update(self, opinions: np.array) -> np.array:
        """Genera nuevo perfil de opiniones a partir del último.
        Itera por cada opinión, busca las opiniones que más se le parezcan
        y las promedia"""
        new_opinion_profile = []
        for opinion in opinions:
            # selección de aquellos agentes cuya opinión se encuentre dentro del rango
            neighbors = np.where((opinions > opinion-self.epsilon) & \
                                 (opinions < opinion+self.epsilon),
                                 True, False)
            # se promedian las opiniones de los agentes seleccionados
            new_opinion = opinions[neighbors].mean()
            new_opinion_profile.append(new_opinion)

        return np.array(new_opinion_profile).reshape(-1, 1)


    def check_convergence(self) -> bool:
        """Función para chequear si la simulación convergió.
        Ahora esta función es trivial porque no hace falta esperar 5 pasos para chequear convergencia.
        Sin embargo, va a ser útil más adelante si llego a experimentar con introducción de ruido."""
        # silos dos últimos perfiles fueron iguales,
        # suma paso de convergencia
        if all(self.history[:, -2] == self.history[:, -1]):
            self.convergence_epochs += 1
        # si la simulación convergió hace más pasos que el early stopping
        # de vuelve True. En caso contrario, devuelve False
        if self.convergence_epochs > self.early_stopping:
            if self.v: print("Early stopping!")
            return True
        return False


    def plot_results(self, ax=None):
        """Genera gráfico de líneas que muestra la evolución de las opiniones"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(13,6))
            
        cmap = plt.get_cmap('hsv', self.n)
        x = np.arange(self.history.shape[1])

        for agent in range(self.n):
            ax.plot(x, self.history[agent, :], color=cmap(agent))


class BatchSimulaciones:

    def __init__(self,
                 models: List[object],
                 epsilons: np.array,
                 bins: int,
                 v: bool = False):
        self.models = models
        self.epsilons = epsilons
        self.bins = bins
        self.v = v

        self.relative_frequencies = None


    def run_experiments(self) -> None:
        """Inicializa todos los modelos a explorar.
        Ejecuta la simulación correspondiente a cada instancia de modelo.
        Genera estadísticas de los resultados. En particular, para cada conjunto de parámetros
        se promedian los histogramas del último perfil de opinión"""
        updated_models = []
        # se aprovecha procesamiento paralelo para acelerar resultados
        if self.v:
            with Pool(5) as pool:
                with tqdm(total=len(self.models), desc='Running experiments...') as pbar:
                    # ejecución de la simulación para cada instancia
                    for model in pool.imap(BatchSimulaciones.run_model, self.models):
                        pbar.update(1)
                        updated_models.append(model)
        else:
            with Pool(5) as pool:
                for model in pool.imap(BatchSimulaciones.run_model, self.models):
                    updated_models.append(model)
        # creación de estadísticos para todas las simulaciones
        relative_frequencies = []
        # se agrupan las simulaciones con mismos parámetros
        for key, model_group in groupby(updated_models, lambda model: model.epsilon):
            # se crea un histograma con el último perfil de opinión para cada simulación
            histograms = [np.histogram(a       = model.history[:, -1],
                                       bins    = self.bins,
                                       range   = (0., 1.),
                                       density = True)[0]
                          for model in model_group]
            # se calcula el promedio de los valores de cada celda de los histogramas
            mean_relative_frequency = np.array(histograms).mean(axis=0)
            relative_frequencies.append(mean_relative_frequency)
        # almacenamiento de resultados como atributos
        self.models = updated_models
        self.relative_frequencies = relative_frequencies


    @staticmethod
    def run_model(model):
        """Wrapper para ejecutar el método `run` de cada modelo"""
        model.run()
        return model


    def plot_results(self, show=True):
        """Función para generar el gráfico en tres dimensiones."""
        x, y = np.meshgrid(range(self.bins), self.epsilons)
        z = np.array(self.relative_frequencies)

        fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
        fig.update_layout(title='Distribución de opiniones', width=800, height=800)

        if show:
            fig.show()
        else:
            return fig


class AsymmetricBoundedConfidenceModel(BoundedConfidenceModel):

    def __init__(self,
                 n: int,
                 epsilon: Tuple[float, float],
                 epochs: int,
                 early_stopping: int = 5,
                 v: bool = False):
        super().__init__(n, epsilon, epochs, early_stopping, v)
        self.epsilon_left, self.epsilon_right = self.epsilon


    def update(self, opinions: np.array) -> np.array:
        """Genera nuevo perfil de opiniones a partir del último.
        Itera por cada opinión, busca las opiniones que más se le parezcan
        y las promedia"""
        new_opinion_profile = []
        for opinion in opinions:
            neighbors = np.where((opinions > opinion-self.epsilon_left) & \
                                 (opinions < opinion+self.epsilon_right),
                                 True, False)
            new_opinion = opinions[neighbors].mean()
            new_opinion_profile.append(new_opinion)

        return np.array(new_opinion_profile).reshape(-1, 1)


class DependentAsymmetryBoundedConfidenceModel(BoundedConfidenceModel):

    def __init__(self,
                 n: int,
                 strength: float,
                 epochs: int,
                 early_stopping: int = 5,
                 v: bool = False,
                 epsilon: int = None,):
        super().__init__(n, epsilon, epochs, early_stopping, v)
        self.strength = strength


    def update(self, opinions: np.array) -> np.array:
        """Genera nuevo perfil de opiniones a partir del último.
        Itera por cada opinión, busca las opiniones que más se le parezcan
        y las promedia"""
        new_opinion_profile = []
        for opinion in opinions:
            right_bias = self.bias(opinion)
            left_bias = 1-right_bias
            neighbors = np.where((opinions > opinion-left_bias) & \
                                 (opinions < opinion+right_bias),
                                 True, False)
            new_opinion = opinions[neighbors].mean()
            new_opinion_profile.append(new_opinion)
        return np.array(new_opinion_profile).reshape(-1, 1)


    def bias(self, opinion):
        return self.strength * opinion + ((1-self.strength)/2)


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
        self.opinion_history = None


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
            self.opinion_history = np.concatenate([self.opinion_history, new_profile[:, 0].reshape(-1, 1)], axis=1)


    def init_params(self):
        low, high = self.opinion_range
        opinions = np.random.uniform(low=low, high=high, size=(self.n, 1))
        # uncertainties = np.random.uniform(low=0., high=2., size=(self.n, 1))
        uncertainties = np.full(shape=(self.n, 1), fill_value=self.uncertainty)
        self.agents = np.concatenate([opinions, uncertainties], axis=1)
        self.agents.sort(axis=0)
        self.opinion_history = self.agents[:, 0].reshape(-1, 1)


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


    def plot_results(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(13,6))

        cmap = plt.get_cmap('hsv', self.n)
        x = np.arange(self.opinion_history.shape[1])

        for agent in range(self.n):
            ax.plot(x, self.opinion_history[agent, :], color=cmap(agent))


class RAModelExtremists(RelativeAgreementModel):

    def __init__(self,
                 n: int,
                 mu: float,
                 epochs: int,
                 uncertainty: float,
                 uncertainty_extremists: float,
                 global_proportion: float,
                 delta: float,
                 opinion_range: Tuple[float] = (-1., 1.),
                 v: bool = False):
        super().__init__(n, mu, epochs,
                         opinion_range,
                         uncertainty, v)
        self.uncertainty_extremists = uncertainty_extremists
        self.global_proportion = global_proportion
        self.delta = delta

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
            self.opinion_history = np.concatenate([
                self.opinion_history, 
                new_profile[:, 0].reshape(-1, 1)], 
               axis=1)
            self.uncertainty_history = np.concatenate([
                self.uncertainty_history, 
                new_profile[:, 1].reshape(-1, 1)], 
               axis=1)

    def init_params(self):
        low, high = self.opinion_range
        opinions = np.random.uniform(low=low, high=high, size=(self.n, 1))
        opinions.sort(axis=0)
        uncertainties = self.make_uncertainties()
        self.agents = np.concatenate([opinions, uncertainties], axis=1)
        self.opinion_history = self.agents[:, 0].reshape(-1, 1)
        self.uncertainty_history = self.agents[:, 1].reshape(-1, 1)


    def make_uncertainties(self):
        p_pos = p_neg = int(self.n_extremists / 2)
        uncertainties = np.full(shape=(self.n, 1), fill_value=self.uncertainty)
        uncertainties[:p_pos] = self.uncertainty_extremists
        uncertainties[-p_neg:] = self.uncertainty_extremists
        return uncertainties


    @property
    def n_extremists(self):
        n_extremists = int(self.n * self.global_proportion)
        if n_extremists % 2 != 0: n_extremists += 1
        return n_extremists


    def plot_results(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(13,6))

        # Create custom colormap: red -> yellow -> green
        colors = ['red', 'yellow', 'limegreen']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list("uncertainty_cmap", colors, N=n_bins)

        # Get timesteps for x-axis
        timesteps = np.arange(self.opinion_history.shape[1])

        # Find global min and max uncertainty for proper scaling
        min_uncertainty = np.min(self.uncertainty_history)
        max_uncertainty = np.max(self.uncertainty_history)

        all_segments = []

        # Plot each agent's trajectory
        for agent in range(self.n):
            # Get opinion and uncertainty trajectories for this agent
            opinions = self.opinion_history[agent, :]
            uncertainties = self.uncertainty_history[agent, :]

            # Create segments for the line with colors based on uncertainty
            points = np.array([timesteps, opinions]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Normalize uncertainties to [0,1] for coloring
            normalized_uncertainties = (uncertainties[:-1] - min_uncertainty) / (max_uncertainty - min_uncertainty)

            # Create line collection with varying colors
            lc = plt.matplotlib.collections.LineCollection(
                segments, cmap=cmap,
                norm=plt.Normalize(0, 1),
                linewidth=1
            )
            lc.set_array(normalized_uncertainties)
            ax.add_collection(lc)

        ax.set_xlim(timesteps.min(), timesteps.max())
        ax.set_ylim(-1.0, 1.0)