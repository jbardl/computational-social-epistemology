from .samplers import sample_agent_types
from .heterogenous_jager_amblard import HeterogeneousJAModel
from tqdm import tqdm
import numpy as np

class HeterogeneousJAModelSensationalism(HeterogeneousJAModel):
    def __init__(self, n, sensationalism_levels, epochs, verbose=True):
        """
        Initialize the Heterogeneous Jager-Amblard model with agent-specific sensationalism.
        
        Parameters:
        n (int): Number of agents
        sensationalism_levels (np.ndarray): Array of shape (n,) containing mu values for each agent
        epochs (int): Maximum number of simulation steps
        verbose (bool): Whether to show progress bar
        """
        super().__init__(n=n, 
                         mu=0.1, # dummy value for mu parameter
                         epochs=epochs, 
                         verbose=verbose)
        
        if len(sensationalism_levels) != n:
            raise ValueError(f"sensationalism_levels must have length {n}")
        self.sensationalism_levels = sensationalism_levels
        
        # Will store the full history of opinions
        self.history = None
        # Will store the agent types (U and T values)
        self.agent_types = None

    def update(self, opinion_profile):
        """Update all agents' opinions through random pairwise interactions"""
        # Generate random pairs of agents
        random_pairs_idxs = np.random.choice(a=self.n, replace=False,
                                           size=(int(self.n / 2), 2))
        
        # Get opinions, agent types, and sensationalism levels for the random pairs
        random_pairs_opinions = opinion_profile[random_pairs_idxs]
        random_pairs_types = self.agent_types[random_pairs_idxs]
        random_pairs_sensationalism = self.sensationalism_levels[random_pairs_idxs]
        
        # Update each pair
        update_result = np.array([
            self.pairwise_update(opinions, types, sensationalism)
            for opinions, types, sensationalism in zip(
                random_pairs_opinions, 
                random_pairs_types,
                random_pairs_sensationalism
            )
        ])
        
        # Create new opinion profile
        new_profile = np.empty((self.n))
        new_profile[random_pairs_idxs.reshape(-1)] = update_result.reshape(-1)
        return new_profile.reshape(-1, 1)

    def pairwise_update(self, pair_opinions, pair_types, pair_sensationalism):
        """
        Update opinions for a pair of agents considering their types and sensationalism levels
        """
        opinion_1, opinion_2 = pair_opinions
        type_1, type_2 = pair_types  # Each type is (U, T)
        mu_1, mu_2 = pair_sensationalism  # Sensationalism levels for each agent
        
        new_opinion_1 = self.individual_update(opinion_1, opinion_2, type_1, mu_2)  # Agent 1 is influenced by agent 2's sensationalism
        new_opinion_2 = self.individual_update(opinion_2, opinion_1, type_2, mu_1)  # Agent 2 is influenced by agent 1's sensationalism
        
        return np.array([new_opinion_1, new_opinion_2])

    def individual_update(self, opinion_1, opinion_2, agent_type, other_sensationalism):
        """
        Update an individual agent's opinion based on interaction,
        considering their personal U and T values and the other agent's sensationalism level
        """
        u, t = agent_type  # Unpack agent's personal thresholds
        difference = np.abs(opinion_1 - opinion_2)
        
        # Assimilation - within latitude of acceptance
        if difference < u:
            influence = other_sensationalism * (opinion_2 - opinion_1)
        # Contrast - within latitude of rejection
        elif difference > t:
            influence = other_sensationalism * (opinion_1 - opinion_2)
        # Non-commitment - between thresholds
        else:
            influence = 0
            
        new_opinion = opinion_1 + influence
        return min(1, max(-1, new_opinion))  # Bound opinion to [-1, 1]