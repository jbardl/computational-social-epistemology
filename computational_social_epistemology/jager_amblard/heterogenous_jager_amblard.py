from .schema import AgentProfile
from .samplers import sample_agent_types
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
import numpy as np


class HeterogeneousJAModel:
    def __init__(self, n, mu, epochs, verbose=True):
        """
        Initialize the Heterogeneous Jager-Amblard model.
        
        Parameters:
        n (int): Number of agents
        mu (float): Learning rate / influence factor
        epochs (int): Maximum number of simulation steps
        verbose (bool): Whether to show progress bar
        """
        self.n = n
        self.mu = mu
        self.epochs = epochs
        self.verbose = verbose
        
        # Will store the full opinion_history of opinions
        self.opinion_history = None
        # Will store the agent types (U and T values)
        self.agent_types = None

    @property
    def last_opinion_profile(self):
        """Get the most recent opinion state"""
        return self.opinion_history[:, -1]

    def init_params(self):
        """Initialize model parameters with heterogeneous agent types"""
        # Sample initial opinions uniformly
        opinions = np.random.uniform(low=-1, high=1, size=self.n)
        opinions.sort()
        self.opinion_history = opinions.reshape(-1, 1)
        
        # Sample agent types (U and T values)
        if self.agent_types is None:
            self.agent_types = sample_agent_types(self.n)

    def run(self):
        """Run the simulation"""
        self.init_params()
        
        iterator = range(self.epochs)
        if self.verbose:
            iterator = tqdm(iterator)
            
        for epoch in iterator:
            # Update based on last opinion profile
            new_opinion_profile = self.update(self.last_opinion_profile)
            # Add result to opinion_history
            self.opinion_history = np.concatenate([self.opinion_history, new_opinion_profile], axis=1)
            # Check convergence
            if self.check_convergence():
                break

    def update(self, opinion_profile):
        """Update all agents' opinions through random pairwise interactions"""
        # Generate random pairs of agents
        random_pairs_idxs = np.random.choice(a=self.n, replace=False,
                                           size=(int(self.n / 2), 2))
        
        # Get opinions and agent types for the random pairs
        random_pairs_opinions = opinion_profile[random_pairs_idxs]
        random_pairs_types = self.agent_types[random_pairs_idxs]
        
        # Update each pair
        update_result = np.array([
            self.pairwise_update(opinions, types)
            for opinions, types in zip(random_pairs_opinions, random_pairs_types)
        ])
        
        # Create new opinion profile
        new_profile = np.empty((self.n))
        new_profile[random_pairs_idxs.reshape(-1)] = update_result.reshape(-1)
        return new_profile.reshape(-1, 1)

    def pairwise_update(self, pair_opinions, pair_types):
        """Update opinions for a pair of agents considering their types"""
        opinion_1, opinion_2 = pair_opinions
        type_1, type_2 = pair_types  # Each type is (U, T)
        
        new_opinion_1 = self.individual_update(opinion_1, opinion_2, type_1)
        new_opinion_2 = self.individual_update(opinion_2, opinion_1, type_2)
        
        return np.array([new_opinion_1, new_opinion_2])

    def individual_update(self, opinion_1, opinion_2, agent_type):
        """
        Update an individual agent's opinion based on interaction
        considering their personal U and T values
        """
        u, t = agent_type  # Unpack agent's personal thresholds
        difference = np.abs(opinion_1 - opinion_2)
        
        # Assimilation - within latitude of acceptance
        if difference < u:
            influence = self.mu * (opinion_2 - opinion_1)
        # Contrast - within latitude of rejection
        elif difference > t:
            influence = self.mu * (opinion_1 - opinion_2)
        # Non-commitment - between thresholds
        else:
            influence = 0
            
        new_opinion = opinion_1 + influence
        return min(1, max(-1, new_opinion))  # Bound opinion to [-1, 1]

    def check_convergence(self):
        """Check if opinions have stabilized"""
        return np.all(self.opinion_history[:, -2] == self.last_opinion_profile)

    def plot_results(self, show_types=True):
        """
        Plot opinion trajectories and optionally agent types
        """
        if show_types:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), 
                                           width_ratios=[3, 1])
        else:
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
        # Plot opinion trajectories
        cmap = plt.get_cmap('hsv', self.n)
        x = np.arange(self.opinion_history.shape[1])
        
        for agent in range(self.n):
            ax1.plot(x, self.opinion_history[agent, :], 
                     color=cmap(agent), 
                     alpha=0.5, 
                     linewidth=1)
        ax1.set_title('Opinion Trajectories')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Opinion')
        
        if show_types:
            # Plot agent types
            ax2 = self.plot_agent_types(ax2)
            
        plt.tight_layout()
        return fig
    
    def plot_agent_types(self, ax):
        ax.scatter(self.agent_types[:, 0], self.agent_types[:, 1], s=2)
        ax.plot([0, 2], [0, 2], 'r--')  # Reference line y=x
        ax.set_xlabel('Latitude of Acceptance (U)')
        ax.set_ylabel('Latitude of Rejection (T)')
        ax.set_title('Agent Types Distribution')
        ax.grid(True)
        return ax

    def plot_opinion_trajectories_by_type(
        self,
        figsize=(12, 6)
    ) -> plt.Figure:
        """
        Plot opinion trajectories with colors based on agent types using matplotlib.
        
        Parameters:
        opinion_history: np.ndarray of shape (n_agents, n_timesteps) containing opinion histories
        agent_types: np.ndarray of shape (n_agents, 2) containing U,T values for each agent
        weights: Dictionary mapping AgentProfile to their proportions in the population
        figsize: Tuple specifying the figure size
        
        Returns:
        plt.Figure: Matplotlib figure object
        """
        # Define colors for each agent type
        colors = {
            AgentProfile.EGOCENTRIC: "#1E90FF",   # Dodger blue
            AgentProfile.MODERATE: "#9370DB",     # Medium purple
            AgentProfile.STUBBORN: "#FF4500",     # Orange red
            AgentProfile.INDIFFERENT: "#FF1493",  # Deep pink
            AgentProfile.OPEN_MINDED: "#32CD32"   # Lime green
        }
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get time steps
        timesteps = np.arange(self.opinion_history.shape[1])
        
        # Function to determine agent type from U,T values
        def get_agent_type(u: float, t: float) -> AgentProfile:
            if 0 <= u < 0.5 and 0 <= t < 1.0:
                return AgentProfile.EGOCENTRIC
            elif 0.5 <= u < 1.0 and 0.5 <= t < 1.0:
                return AgentProfile.MODERATE
            elif 0 <= u < 0.3 and 1.0 <= t <= 2.0:
                return AgentProfile.STUBBORN
            elif 0.3 <= u < 0.8 and 1.0 <= t <= 2.0:
                return AgentProfile.INDIFFERENT
            elif 0.8 <= u <= 2.0 and 1.0 <= t <= 2.0:
                return AgentProfile.OPEN_MINDED
            else:
                return None  # For points outside defined regions
        
        # Plot trajectories for each agent
        type_counts = {profile: 0 for profile in AgentProfile}
        
        for i, (u, t) in enumerate(self.agent_types):
            agent_type = get_agent_type(u, t)
            if agent_type:
                type_counts[agent_type] += 1
                ax.plot(timesteps, self.opinion_history[i], 
                        color=colors[agent_type], 
                        alpha=0.5, 
                        linewidth=1)
        
        # Calculate proportions for title
        total_agents = sum(type_counts.values())
        type_proportions = {
            profile: count/total_agents 
            for profile, count in type_counts.items() 
            if count > 0
        }
        
        # Create legend elements
        legend_elements = [
            Line2D([0], [0], color=colors[profile], 
                   label=f'{profile.value.title()} ({count/total_agents:.1%})')
            for profile, count in type_counts.items()
            if count > 0
        ]
        
        # Customize plot
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Opinion')
        ax.set_title('Opinion Trajectories by Agent Type')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_ylim(-1.1, 1.1)
        
        # Add legend
        ax.legend(handles=legend_elements, 
                  bbox_to_anchor=(0.5, 1.15),
                  loc='center', 
                  ncol=len(legend_elements),
                  frameon=True,
                  fancybox=True,
                  shadow=True)
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()
        
        return fig