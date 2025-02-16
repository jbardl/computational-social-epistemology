from .samplers import sample_agent_types
from .config_generation import generate_systematic_configs
from .heterogenous_jager_amblard import HeterogeneousJAModel

import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm

class OpinionDistributionAnalyzer:
    def __init__(self, n_agents=100, mu=0.3, epochs=1000, n_experiments=30,
                 opinion_bins=100):
        """
        Initialize the analyzer.
        
        Parameters:
        n_agents: Number of agents per experiment
        mu: Learning rate
        epochs: Maximum epochs per experiment
        n_experiments: Number of times to repeat each experiment
        opinion_bins: Number of bins for discretizing opinion space
        """
        self.n_agents = n_agents
        self.mu = mu
        self.epochs = epochs
        self.n_experiments = n_experiments
        self.opinion_bins = opinion_bins
        
        # Generate systematic configurations
        self.configs = generate_systematic_configs()
        
        # Create opinion space bins
        self.opinion_space = np.linspace(-1, 1, opinion_bins)
        
    def run_analysis(self):
        """Run experiments and collect distribution data"""
        n_configs = len(self.configs)
        
        # Initialize array to store results
        self.distributions = np.zeros((n_configs, self.opinion_bins))
        
        # Run experiments for each configuration
        for config_idx, config in enumerate(tqdm(self.configs, desc="Analyzing configurations")):
            config_distributions = []
            
            # Run multiple experiments for this configuration
            for _ in range(self.n_experiments):
                model = HeterogeneousJAModel(
                    n=self.n_agents,
                    mu=self.mu,
                    epochs=self.epochs,
                    verbose=False
                )
                model.agent_types = sample_agent_types(
                    self.n_agents,
                    distribution="weighted",
                    profile_weights=config.weights
                )
                model.run()
                
                # Get final opinion distribution
                final_opinions = model.last_opinion_profile
                hist, _ = np.histogram(final_opinions, bins=self.opinion_bins, 
                                       range=(-1, 1), density=True)
                config_distributions.append(hist)
            
            # Average the distributions for this configuration
            self.distributions[config_idx] = np.mean(config_distributions, axis=0)

    def _count_opinion_clusters(self, opinions, threshold=0.1):
        """Count number of distinct opinion clusters"""
        sorted_opinions = np.sort(opinions)
        gaps = np.diff(sorted_opinions)
        return 1 + np.sum(gaps > threshold)
    
    def _calculate_polarization(self, opinions):
        """Calculate polarization as variance of opinions"""
        return np.var(opinions)
    
    def _calculate_consensus(self, opinions, threshold=0.1):
        """Calculate degree of consensus as proportion of opinions within threshold"""
        mean_opinion = np.mean(opinions)
        return np.mean(np.abs(opinions - mean_opinion) < threshold)
    
    def plot_distribution_3d(self):
        """Create interactive 3D visualization of opinion distributions"""
        # Create meshgrid for 3D surface
        x_opinions = np.linspace(-1, 1, self.opinion_bins)
        y_configs = np.arange(len(self.configs))
        X, Y = np.meshgrid(x_opinions, y_configs)
        
        # Create figure
        fig = go.Figure()
        
        # Add surface plot
        fig.add_trace(go.Surface(
            x=X,
            y=Y,
            z=self.distributions,
            colorscale='Viridis',
            colorbar=dict(
                title='Density',
                titleside='right'
            ),
            hovertemplate=(
                'Opinion: %{x:.2f}<br>' +
                'Configuration: %{y}<br>' +
                'Density: %{z:.2f}<extra></extra>'
            )
        ))
        
        # Update layout
        fig.update_layout(
            title='Opinion Distribution Across Different Community Compositions',
            scene=dict(
                xaxis_title='Opinion Space',
                yaxis_title='Configuration Index',
                yaxis=dict(
                    ticktext=[config.name for config in self.configs],
                    tickvals=list(range(len(self.configs))),
                    tickmode='array'
                ),
                zaxis_title='Density',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            width=1000,
            height=800
        )
        
        return fig
    
    def plot_distribution_heatmap(self):
        """Create interactive 2D heatmap visualization"""
        fig = go.Figure()
        
        # Add heatmap
        fig.add_trace(go.Heatmap(
            z=self.distributions,
            x=np.linspace(-1, 1, self.opinion_bins),
            y=[config.name for config in self.configs],
            colorscale='Viridis',
            colorbar=dict(
                title='Density',
                titleside='right'
            ),
            hoverongaps=False,
            hovertemplate=(
                'Opinion: %{x:.2f}<br>' +
                'Configuration: %{y}<br>' +
                'Density: %{z:.2f}<extra></extra>'
            )
        ))
        
        # Update layout
        fig.update_layout(
            title='Opinion Distribution Heatmap',
            xaxis_title='Opinion Space',
            yaxis_title='Community Configuration',
            yaxis=dict(
                tickmode='array',
                ticktext=[config.name for config in self.configs],
                tickvals=list(range(len(self.configs)))
            ),
            width=1000,
            height=800
        )
        
        return fig
    
    def plot_distribution_ridgeline(self):
        """Create a ridgeline plot showing distributions for each configuration"""
        fig = go.Figure()
        
        # Calculate offset for ridgeline effect
        max_density = np.max(self.distributions)
        offset_step = max_density * 0.3
        
        # Add traces for each configuration
        for i, config in enumerate(self.configs):
            offset = i * offset_step
            
            # Add filled area
            fig.add_trace(go.Scatter(
                x=np.linspace(-1, 1, self.opinion_bins),
                y=self.distributions[i] + offset,
                fill='tonexty',
                name=config.name,
                mode='none',
                fillcolor=f'rgba(0, 100, 180, 0.2)',
                showlegend=True,
                hovertemplate=(
                    'Opinion: %{x:.2f}<br>' +
                    'Density: %{y:.2f}<br>' +
                    f'Config: {config.name}<extra></extra>'
                )
            ))
        
        # Update layout
        fig.update_layout(
            title='Opinion Distributions by Community Configuration (Ridgeline Plot)',
            xaxis_title='Opinion Space',
            yaxis_title='Density (offset for visibility)',
            showlegend=True,
            height=1000,
            width=1000,
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        
        return fig
