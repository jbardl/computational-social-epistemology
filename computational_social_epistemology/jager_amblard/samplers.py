from .schema import AgentProfile

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import numpy as np
import pandas as pd

def sample_agent_types(num_points, distribution="uniform", profile_weights=None):
    """
    Generate agent types (U,T) using different sampling strategies.
    
    Parameters:
    num_points (int): Number of agents to generate
    distribution (str): Sampling strategy to use:
        - "uniform": Uniform distribution across all valid types
        - "gaussian": Gaussian clusters around profile centers
        - "weighted": Weighted sampling from different agent profiles
    profile_weights (dict): When using "weighted" distribution, specifies the 
        proportion of each agent profile. Should sum to 1.
    
    Returns:
    np.ndarray: Array of shape (num_points, 2) with sampled points (U,T)
    """
    if distribution == "uniform":
        return _sample_uniform(num_points)
    
    elif distribution == "gaussian":
        return _sample_gaussian_clusters(num_points)
    
    elif distribution == "weighted":
        if profile_weights is None:
            profile_weights = {
                AgentProfile.EGOCENTRIC: 0.2,
                AgentProfile.OPEN_MINDED: 0.2,
                AgentProfile.MODERATE: 0.2,
                AgentProfile.STUBBORN: 0.2,
                AgentProfile.INDIFFERENT: 0.2
            }
        return _sample_weighted_profiles(num_points, profile_weights)
    
    else:
        raise ValueError(f"Unknown distribution type: {distribution}")

def _sample_uniform(num_points):
    """Original uniform sampling strategy"""
    x = np.random.uniform(0, 2, num_points)
    y = np.random.uniform(x, 2)
    return np.column_stack((x, y))

def _sample_gaussian_clusters(num_points, noise=0.1):
    """
    Sample points in Gaussian clusters around typical agent profiles.
    Adds some noise to create more natural distributions.
    """
    # Define cluster centers (U, T) for each profile
    centers = {
        AgentProfile.EGOCENTRIC: (0.3, 0.4),     # Small gap
        AgentProfile.MODERATE: (0.8, 1.2),       # Balanced
        AgentProfile.STUBBORN: (0.2, 1.5),       # Low acceptance
        AgentProfile.INDIFFERENT: (0.5, 1.8),    # Large gap
        AgentProfile.OPEN_MINDED: (1.5, 1.8)    # High acceptance
    }
    
    # Sample equal numbers from each cluster
    points_per_cluster = num_points // len(centers)
    remainder = num_points % len(centers)
    
    samples = []
    for center in centers.values():
        # Generate Gaussian samples around center
        cluster_size = points_per_cluster + (remainder > 0)
        remainder = max(0, remainder - 1)
        
        u = np.random.normal(center[0], noise, cluster_size)
        t = np.random.normal(center[1], noise, cluster_size)
        
        # Ensure constraints are met
        u = np.clip(u, 0, 2)
        t = np.clip(t, u, 2)  # T must be greater than U
        
        samples.append(np.column_stack((u, t)))
    
    return np.vstack(samples)

def _sample_weighted_profiles(num_points, profile_weights):
    """
    Sample points based on specified weights for each agent profile.
    """
    # Verify weights sum to 1
    if not np.isclose(sum(profile_weights.values()), 1.0):
        raise ValueError("Profile weights must sum to 1")
    
    # Define typical ranges for each profile
    profile_ranges = {
        AgentProfile.EGOCENTRIC: (
            lambda: np.random.uniform(0, 0.5),       # U
            lambda u: np.random.uniform(u, 1.0)  # T
        ),
        AgentProfile.MODERATE: (
            lambda: np.random.uniform(0.5, 1.0),           # U
            lambda u: np.random.uniform(u, 1.0)  # T
        ),
        AgentProfile.STUBBORN: (
            lambda: np.random.uniform(0, 0.3),      # U
            lambda u: np.random.uniform(1.0, 2.0)   # T
        ),
        AgentProfile.INDIFFERENT: (
            lambda: np.random.uniform(0.3, 0.8),       # U
            lambda u: np.random.uniform(1.0, 2.0)  # T
        ),
        AgentProfile.OPEN_MINDED: (
            lambda: np.random.uniform(0.8, 2.0),    # U
            lambda u: np.random.uniform(max(u, 1.0), 2.0)     # T
        )
    }
    
    # Calculate number of agents for each profile
    profile_counts = {
        profile: int(weight * num_points)
        for profile, weight in profile_weights.items()
    }
    
    # Adjust for rounding errors
    total = sum(profile_counts.values())
    if total < num_points:
        # Add remaining points to the profile with highest weight
        max_profile = max(profile_weights.items(), key=lambda x: x[1])[0]
        profile_counts[max_profile] += num_points - total
    
    # Generate samples for each profile
    samples = []
    for profile, count in profile_counts.items():
        u_sampler, t_sampler = profile_ranges[profile]
        
        for _ in range(count):
            u = u_sampler()
            t = t_sampler(u)
            samples.append([u, t])

    samples = np.array(samples)
    np.random.shuffle(samples)
    
    return samples

def plot_agent_type_regions(n_points=1000, figsize=(12, 10)):
    """
    Create a visualization of agent type regions with uniform sample points.
    
    Parameters:
    n_points (int): Number of sample points to generate
    figsize (tuple): Size of the figure
    """
    # Generate uniform sample of points
    uniform_points = sample_agent_types(n_points, distribution="uniform")
    
    # Define regions for each agent type as polygon vertices
    regions = {
        AgentProfile.EGOCENTRIC: np.array([
            [0, 0], [0.5, 0.5], [0.5, 1.0], [0, 1.0]
        ]),
        AgentProfile.MODERATE: np.array([
            [0.5, 0.5], [1.0, 1.0], [1.0, 1.0], [0.5, 1.0]
        ]),
        AgentProfile.STUBBORN: np.array([
            [0, 1.0], [0.3, 1.0], [0.3, 2.0], [0, 2.0]
        ]),
        AgentProfile.INDIFFERENT: np.array([
            [0.3, 1.0], [0.8, 1.0], [0.8, 2.0], [0.3, 2.0]
        ]),
        AgentProfile.OPEN_MINDED: np.array([
            [0.8, 1.0], [1.0, 1.0], [2.0, 2.0], [0.8, 2.0]
        ])
    }
    
    # Define colors for each region
    colors = {
        AgentProfile.INDIFFERENT: "#FFB6C1",  # Light pink
        AgentProfile.STUBBORN: "#FFD700",     # Gold
        AgentProfile.OPEN_MINDED: "#98FB98",  # Pale green
        AgentProfile.EGOCENTRIC: "#87CEEB",   # Sky blue
        AgentProfile.MODERATE: "#DDA0DD"      # Plum
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot reference line y=x
    ax.plot([0, 2], [0, 2], 'k--', alpha=0.5, label='T=U line')
    
    # Plot uniform sample points
    ax.scatter(uniform_points[:, 0], uniform_points[:, 1], 
              c='gray', alpha=0.2, s=20, label='Sample points')
    
    # Plot regions
    for profile, vertices in regions.items():
        polygon = Polygon(vertices, alpha=0.3, 
                          facecolor=colors[profile], 
                          edgecolor='black', 
                          label=profile.value.title())
        ax.add_patch(polygon)
        
        # Add text label in center of region
        center = vertices.mean(axis=0)
        ax.text(center[0], center[1], profile.value.upper(),
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=10)
    
    # Customize plot
    ax.set_xlabel('Latitude of Acceptance (U)', fontsize=12)
    ax.set_ylabel('Latitude of Rejection (T)', fontsize=12)
    ax.set_title('Agent Type Regions in U-T Space', fontsize=14, pad=20)
    
    # Set axis limits
    ax.set_xlim(-0.1, 2.1)
    ax.set_ylim(-0.1, 2.1)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add legend
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    return fig