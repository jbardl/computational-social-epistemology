from .schema import AgentProfile, CommunityConfig
from typing import List
from itertools import combinations
import pandas as pd
import plotly.graph_objects as go

def generate_systematic_configs() -> List[CommunityConfig]:
    """
    Generate systematic configurations exploring different agent type combinations.
    
    Returns:
    List[CommunityConfig]: List of all possible configurations
    """
    configs = []
    profiles = list(AgentProfile)
    
    # Generate configs for different numbers of agent types
    for n_types in range(1, len(profiles) + 1):
        # Get all possible combinations of n_types agent types
        for type_combination in combinations(profiles, n_types):
            # Equal distribution among selected types
            weight = 1.0 / len(type_combination)
            weights = {profile: 0.0 for profile in profiles}  # Initialize all to 0
            for profile in type_combination:
                weights[profile] = weight
            
            # Create readable name and description
            profile_names = [p.value.title() for p in type_combination]
            name = f"{n_types}-Type: " + "/".join(profile_names)
            description = f"Equal distribution ({weight:.2%} each) among: {', '.join(profile_names)}"
            
            configs.append(CommunityConfig(
                name=name,
                weights=weights,
                description=description,
                composition_type=f"{n_types}-type-equal"
            ))
    
    return configs

def print_config_summary(configs: List[CommunityConfig]):
    """Print a summary of all configurations"""
    # Convert to DataFrame for easier viewing
    rows = []
    for config in configs:
        row = {
            'Name': config.name,
            'Type': config.composition_type,
            'Description': config.description
        }
        # Add columns for each agent type's weight
        for profile in AgentProfile:
            row[profile.value.title()] = config.weights[profile]
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Reorder columns
    cols = ['Name', 'Type', 'Description'] + [p.value.title() for p in AgentProfile]
    df = df[cols]
    
    return df

def plot_config_distributions(configs: List[CommunityConfig]):
    """
    Create a visualization showing the distribution of agent types in each configuration
    """
    fig = go.Figure()
    
    # Get all profile names
    profiles = list(AgentProfile)
    profile_names = [p.value.title() for p in profiles]
    
    # Create traces for each agent type
    for profile in profiles:
        weights = [config.weights[profile] for config in configs]
        fig.add_trace(go.Bar(
            name=profile.value.title(),
            x=[config.name for config in configs],
            y=weights,
            text=[f'{w:.1%}' if w > 0 else '' for w in weights],
            textposition='inside',
        ))
    
    # Update layout to stack bars
    fig.update_layout(
        barmode='stack',
        title='Agent Type Distribution in Each Configuration',
        xaxis_title='Configuration',
        yaxis_title='Proportion of Agents',
        yaxis=dict(
            tickformat=',.0%',
            range=[0, 1]
        ),
        showlegend=True,
        legend_title='Agent Type',
        height=600
    )
    
    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=45)
    
    return fig