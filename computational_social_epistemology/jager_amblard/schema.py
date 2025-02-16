from typing import Dict, Tuple, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import numpy as np

class AgentProfile(Enum):
    EGOCENTRIC = "egocentric"      # Small gap between U and T - quickly reject different opinions
    MODERATE = "moderate"          # Balanced U and T values
    STUBBORN = "stubborn"          # Low U value - rarely accept different opinions
    INDIFFERENT = "indifferent"    # Large gap between U and T - mostly non-committal
    OPEN_MINDED = "open minded"    # High U value - readily accept different opinions

@dataclass
class AgentType(ABC):
    cluster_center: Tuple[float, float]
    range: Tuple[Callable, Callable]

    @abstractmethod
    def __repr__(self):
        pass

@dataclass
class EgocentricAgent(AgentType):
    cluster_center = (0.3, 0.4)
    range = (
        lambda: np.random.uniform(0, 0.5),       # U
        lambda u: np.random.uniform(u, u + 0.4)  # T
    )

    def __repr__(self):
        return "egocentric"
    
@dataclass
class ModerateAgent(AgentType):
    cluster_center = (0.8, 1.2)
    range = (
        lambda: np.random.uniform(0.6, 1.0),           # U
        lambda u: np.random.uniform(u + 0.2, u + 0.6)  # T
    )

    def __repr__(self):
        return "moderate"

@dataclass
class StubbornAgent(AgentType):
    cluster_center = (0.2, 1.5)
    range = (
        lambda: np.random.uniform(0, 0.3),      # U
        lambda u: np.random.uniform(1.0, 2.0)   # T
    )

    def __repr__(self):
        return "stubborn"

@dataclass
class IndifferentAgent(AgentType):
    cluster_center = (0.5, 1.8)
    range = (
        lambda: np.random.uniform(0.3, 0.7),       # U
        lambda u: np.random.uniform(u + 0.7, 2.0)  # T
    )

    def __repr__(self):
        return "indifferent"

# OpenMindedAgent
@dataclass
class OpenMindedAgent(AgentType):
    cluster_center = (1.5, 1.8)
    range = (
        lambda: np.random.uniform(0.8, 2.0),    # U
        lambda u: np.random.uniform(u, 2.0)     # T
    )

    def __repr__(self):
        return "open minded"

@dataclass
class CommunityConfig:
    """Enhanced configuration class for experiments"""
    name: str
    weights: Dict[AgentProfile, float]
    description: str
    composition_type: str  # To track how this config was generated