from typing import Dict, List, Tuple
from dataclasses import dataclass
from tabulate import tabulate

import numpy as np
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

@dataclass
class PolarizationReport:
    """Class to store polarization metrics and their changes"""
    initial: Dict[str, float]
    final: Dict[str, float]
    changes: Dict[str, float]

def spread(opinions: np.ndarray) -> float:
    """
    Calculate the spread (apertura) of opinions.
    Measures distance between most extreme opinions.
    
    Args:
        opinions: 1D numpy array of opinion values
    """
    return np.max(opinions) - np.min(opinions)

def dispersion(opinions: np.ndarray) -> float:
    """
    Calculate the dispersion of opinions using mean absolute deviation.
    Measures average distance of opinions from the mean.
    
    Args:
        opinions: 1D numpy array of opinion values
    """
    mean_opinion = np.mean(opinions)
    return np.mean(np.abs(opinions - mean_opinion))

def coverage(opinions: np.ndarray, bins: int = 20) -> float:
    """
    Calculate the coverage of opinions across the opinion space.
    Uses histogram to measure what fraction of possible opinion values are represented.
    
    Args:
        opinions: 1D numpy array of opinion values
        bins: Number of bins to divide opinion space into
    """
    hist, _ = np.histogram(opinions, bins=bins, range=(opinions.min(), opinions.max()))
    return np.sum(hist > 0) / bins

def regionalization(opinions: np.ndarray, threshold: float = 0.1) -> float:
    """
    Calculate the regionalization by identifying gaps in opinion distribution.
    Uses kernel density estimation to find regions of low density.
    
    Args:
        opinions: 1D numpy array of opinion values
        threshold: Density threshold below which a region is considered empty
    """
    # Use KDE to estimate opinion density
    kde = gaussian_kde(opinions)
    x = np.linspace(min(opinions), max(opinions), 100)
    density = kde(x)
    
    # Count regions where density falls below threshold
    empty_regions = np.sum(np.diff((density < threshold).astype(int)) == 1)
    return empty_regions

def identify_groups(opinions: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Identify opinion groups using K-means clustering.
    Number of clusters is determined by silhouette score.
    
    Args:
        opinions: 1D numpy array of opinion values
    Returns:
        Tuple of (group labels, number of groups)
    """
    # Reshape for sklearn
    X = opinions.reshape(-1, 1)
    
    # Try different numbers of clusters
    max_clusters = min(len(opinions) // 5, 10)  # Reasonable upper limit
    best_n_clusters = 2
    best_score = -1
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        
        if len(np.unique(labels)) > 1:  # Need at least 2 clusters for silhouette
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
    
    # Final clustering with optimal number of clusters
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    
    return labels, best_n_clusters

def group_distinctness(opinions: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate how distinct the identified groups are.
    Uses ratio of between-cluster to within-cluster variance.
    
    Args:
        opinions: 1D numpy array of opinion values
        labels: Group assignments for each opinion
    """
    if len(np.unique(labels)) < 2:
        return 0.0
        
    group_means = [np.mean(opinions[labels == i]) for i in np.unique(labels)]
    between_var = np.var(group_means)
    within_var = np.mean([np.var(opinions[labels == i]) for i in np.unique(labels)])
    
    if within_var == 0:
        return 1.0
    return between_var / (between_var + within_var)

def group_divergence(opinions: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate average distance between group means.
    
    Args:
        opinions: 1D numpy array of opinion values
        labels: Group assignments for each opinion
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0
        
    group_means = [np.mean(opinions[labels == i]) for i in unique_labels]
    
    # Calculate average pairwise distance between group means
    n_groups = len(group_means)
    total_dist = 0
    count = 0
    
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            total_dist += abs(group_means[i] - group_means[j])
            count += 1
            
    return total_dist / count if count > 0 else 0

def group_consensus(opinions: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate average internal consistency within groups.
    Higher values indicate stronger within-group agreement.
    
    Args:
        opinions: 1D numpy array of opinion values
        labels: Group assignments for each opinion
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 1.0
        
    # Calculate average deviation within each group
    deviations = []
    for label in unique_labels:
        group_opinions = opinions[labels == label]
        group_mean = np.mean(group_opinions)
        group_dev = np.mean(np.abs(group_opinions - group_mean))
        deviations.append(group_dev)
    
    # Transform to 0-1 scale where 1 means perfect consensus
    max_possible_dev = np.ptp(opinions)  # Range of possible opinions
    if max_possible_dev == 0:
        return 1.0
    
    avg_deviation = np.mean(deviations)
    return 1 - (avg_deviation / max_possible_dev)

def size_parity(opinions: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate how evenly sized the groups are using entropy.
    Higher values indicate more equal-sized groups.
    
    Args:
        opinions: 1D numpy array of opinion values
        labels: Group assignments for each opinion
    """
    unique_labels = np.unique(labels)
    n_groups = len(unique_labels)
    
    if n_groups < 2:
        return 0.0
    
    # Calculate proportion of opinions in each group
    props = []
    for label in unique_labels:
        prop = np.sum(labels == label) / len(labels)
        props.append(prop)
    
    # Calculate entropy
    entropy = -np.sum([p * np.log(p) if p > 0 else 0 for p in props])
    max_entropy = np.log(n_groups)
    
    return entropy / max_entropy if max_entropy > 0 else 0

def community_fragmentation(opinions: np.ndarray) -> float:
    """
    Calculate degree of community fragmentation based on identified groups.
    Combines number of groups with their distinctness.
    
    Args:
        opinions: 1D numpy array of opinion values
    """
    labels, n_groups = identify_groups(opinions)
    distinctness = group_distinctness(opinions, labels)
    
    # Normalize number of groups to 0-1 scale (assuming max 10 groups)
    norm_groups = (n_groups - 1) / 9  # -1 because 1 group means no fragmentation
    
    # Combine metrics
    return (norm_groups + distinctness) / 2

def calculate_all_metrics(opinions: np.ndarray) -> Dict[str, float]:
    """
    Calculate all polarization metrics for a given opinion distribution.
    
    Args:
        opinions: 1D numpy array of opinion values
    Returns:
        Dictionary of metric names and values
    """
    # First calculate basic metrics
    metrics = {
        'spread': spread(opinions),
        'dispersion': dispersion(opinions),
        'coverage': coverage(opinions),
        'regionalization': regionalization(opinions),
    }
    
    # Calculate group-based metrics
    labels, _ = identify_groups(opinions)
    
    metrics.update({
        'group_distinctness': group_distinctness(opinions, labels),
        'group_divergence': group_divergence(opinions, labels),
        'group_consensus': group_consensus(opinions, labels),
        'size_parity': size_parity(opinions, labels),
        'community_fragmentation': community_fragmentation(opinions)
    })
    
    return metrics

def generate_polarization_report(model) -> PolarizationReport:
    """
    Generate a comprehensive polarization report comparing initial and final states.
    
    Args:
        model: A simulation model (RA, BC, or JA) that has been run and has a history attribute
    Returns:
        PolarizationReport containing initial, final, and change metrics
    """
    # Get initial and final opinion distributions
    initial_opinions = model.opinion_history[:, 0]
    final_opinions = model.opinion_history[:, -1]
    
    # Calculate metrics for both distributions
    initial_metrics = calculate_all_metrics(initial_opinions)
    final_metrics = calculate_all_metrics(final_opinions)
    
    # Calculate changes
    changes = {
        metric: final_metrics[metric] - initial_metrics[metric]
        for metric in initial_metrics.keys()
    }
    
    return PolarizationReport(
        initial=initial_metrics,
        final=final_metrics,
        changes=changes
    )

def print_polarization_report(report: PolarizationReport) -> None:
    """
    Print a formatted polarization report.
    
    Args:
        report: PolarizationReport object containing metrics
    """
    headers = ['Metric', 'Initial', 'Final', 'Change']
    table = []
    
    for metric in report.initial.keys():
        row = [
            metric,
            f"{report.initial[metric]:.3f}",
            f"{report.final[metric]:.3f}",
            f"{report.changes[metric]:+.3f}"
        ]
        table.append(row)
    
    print("\nPolarization Analysis Report")
    print("===========================")
    print(tabulate(table, headers=headers, tablefmt='grid'))