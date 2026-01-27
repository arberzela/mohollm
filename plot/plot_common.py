"""
Common utilities and shared functionality for plotting scripts.

This module provides:
- Matplotlib configuration presets
- Common data processing functions (HV computation, normalization, etc.)
- File and method name extraction utilities
- Reusable plotting helpers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from typing import Dict, List, Tuple, Optional, Any
import os
import glob


# ============================================================================
# MATPLOTLIB CONFIGURATION PRESETS
# ============================================================================

class PlotStyle:
    """Centralized matplotlib styling configurations."""
    
    # Default style for most plots
    DEFAULT = {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
    
    # Style for publication-quality plots
    PUBLICATION = {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 22,
        "legend.fontsize": 14,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "axes.linewidth": 1.2,
        "grid.linewidth": 0.6,
        "lines.linewidth": 2.8,
        "patch.linewidth": 1.0,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
    }
    
    # Style for subplots
    SUBPLOT = {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 20,
        "legend.fontsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    }
    
    # Style for utility plots with more data
    UTILITY = {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.figsize": (8, 4),
    }
    
    @staticmethod
    def apply(style: str = "default") -> None:
        """Apply a matplotlib style preset."""
        style_map = {
            "default": PlotStyle.DEFAULT,
            "publication": PlotStyle.PUBLICATION,
            "subplot": PlotStyle.SUBPLOT,
            "utility": PlotStyle.UTILITY,
        }
        plt.rcParams.update(style_map.get(style.lower(), PlotStyle.DEFAULT))


# ============================================================================
# HYPERVOLUME COMPUTATION
# ============================================================================

def compute_hypervolume(df: pd.DataFrame, reference_point: List[float]) -> float:
    """
    Compute hypervolume for a set of objectives.
    
    Args:
        df: DataFrame containing objective values
        reference_point: Reference point for HV calculation
        
    Returns:
        Hypervolume value
    """
    objectives = df.to_numpy()
    ind = HV(ref_point=reference_point)
    hypervolume = ind(objectives)
    return hypervolume


def convert_data_to_hv_over_time(
    fvals: pd.DataFrame, 
    reference_point: List[float] = [1.0, 1.0]
) -> List[float]:
    """
    Convert objective values to hypervolume trajectory over time.
    
    Args:
        fvals: DataFrame containing objective values
        reference_point: Reference point for HV calculation
        
    Returns:
        List of hypervolume values at each step
    """
    hypervolume = []
    for step in range(1, len(fvals) + 1):
        hv = compute_hypervolume(fvals.iloc[:step], reference_point)
        hypervolume.append(hv)
    return hypervolume


def convert_data_to_log_hv_diff_over_time(
    fvals: pd.DataFrame,
    reference_point: List[float],
    max_hv: float
) -> np.ndarray:
    """
    Compute log hypervolume difference trajectory.
    
    Args:
        fvals: DataFrame containing objective values
        reference_point: Reference point for HV calculation
        max_hv: Maximum achievable hypervolume
        
    Returns:
        Array of log HV differences
    """
    hv = []
    for step in range(1, len(fvals) + 1):
        hv_val = compute_hypervolume(fvals.iloc[:step], reference_point)
        hv.append(hv_val)
    log_diff = np.log10(np.maximum((max_hv - np.array(hv)), 1e-8))
    return log_diff


# ============================================================================
# DATA NORMALIZATION
# ============================================================================

def normalize_data(
    fvals: pd.DataFrame, 
    min_max_metrics: Dict[str, Tuple[float, float]]
) -> pd.DataFrame:
    """
    Normalize data using min-max scaling.
    
    Args:
        fvals: DataFrame containing values to normalize
        min_max_metrics: Dictionary mapping column names to (min, max) tuples
        
    Returns:
        Normalized DataFrame
    """
    for column, values in min_max_metrics.items():
        min_val, max_val = values
        if max_val - min_val != 0:
            fvals[column] = (fvals[column] - min_val) / (max_val - min_val)
    return fvals


# ============================================================================
# FILE AND METHOD NAME UTILITIES
# ============================================================================

def extract_method_name(files: List[str]) -> List[str]:
    """
    Extract method names from file paths.
    
    Assumes file structure: .../method_name/seed_X/filename.csv
    
    Args:
        files: List of file paths
        
    Returns:
        List of unique method names
    """
    methods = []
    for file in files:
        parts = file.split(os.sep)
        # Look for the method name (typically 2 levels up from the file)
        if len(parts) >= 3:
            method = parts[-3]
            if method not in methods:
                methods.append(method)
    return methods


def group_data_by_method(
    observed_fvals_files: List[str], 
    filter_str: Optional[str] = None
) -> Dict[str, List[str]]:
    """
    Group files by method name.
    
    Args:
        observed_fvals_files: List of file paths
        filter_str: Optional filter string to include only matching files
        
    Returns:
        Dictionary mapping method names to lists of file paths
    """
    method_names = extract_method_name(observed_fvals_files)
    method_files = {}
    
    for method in method_names:
        files = [f for f in observed_fvals_files if f"/{method}/" in f]
        if filter_str:
            files = [f for f in files if filter_str in f]
        if files:
            method_files[method] = files
            
    return method_files


def custom_sort_key(method_name: str) -> Tuple[int, str]:
    """
    Custom sorting key to prioritize MOHOLLM and LLM methods.
    
    Args:
        method_name: Name of the method
        
    Returns:
        Tuple for sorting (priority, name)
    """
    if "MOHOLLM" in method_name:
        return (-2, method_name)
    if "LLM" == method_name or "LLM" in method_name:
        return (-1, method_name)
    return (0, method_name)


# ============================================================================
# DATA PROCESSING
# ============================================================================

def process_trials_data(
    method_files: Dict[str, List[str]],
    columns: Optional[List[str]] = None,
    trials: Optional[int] = None,
    normalize: bool = False,
    min_max_metrics: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Process trial data for multiple methods.
    
    Args:
        method_files: Dictionary mapping method names to file lists
        columns: Columns to read from CSV files
        trials: Maximum number of trials to include
        normalize: Whether to normalize the data
        min_max_metrics: Normalization parameters
        
    Returns:
        Dictionary with 'mean' and 'std' statistics for each method
    """
    data = {"mean": {}, "std": {}}
    
    for method_name, files in method_files.items():
        all_data = []
        
        for file in files:
            df = pd.read_csv(file)
            if columns:
                df = df[columns]
            if trials:
                df = df.iloc[:trials]
            if normalize and min_max_metrics:
                df = normalize_data(df, min_max_metrics)
            all_data.append(df.values)
        
        if all_data:
            all_data = np.array(all_data)
            data["mean"][method_name] = np.mean(all_data, axis=0)
            data["std"][method_name] = np.std(all_data, axis=0)
    
    return data


# ============================================================================
# PLOTTING UTILITIES
# ============================================================================

def apply_axis_style(ax: plt.Axes) -> None:
    """
    Apply consistent styling to a matplotlib axis.
    
    Args:
        ax: Matplotlib axis object
    """
    ax.grid(True, linestyle="--", alpha=0.5, linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)


def set_axis_limits(
    ax: plt.Axes,
    x_lim: Optional[Tuple[float, float]] = None,
    y_lim: Optional[Tuple[float, float]] = None
) -> None:
    """
    Set axis limits if provided.
    
    Args:
        ax: Matplotlib axis object
        x_lim: Optional x-axis limits
        y_lim: Optional y-axis limits
    """
    if x_lim and all(x != "" and x is not None for x in x_lim):
        ax.set_xlim(x_lim)
    if y_lim and all(y != "" and y is not None for y in y_lim):
        ax.set_ylim(y_lim)


def save_plot(
    fig: plt.Figure,
    path: str,
    filename: str,
    formats: List[str] = ["pdf", "png"]
) -> None:
    """
    Save plot in multiple formats.
    
    Args:
        fig: Matplotlib figure object
        path: Output directory path
        filename: Base filename (without extension)
        formats: List of file formats to save
    """
    os.makedirs(path, exist_ok=True)
    for fmt in formats:
        output_file = os.path.join(path, f"{filename}.{fmt}")
        fig.savefig(output_file, format=fmt, bbox_inches="tight", dpi=300)
    plt.close(fig)


# ============================================================================
# STRING PARSING UTILITIES
# ============================================================================

def parse_string_to_list(s: str) -> Optional[List]:
    """
    Safely parse a string representation of a list into an actual list.
    
    Args:
        s: String to parse
        
    Returns:
        Parsed list or None if parsing fails
    """
    import ast
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return parsed
        return None
    except (ValueError, SyntaxError):
        return None


def get_vectors_from_dicts(dict_list: List[Dict]) -> np.ndarray:
    """
    Convert a list of configuration dictionaries into a 2D numpy array.
    
    Args:
        dict_list: List of dictionaries with the same keys
        
    Returns:
        2D numpy array of values
    """
    if not dict_list:
        return np.array([])
    headers = sorted(dict_list[0].keys())
    vectors = [[d.get(h, 0) for h in headers] for d in dict_list]
    return np.array(vectors)


# ============================================================================
# BATCH FILE OPERATIONS
# ============================================================================

def gather_files(
    data_path: str,
    pattern: str = "observed_fvals*.csv",
    filter_str: Optional[str] = None
) -> List[str]:
    """
    Gather all matching files from a directory tree.
    
    Args:
        data_path: Root directory to search
        pattern: Glob pattern for file matching
        filter_str: Optional filter string
        
    Returns:
        List of matching file paths
    """
    all_files = glob.glob(
        os.path.join(data_path, "**", pattern), 
        recursive=True
    )
    
    if filter_str:
        all_files = [f for f in all_files if filter_str in f]
        
    return all_files
