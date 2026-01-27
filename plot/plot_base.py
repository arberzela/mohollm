"""
Base classes for common plotting patterns.

This module provides abstract base classes and concrete implementations
for different types of plots (hypervolume, fvals, etc.).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from plot_settings import get_color, LINE_STYLE_HV, LABEL_MAP_HV, MARKER_MAP
from plot_common import (
    PlotStyle,
    compute_hypervolume,
    convert_data_to_hv_over_time,
    normalize_data,
    apply_axis_style,
    set_axis_limits,
    save_plot,
    group_data_by_method,
    custom_sort_key,
)


class BasePlotter(ABC):
    """Abstract base class for plotters."""
    
    def __init__(
        self,
        benchmark: str,
        title: str,
        output_path: str,
        filename: str,
        style: str = "default"
    ):
        """
        Initialize the plotter.
        
        Args:
            benchmark: Benchmark name
            title: Plot title
            output_path: Directory to save plots
            filename: Base filename for output
            style: Matplotlib style preset to use
        """
        self.benchmark = benchmark
        self.title = title
        self.output_path = output_path
        self.filename = filename
        self.style = style
        PlotStyle.apply(style)
    
    @abstractmethod
    def process_data(
        self,
        method_files: Dict[str, List[str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Process raw data files into plot-ready format."""
        pass
    
    @abstractmethod
    def create_plot(
        self,
        data: Dict[str, Any],
        **kwargs
    ) -> plt.Figure:
        """Create the plot from processed data."""
        pass
    
    def plot(
        self,
        method_files: Dict[str, List[str]],
        **kwargs
    ) -> None:
        """
        Complete plotting pipeline: process data, create plot, save.
        
        Args:
            method_files: Dictionary mapping method names to file paths
            **kwargs: Additional arguments passed to processing and plotting
        """
        data = self.process_data(method_files, **kwargs)
        fig = self.create_plot(data, **kwargs)
        save_plot(fig, self.output_path, self.filename)


class HypervolumePlotter(BasePlotter):
    """Plotter for hypervolume over time."""
    
    def __init__(
        self,
        benchmark: str,
        title: str,
        output_path: str,
        filename: str,
        reference_point: List[float],
        style: str = "default"
    ):
        """
        Initialize hypervolume plotter.
        
        Args:
            benchmark: Benchmark name
            title: Plot title
            output_path: Directory to save plots
            filename: Base filename for output
            reference_point: Reference point for HV calculation
            style: Matplotlib style preset
        """
        super().__init__(benchmark, title, output_path, filename, style)
        self.reference_point = reference_point
    
    def process_data(
        self,
        method_files: Dict[str, List[str]],
        columns: Optional[List[str]] = None,
        trials: Optional[int] = None,
        normalize: bool = False,
        min_max_metrics: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Process data files to compute hypervolume trajectories.
        
        Args:
            method_files: Dictionary mapping method names to file paths
            columns: Columns to read from CSV
            trials: Maximum number of trials
            normalize: Whether to normalize data
            min_max_metrics: Normalization parameters
            
        Returns:
            Dictionary with 'mean' and 'std' HV trajectories
        """
        data = {"mean": {}, "std": {}}
        
        for method_name, files in method_files.items():
            all_hvs = []
            
            for file in files:
                df = pd.read_csv(file)
                if columns:
                    df = df[columns]
                if trials:
                    df = df.iloc[:trials]
                if normalize and min_max_metrics:
                    df = normalize_data(df, min_max_metrics)
                    
                hv_trajectory = convert_data_to_hv_over_time(df, self.reference_point)
                all_hvs.append(hv_trajectory)
            
            if all_hvs:
                all_hvs = np.array(all_hvs)
                data["mean"][method_name] = np.mean(all_hvs, axis=0)
                data["std"][method_name] = np.std(all_hvs, axis=0)
        
        return data
    
    def create_plot(
        self,
        data: Dict[str, Dict[str, np.ndarray]],
        x_lim: Optional[Tuple[float, float]] = None,
        y_lim: Optional[Tuple[float, float]] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Create hypervolume plot.
        
        Args:
            data: Processed HV data with mean and std
            x_lim: X-axis limits
            y_lim: Y-axis limits
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(5, 4.5))
        
        methods = sorted(data["mean"].keys(), key=custom_sort_key)
        
        for i, method in enumerate(methods):
            mean = data["mean"][method]
            std = data["std"][method]
            x = np.arange(1, len(mean) + 1)
            
            color = get_color(method, i)
            label = LABEL_MAP_HV.get(method, method)
            linestyle = LINE_STYLE_HV[i % len(LINE_STYLE_HV)]
            marker = MARKER_MAP.get(method, MARKER_MAP.get("default", "o"))
            
            ax.plot(x, mean, label=label, color=color, linestyle=linestyle,
                   linewidth=2, marker=marker, markersize=4, markevery=5)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)
        
        ax.set_xlabel("Trials")
        ax.set_ylabel("Hypervolume")
        ax.set_title(self.title)
        
        set_axis_limits(ax, x_lim, y_lim)
        apply_axis_style(ax)
        
        ax.legend(frameon=True, fancybox=False, shadow=False,
                 bbox_to_anchor=(1.05, 1), loc="upper left")
        
        return fig


class FvalPlotter(BasePlotter):
    """Plotter for objective function values over time."""
    
    def process_data(
        self,
        method_files: Dict[str, List[str]],
        columns: Optional[List[str]] = None,
        trials: Optional[int] = None,
        normalize: bool = False,
        min_max_metrics: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Process data files to compute fval trajectories."""
        data = {"mean": {}, "std": {}}
        
        for method_name, files in method_files.items():
            all_fvals = []
            
            for file in files:
                df = pd.read_csv(file)
                if columns:
                    df = df[columns]
                if trials:
                    df = df.iloc[:trials]
                if normalize and min_max_metrics:
                    df = normalize_data(df, min_max_metrics)
                    
                all_fvals.append(df.values)
            
            if all_fvals:
                all_fvals = np.array(all_fvals)
                data["mean"][method_name] = np.mean(all_fvals, axis=0)
                data["std"][method_name] = np.std(all_fvals, axis=0)
        
        return data
    
    def create_plot(
        self,
        data: Dict[str, Dict[str, np.ndarray]],
        x_lim: Optional[Tuple[float, float]] = None,
        y_lim: Optional[Tuple[float, float]] = None,
        use_log_scale: bool = False,
        **kwargs
    ) -> plt.Figure:
        """Create fval plot."""
        fig, ax = plt.subplots(figsize=(5, 3.5))
        
        methods = sorted(data["mean"].keys(), key=custom_sort_key)
        
        for i, method in enumerate(methods):
            mean = data["mean"][method]
            std = data["std"][method]
            x = np.arange(1, len(mean) + 1)
            
            color = get_color(method, i)
            label = LABEL_MAP_HV.get(method, method)
            
            ax.plot(x, mean, label=label, color=color, linewidth=2)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)
        
        ax.set_xlabel("Trials")
        ax.set_ylabel("Fval")
        ax.set_title(self.title)
        
        if use_log_scale:
            ax.set_yscale("log")
        
        set_axis_limits(ax, x_lim, y_lim)
        apply_axis_style(ax)
        
        ax.legend(frameon=True, fancybox=False, shadow=False)
        
        return fig


class SubplotPlotter(BasePlotter):
    """Base class for creating subplot figures."""
    
    def __init__(
        self,
        benchmarks: List[str],
        titles: List[str],
        output_path: str,
        filename: str,
        ncols: int = 2,
        style: str = "subplot"
    ):
        """
        Initialize subplot plotter.
        
        Args:
            benchmarks: List of benchmark names
            titles: List of subplot titles
            output_path: Directory to save plots
            filename: Base filename for output
            ncols: Number of columns in subplot grid
            style: Matplotlib style preset
        """
        super().__init__("", "", output_path, filename, style)
        self.benchmarks = benchmarks
        self.titles = titles
        self.ncols = ncols
        self.nrows = (len(benchmarks) + ncols - 1) // ncols
    
    def create_subplot_grid(self) -> Tuple[plt.Figure, np.ndarray]:
        """Create figure with subplot grid."""
        figsize = (self.ncols * 6, self.nrows * 4)
        fig, axes = plt.subplots(self.nrows, self.ncols, figsize=figsize)
        
        # Flatten axes array for easier indexing
        if self.nrows == 1 and self.ncols == 1:
            axes = np.array([axes])
        elif self.nrows == 1 or self.ncols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        return fig, axes
