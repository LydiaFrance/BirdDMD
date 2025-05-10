"""
Visualization module for BirdDMD.

This module provides functions for visualizing DMD results, marker data, and analysis outputs.
The functions are organized into the following categories:
1. Marker Data Visualization
2. DMD Mode Analysis
3. PCA Score Visualization
4. Mode Reconstruction Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union, Any
from pydmd.bopdmd import BOPDMD
from pydmd.preprocessing.hankel import hankel_preprocessing

from .data_handling import normalise_data

# -------------------- Marker Data Visualization --------------------

def plot_markers_overDist(dataframe, marker_column_names, x_axis='HorzDistance'):
    """
    Plot marker positions over distance or time.
    
    Creates a figure showing all marker positions across variables, with subplots
    for each marker coordinate (x, y, z).
    
    Parameters:
    -----------
    dataframe : pandas DataFrame
        DataFrame containing marker data
    marker_column_names : list
        List of marker column names to plot
    x_axis : str, default='HorzDistance'
        Column name to use for x-axis ('HorzDistance' or 'time')
    """
    fig, axes = plt.subplots(8, 3, figsize=(6,10), sharex=True, sharey=False)
    ax = axes.flatten()
    
    for ii, marker in enumerate(marker_column_names):
        if x_axis.lower().startswith('horz'):
            ax[ii].scatter(-dataframe[x_axis], dataframe[marker], s=0.1, c=dataframe['time'])
        else:
            ax[ii].scatter(dataframe[x_axis], dataframe[marker], s=0.1, c=dataframe['time'])
            
        ax[ii].grid(('on'), which='major', axis='x', linestyle='-', linewidth=0.5, color='k', alpha=0.5)
        ax[ii].xaxis.set_major_locator(plt.MaxNLocator(4))
        
        if ii % 3 == 0:
            ax[ii].set_ylabel(marker.split('_')[0] + " " + marker.split('_')[1], fontsize=8)
            
        if ii < 23:
            pass
        else:
            ax[ii].set_xlabel(x_axis)
            
        ax[ii].yaxis.set_ticklabels([])
        ax[ii].spines['top'].set_visible(False)
        ax[ii].spines['right'].set_visible(False)
        ax[ii].spines['left'].set_visible(False)
        ax[ii].spines['bottom'].set_visible(False)
        ax[ii].tick_params(axis='both', which='both', length=0)

    ax[0].title.set_text('x')
    ax[1].title.set_text('y')
    ax[2].title.set_text('z')
    plt.tight_layout()
    return fig

def plot_2d_markers(dataframe, marker_column_names):
    """
    Plot 2D projections of marker positions.
    
    Creates a figure showing 2D projections (y-z plane) of marker positions,
    with subplots for each marker.
    
    Parameters:
    -----------
    dataframe : pandas DataFrame
        DataFrame containing marker data
    marker_column_names : list
        List of marker column names to plot
    """
    marker_column_names = marker_column_names[::3]  # Remove all but every third marker
    marker_column_names = [marker.split('_x')[0] for marker in marker_column_names]

    fig, axes = plt.subplots(2,4, figsize=(5,8), sharex=True, sharey=True, tight_layout=True)
    ax = axes.flatten()
    
    for ii, marker in enumerate(marker_column_names):
        ax[ii].scatter(dataframe[marker+"_y"], dataframe[marker+"_z"], s=0.1, c=dataframe['time'])
        ax[ii].set_aspect('equal','box')
        ax[ii].axis('off')
        ax[ii].set_title(marker, fontsize=8)
    
    return fig

def plot_single_sequence(dataframe, seq_num, marker_name="right_wingtip_z", x_axis='HorzDistance'):
    """
    Plot a single sequence of marker data.
    
    Parameters:
    -----------
    dataframe : pandas DataFrame
        DataFrame containing marker data
    seq_num : int
        Index of the sequence to plot
    marker_name : str, default="right_wingtip_z"
        Name of the marker to plot
    x_axis : str, default='HorzDistance'
        Column name to use for x-axis
    """
    seqList = dataframe['seqID'].unique()
    seqID = seqList[seq_num]

    fig, ax = plt.subplots(1,1, figsize=(5,5), tight_layout=True)
    ax.scatter(dataframe[dataframe['seqID'] == seqID][x_axis], 
               dataframe[dataframe['seqID'] == seqID][marker_name], 
               s=10, c=dataframe[dataframe['seqID'] == seqID]['time'])
    
    if x_axis.startswith('Horz'):
        ax.set_xlabel('Horizontal Distance to Perch (m)')
    elif x_axis.startswith('time'):
        ax.set_xlabel('time from take-off jump (s)')
    else:
        ax.set_xlabel(x_axis)
    ax.set_ylabel(marker_name)
    
    return fig

# -------------------- DMD Mode Analysis --------------------

def plot_amplitude_ranking(
    markers: np.ndarray,
    times: np.ndarray,
    max_modes: int = 20,
    d: int = 2,
    eig_constraints: set = {"conjugate_pairs"},
    figsize: Tuple[int, int] = (6, 2)
) -> Tuple[plt.Figure, plt.Axes, np.ndarray]:
    """
    Plot DMD mode amplitude ranking to help determine appropriate rank.
    
    Parameters:
    -----------
    markers : np.ndarray
        Marker data array
    times : np.ndarray
        Time vector corresponding to markers
    max_modes : int, default=20
        Maximum number of modes to compute
    d : int, default=2
        Hankel matrix delay parameter
    eig_constraints : set, default={"conjugate_pairs"}
        Constraints for BOPDMD eigenvalues
    figsize : tuple, default=(6, 2)
        Size of the output plot
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    ax : matplotlib.axes.Axes
        The axes object
    sorted_amplitudes : np.ndarray
        Array of sorted absolute amplitudes (descending)
    """
 
    # Input validation
    if times.shape[0] != markers.shape[0]:
        raise ValueError("Shape mismatch: `times` and `markers` must have the same number of frames.")
    
    required_steps = max_modes + d
    if markers.shape[0] <= required_steps:
        raise ValueError(f"Not enough time steps ({markers.shape[0]}) to compute "
                       f"{max_modes} modes with delay d={d}. Need > {required_steps}.")

    # Prepare data
    normalised_markers, _ = normalise_data(markers)
    reshaped_markers = normalised_markers.reshape(markers.shape[0], -1)
    transposed_markers = reshaped_markers.T

    # Fit DMD model
    print(f"Fitting DMD with max_modes={max_modes}, d={d}...")
    dmd_model = hankel_preprocessing(
        BOPDMD(svd_rank=max_modes, eig_constraints=eig_constraints),
        d=d
    )
    dmd_model.fit(transposed_markers, t=times[1:])
    print("DMD fit complete.")

    # Get amplitudes
    amplitudes = dmd_model.amplitudes
    if amplitudes is None or len(amplitudes) == 0:
        print("Warning: DMD fit did not produce amplitudes.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No amplitudes found', ha='center', va='center')
        return fig, ax, np.array([])

    # Calculate and sort amplitudes
    abs_amplitudes = np.abs(amplitudes)
    sort_indices = np.argsort(abs_amplitudes)[::-1]
    sorted_amplitudes = abs_amplitudes[sort_indices]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    ranks = np.arange(len(sorted_amplitudes))
    ax.scatter(ranks, sorted_amplitudes, marker='o', s=15)
    ax.set_ylabel(r"Amplitude $|\beta|$")
    ax.set_xlabel("DMD Mode Rank (Sorted by Amplitude)")
    ax.set_title(f"DMD Mode Amplitude Ranking (max_modes={max_modes})")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=8)
    fig.tight_layout()

    return fig, ax, sorted_amplitudes

# -------------------- PCA Score Visualization --------------------

def plot_score_multi_PCs(
    left_scores_df,
    right_scores_df,
    PC_num_list=range(1,13),
    x_column='HorzDistance',
    figsize=(5,5)
):
    """
    Plot left and right PC scores on the same subplots.
    
    Parameters:
    -----------
    left_scores_df : pandas DataFrame
        DataFrame containing left PC scores
    right_scores_df : pandas DataFrame
        DataFrame containing right PC scores
    PC_num_list : list, default=range(1,13)
        List of PC numbers to plot (1-based indexing)
    x_column : str, default='HorzDistance'
        Name of the column to use for x-axis
    figsize : tuple, default=(5,5)
        Figure size in inches
    """
    # Color scheme
    colour_PC_dict = {
        'PC01': '#B5E675', 'PC02': '#6ED8A9', 'PC03': '#51B3D4',
        'PC04': '#4579AA', 'PC05': '#F19EBA', 'PC06': '#BC96C9',
        'PC07': '#917AC2', 'PC08': '#BE607F', 'PC09': '#624E8B', 
        'PC10': '#888888', 'PC11': '#888888', 'PC12': '#888888'
    }
    
    # PC titles
    PC_titles = [
        "wing lifting", "wing spreading", "wing sweeping",
        "tail spreading", "counter pitching", "collective pitching",
        "handwing spreading", "M-folding", "handwing sweeping", 
        "PC10","PC11", "PC12",
    ]
    
    # Score limits
    score_limits = {
        'PC01': (-0.55, 0.55), 'PC02': (-0.5, 0.4), 'PC03': (-0.15, 0.15),
        'PC04': (-0.1, 0.1), 'PC05': (-0.07, 0.07), 'PC06': (-0.09, 0.09),
        'PC07': (-0.1, 0.1), 'PC08': (-0.07, 0.07), 'PC09': (-0.07, 0.07),
        'PC10': (-0.05, 0.05), 'PC11': (-0.1, 0.1), 'PC12': (-0.04, 0.04)
    }

    # Create figure
    fig, axes = plt.subplots(4, 3, figsize=figsize, sharex=True)
    axes = axes.flatten()

    # Plot each PC
    for i, pc_num in enumerate(PC_num_list):
        ax = axes[i]
        pc_name = f'PC{pc_num:02}'
        
        # Plot left and right lines
        ax.plot(left_scores_df[x_column], left_scores_df[pc_name], 
                color=colour_PC_dict[pc_name], 
                linewidth=1.5,
                linestyle='-',
                label='Left' if i == 0 else None)
        
        ax.plot(right_scores_df[x_column], right_scores_df[pc_name], 
                color=colour_PC_dict[pc_name], 
                linewidth=1.5,
                linestyle='--',
                label='Right' if i == 0 else None)
        
        # Add zero line
        ax.axhline(y=0, color='#333333', linestyle=':', linewidth=0.5)
        
        # Customize appearance
        ax.set_title(PC_titles[i], fontsize=8, position=(0.5, 0.9))
        ax.tick_params(axis='both', labelsize=8, direction='in')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Set y limits
        ymin, ymax = score_limits[pc_name]
        ax.set_ylim(ymin, ymax)

    # Add common x-label
    fig.text(0.5, -0.02, 'horizontal distance to perch (m)', ha='center')
    fig.tight_layout()
    
    return fig 