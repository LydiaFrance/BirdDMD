
import os 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pydmd.bopdmd import BOPDMD
from pydmd.preprocessing.hankel import hankel_preprocessing




def plot_dmd_histograms(DMD_results, modes, figsize=(8, 2), bins=50, color='dodgerblue', x_ticks=[0, 15, 30, 60, 120], x_lim=(0, 120), alpha=0.4):
    """
    Plots histograms for the specified modes from the DMD results.

    Parameters:
    - DMD_results: DataFrame containing DMD results with 'Lambda' column.
    - modes: List of mode indices to plot.
    - figsize: Tuple specifying the figure size.
    - bins: Number of bins for the histogram.
    - color: Color of the histogram bars.
    - x_ticks: List of x-axis ticks.
    - x_lim: Tuple specifying the x-axis limits.
    - alpha: Transparency level for the histogram bars.
    """
    num_modes = len(modes)
    fig, axes = plt.subplots(1, num_modes, figsize=figsize, sharey=False, tight_layout=True)
    for counter, ii in enumerate(modes):
        lambda_i_values = DMD_results['Lambda'].apply(lambda x: x[ii])
        axes[counter].hist(abs(lambda_i_values), bins=bins, alpha=alpha, color=color, range=(0, 200))

        axes[counter].set_xticks(x_ticks)
        axes[counter].set_xlim(x_lim)
        axes[counter].grid(alpha=0.3)
        axes[counter].tick_params(axis='x', which='major', labelsize=6, rotation=90)
        axes[counter].tick_params(axis='y', which='major', labelsize=6)
        axes[counter].set_title(f"Mode {ii + 1}", fontsize=8)
        axes[counter].set_xlabel("Frequency (Hz)", fontsize=8)

    plt.show()



# ------- Get DMD results from all sequences ------- 

def get_every_sequence_result(DMD_results_dict:dict, 
                              ii_mode: int, 
                              result_column_name: str = "Lambda"):

    """
    ii_mode: which mode has been selected
    """

    every_sequence_result = DMD_results_dict[result_column_name].apply(lambda x: x[ii_mode])

    return every_sequence_result


def plot_aggregate_DMD_histogram(every_sequence_result, ax, nBins = 100):

    ax.hist(abs(every_sequence_result), bins=nBins, alpha=0.2, color='dodgerblue')

    return ax

