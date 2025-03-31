import numpy as np

import matplotlib.pyplot as plt

def plot_markers_overDist(dataframe, marker_column_names, x_axis='HorzDistance'):

    # Create a figure which shows all the instances of the first wingbeat across all the variables. 
    fig, axes = plt.subplots(8, 3, figsize=(6,10), sharex=True, sharey=False)
    ax = axes.flatten()
    for ii, marker in enumerate(marker_column_names):
        if x_axis.lower().startswith('horz'):
            ax[ii].scatter(-dataframe[x_axis], dataframe[marker], s=0.1, c=dataframe['time'])
        else:
            ax[ii].scatter(dataframe[x_axis], dataframe[marker], s=0.1, c=dataframe['time'])
        ax[ii].grid(('on'), which='major', axis='x', linestyle='-', linewidth=0.5, color='k', alpha=0.5)

        # turn off the x axis for all but the last plot
        # Make sure there are only 3 ticks
        ax[ii].xaxis.set_major_locator(plt.MaxNLocator(4))
        
        # For every 3rd plot, label the y axis
        if ii % 3 == 0:
            ax[ii].set_ylabel(marker.split('_')[0] + " " + marker.split('_')[1], fontsize=8)

        if ii < 23:
            pass
        else:
            ax[ii].set_xlabel(x_axis)
            
        # Turn off the y axis for all and the y labels
        ax[ii].yaxis.set_ticklabels([])

        # Turn off box
        ax[ii].spines['top'].set_visible(False)
        ax[ii].spines['right'].set_visible(False)
        ax[ii].spines['left'].set_visible(False)
        ax[ii].spines['bottom'].set_visible(False)

        # Turn off ticks
        ax[ii].tick_params(axis='both', which='both', length=0)

    ax[0].title.set_text('x')
    ax[1].title.set_text('y')
    ax[2].title.set_text('z')

    plt.tight_layout()

def plot_2d_markers(dataframe, marker_column_names):

    # Remove all but every third marker
    marker_column_names = marker_column_names[::3]

    # remove the _x from the marker names
    marker_column_names = [marker.split('_x')[0] for marker in marker_column_names]

    fig, axes = plt.subplots(2,4, figsize=(5,8), sharex=True, sharey=True, tight_layout=True)
    ax = axes.flatten()
    for ii, marker in enumerate(marker_column_names):


        ax[ii].scatter(dataframe[marker+"_y"], dataframe[marker+"_z"], s=0.1, c=dataframe['time'])
        ax[ii].set_aspect('equal','box')
        
        # Turn off axis
        ax[ii].axis('off')

        ax[ii].set_title(marker, fontsize=8)

def plot_single_sequence(dataframe, seq_num, marker_name="right_wingtip_z", x_axis='HorzDistance'):

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


# ---- PCA RECONSTRUCTION PLOTS ----


def plot_score_multi_PCs(left_scores_df, right_scores_df, PC_num_list=range(1,13), 
                          x_column='HorzDistance', figsize=(5,5)):
    """
    Plot left and right scores on the same subplots.
    
    Parameters:
    -----------
    left_scores_df : pandas DataFrame
        DataFrame containing left PC scores
    right_scores_df : pandas DataFrame
        DataFrame containing right PC scores
    PC_num_list : list
        List of PC numbers to plot (1-based indexing)
    x_column : str
        Name of the column to use for x-axis
    figsize : tuple
        Figure size in inches
    """
    # Color scheme from original
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
    score_95 = {}
    score_5 = {}
    score_95['PC01'] = 0.55
    score_5['PC01'] = -0.55
    score_95['PC02'] = 0.4
    score_5['PC02'] = -0.5
    score_95['PC03'] = 0.15
    score_5['PC03'] = -0.15
    score_95['PC04'] = 0.1
    score_5['PC04'] = -0.1
    score_95['PC05'] = 0.07
    score_5['PC05'] = -0.07
    score_95['PC06'] = 0.09
    score_5['PC06'] = -0.09
    score_95['PC07'] = 0.1
    score_5['PC07'] = -0.1
    score_95['PC08'] = 0.07
    score_5['PC08'] = -0.07
    score_95['PC09'] = 0.07
    score_5['PC09'] = -0.07
    score_95['PC10'] = 0.05
    score_5['PC10'] = -0.05
    score_95['PC11'] = 0.1
    score_5['PC11'] = -0.1
    score_95['PC12'] = 0.04
    score_5['PC12'] = -0.04

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
                label='Left' if i == 0 else None)  # Only add label for first subplot
        
        ax.plot(right_scores_df[x_column], right_scores_df[pc_name], 
                color=colour_PC_dict[pc_name], 
                linewidth=1.5,
                linestyle='--',
                label='Right' if i == 0 else None)  # Only add label for first subplot
        
        

        # Add zero line
        ax.axhline(y=0, color='#333333', linestyle=':', linewidth=0.5)
        
        # Customize appearance
        ax.set_title(PC_titles[i], fontsize=8, position=(0.5, 0.9))
        ax.tick_params(axis='both', labelsize=8, direction='in')
        
        # Remove frame elements
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Add legend to first subplot only
        # if i == 0:
            # ax.legend(loc='upper right', fontsize=8)

        # Set y limits for all subplots
        ax.set_ylim(score_5[pc_name], score_95[pc_name])

    # Add common x-label
    fig.text(0.5, -0.02, 'horizontal distance to perch (m)', ha='center')
    
    # Adjust layout
    fig.tight_layout()
    
    return fig