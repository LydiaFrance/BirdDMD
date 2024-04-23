import numpy as np

import matplotlib.pyplot as plt

def plot_markers_overDist(dataframe, marker_column_names):

    # Create a figure which shows all the instances of the first wingbeat across all the variables. 
    fig, axes = plt.subplots(8, 3, figsize=(10,10), sharex=True, sharey=False)
    ax = axes.flatten()
    for ii, marker in enumerate(marker_column_names):
        ax[ii].scatter(-dataframe['HorzDistance'], dataframe[marker], s=0.1, c=dataframe['time'])
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
            ax[ii].set_xlabel('Horizontal Distance to Perch (m)')
            
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

def plot_single_sequence(dataframe, seq_num, marker_name="right_wingtip_z"):

    seqList = dataframe['seqID'].unique()

    seqID = seqList[seq_num]


    fig, ax = plt.subplots(1,1, figsize=(5,5), tight_layout=True)
    ax.scatter(dataframe[dataframe['seqID'] == seqID]['HorzDistance'], 
               dataframe[dataframe['seqID'] == seqID][marker_name], 
                s=10, c=dataframe[dataframe['seqID'] == seqID]['time'])
    
    ax.set_xlabel('Horizontal Distance to Perch (m)')
    ax.set_ylabel(marker_name)