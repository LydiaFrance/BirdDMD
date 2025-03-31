#!/usr/bin/env python
# coding: utf-8

# # DMD on a single wingbeat
# 
# > 2024-04-01 
# 
# The aim of this analysis is to find the dynamic modes of a single flapping wingbeat by hawks. 
# 
# The data contains 8 markers from hawks wings and tail during flight. The markers are tracked in 3D space and the data is collected at 200 Hz using motion capture. 
# 
# While the flights contain many flight behaviours, here we focus on a single wingbeat, the first after the take-off jump. The duration is around 0.2s. 
# 
# By finding the modes we hope to model the morphing dynamics of a bird wingbeat during flight in a reduced dimension space, which is easier for analysis and any bioinspired flapping design control.
# 
# ## Loading Data
# 
# > 2024-04-03 
# >
# > 2024-04-19 
# 
# The data is saved as `npz` which contain numpy arrays. We will load the bilateral data, which means marker data is complete on both sides for every frame. 

# In[24]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'widget')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='svg'")


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from mpl_toolkits.mplot3d import Axes3D

from pydmd import DMD
from pydmd.bopdmd import BOPDMD
from pydmd.plotter import plot_eigs, plot_summary
from pydmd.preprocessing.hankel import hankel_preprocessing

from morphing_birds import Hawk3D, plot_plotly, animate_plotly, animate_plotly_compare, animate, animate_compare

# from BirdDMD import plot_markers_overDist, plot_2d_markers, plot_single_sequence, plot_score_multi_PCs, reorder_dmd_results, reconstruct_dmd

from BirdDMD import plot_markers_overDist, plot_2d_markers, plot_single_sequence, plot_score_multi_PCs, reorder_dmd_results, reconstruct_dmd

np.set_printoptions(suppress=True, precision=3)
# Make matplotlib use the font Andale Mono
plt.rcParams['font.family'] = 'Andale Mono'




# In[3]:


column_names = np.load("../data/samples/ColumnNames2.npz")
file = np.load("../data/samples/Initial_9mStraightTurnToothless_Bilateral2.npz", allow_pickle=True)
marker_data = file["marker_data"]
info_data = file["info_data"]

marker_column_names = column_names["marker_column_names"]
info_column_names = column_names["info_column_names"]

# Create a pandas dataframe
marker_df = pd.DataFrame(marker_data, columns=marker_column_names)
info_df = pd.DataFrame(info_data, columns=info_column_names)

# Concatenate the dataframes vertically
wingbeat_df = pd.concat([info_df,marker_df], axis=1)

# Number of sequences
n_sequences = wingbeat_df["seqID"].nunique()
print("Number of sequences: ", n_sequences)

print(str(wingbeat_df.shape) + " dataframe loaded." )
wingbeat_df.head()


# ## Average Shape of the Hawk
# 
# Within the data set there are 8 markers, which are used to represent the hawk's wing and tail. All markers are represented as relative to the centre of mass. 
# 
# The markers are named as follows:
# 
# - Left Wingtip   (similar to the fingertip)
# - Right Wingtip  (similar to the fingertip)
# - Left Primary   (similar to the wrist)
# - Right Primary  (similar to the wrist)
# - Left Secondary (trailing edge of the wing)
# - Right Secondary (trailing edge of the wing)
# - Left Tailtip
# - Right Tailtip
# 
# You can see the average shape of the hawk by plotting the markers in 3D space. The blue points are measured by the motion capture, the rest of the grey points are just for visualisation purposes and are estimated using measurements of body size in the hawks.

# In[4]:


hawk3d = Hawk3D("../data/mean_hawk_shape.csv")

plot_plotly(hawk3d)


# ## Plot Raw Data
# 
# This is a single initial wingbeat from take-off by one hawk called Toothless. There are around 175 flights at 250 fps. We can plot them over the flight to see the markers changing in x, y, and z. 
# 
# Note the x axis is the horizontal distance to the perch as measured at the centre of mass of the bird. 

# In[5]:


plot_markers_overDist(wingbeat_df, marker_column_names, x_axis='time')


# # Plot the Marker Trajectories in 2d

# In[6]:


plot_2d_markers(wingbeat_df, marker_column_names)


# ## Find Sequence Lengths

# In[7]:


# Using seqID as a category, find the number of rows in each category
seqID_counts = wingbeat_df['seqID'].value_counts()

# Create a histogram
plt.figure(figsize=(3,3))
plt.hist(seqID_counts, bins=20)
plt.xlabel("Number of frames per sequence")

plt.show()

# Find the sequences with the most frames
seqID_counts.idxmax()

# Find what number 04_09_048_1 is in the list of unique sequences
# wingbeat_df['seqID'].unique().tolist().index('04_09_048_1')
wingbeat_df['seqID'].unique().tolist().index('04_09_051_2')


# In[8]:


wingbeat_df["Turn"]


# ## Plot a Single Sequence

# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'inline')
# plot_single_sequence(wingbeat_df, 54, marker_name="right_wingtip_z", x_axis="time")
plot_single_sequence(wingbeat_df, 60, marker_name="right_wingtip_z", x_axis="time")


# ## Remove Duplicated Time Frames

# In[11]:


duplicates = wingbeat_df[wingbeat_df.duplicated(subset=['frameID'], keep=False)]

ratio_duplicates = (duplicates.shape[0]/wingbeat_df.shape[0])*100
print(f"Duplicated frames are {np.round(ratio_duplicates,2)}% of the total data. ")


# Remove duplicates
wingbeat_df = wingbeat_df.drop_duplicates(subset=['frameID'], keep='first')

# Remember to reset the index
wingbeat_df = wingbeat_df.reset_index(drop=True)


# # DMD on Single Flight

# In[12]:


# %matplotlib inline
# %matplotlib inline


def load_sequence_data(wingbeat_df, seqID, marker_column_names):
    """Load markers and time data for a given sequence ID."""
    condition = wingbeat_df['seqID'] == seqID
    single_seq_markers = wingbeat_df[condition][marker_column_names].to_numpy().astype(np.float64)
    single_seq_time = wingbeat_df[condition]['time'].to_numpy().astype(np.float64)
    return single_seq_markers, single_seq_time

def normalise_data(markers):
    """Normalize marker data by subtracting the average shape."""

    hawk3d = Hawk3D("../data/mean_hawk_shape.csv")
    average_shape = hawk3d.markers.reshape(24,1).T

    return markers - average_shape, average_shape

def transpose_data(data):
    """Transpose the given data."""
    return data.T

def add_average_shape(data, average_shape):
    """Add the average shape back to the data."""
    return data + average_shape

def reshape_data(data, rows, cols, depth):
    """Reshape data to the specified dimensions."""
    return data.reshape(rows, cols, depth)


# Load and process a single sequence
seqList = wingbeat_df['seqID'].unique()
# seqID = seqList[54]
seqID = seqList[60]
seqInfo = wingbeat_df[wingbeat_df['seqID'] == seqID]
markers, times = load_sequence_data(wingbeat_df, seqID, marker_column_names)

# Normalise the data using a class instance to handle the hawk shape
normalised_markers, average_shape = normalise_data(markers)
transposed_markers = transpose_data(normalised_markers)

# Dynamic Mode Decomposition (DMD)
n_Modes = 8
delay_optdmd = hankel_preprocessing(BOPDMD(svd_rank=n_Modes, eig_constraints={"conjugate_pairs"}), d=2)
delay_optdmd.fit(transposed_markers, t=times[1:])

plt.close('all')
# plot_summary(delay_optdmd, index_modes=[0,1,2], order='F') # pick correct modes


# ## Test how many Modes

# In[ ]:


n_Modes = 20
delay_optdmd = hankel_preprocessing(BOPDMD(svd_rank=n_Modes, eig_constraints={"conjugate_pairs"}), d=2)
delay_optdmd.fit(transposed_markers, t=times[1:])

example_order = np.argsort(-np.abs(delay_optdmd.amplitudes))
example_amplitudes = np.abs(delay_optdmd.amplitudes[example_order])

plt.figure(figsize=(6, 2))
plt.scatter(np.arange(20), example_amplitudes)
plt.ylabel(r"$\beta$")
plt.xlabel("DMD rank")


# 

# In[ ]:





# ## Compare Original Flight and DMD Reconstruction

# In[13]:


get_ipython().run_line_magic('matplotlib', 'widget')
get_ipython().run_line_magic('matplotlib', 'widget')

# Forecast the data to recreate the flight with a reduced number of modes
forecast_mean = delay_optdmd.forecast(times)

# Reshape forecast data and add the average shape back
forecast = np.real(forecast_mean[:24,:])
forecast_transposed = transpose_data(forecast)
forecast_plus_mean = add_average_shape(forecast_transposed, average_shape)

# Animate forecast and original in comparison
keypoints = reshape_data(forecast_plus_mean, -1, 8, 3)
# Take away the first frame and add an extra frame to the end
keypoints = np.concatenate([keypoints[1:], keypoints[-1:]], axis=0)

original_markers = reshape_data(markers, -1, 8, 3)

hawk3d = Hawk3D("../data/mean_hawk_shape.csv")
animate_plotly_compare(hawk3d, keypoints_frames_list=[original_markers,keypoints])
# animate_plotly_compare(hawk3d, keypoints_frames_list=[keypoints])

# animation = animate(hawk3d, keypoints)
# animation.save(f"../figures/gifs/2025-03-28/{n_Modes}Total_Pair1.gif", writer='Pillow', fps=5, dpi=300)
# plt.close("all")


# ## Upsample the Data

# 

# In[ ]:


# Original time points
# print(times)

# Create interpolated time points for DMD reconstruction
fake_time = np.arange(times[0], times[-1]+0.001, 0.001)
# print(fake_time)

# Create expanded original data by repeating frames
expanded_markers = []
current_frame = 0

for t in fake_time:
    # Keep using same frame until we hit a time point in the original data
    while current_frame < len(times)-1 and t >= times[current_frame+1]:
        current_frame += 1
    expanded_markers.append(markers[current_frame].copy())

expanded_markers = np.array(expanded_markers)
original_markers = expanded_markers.reshape(-1, 8, 3)

# Use reconstruction for forecast
reconstruction = reconstruct_dmd(fake_time,  
                               np.imag(delay_optdmd.eigs), 
                               delay_optdmd.modes[:24,:], 
                               delay_optdmd.amplitudes)

# Reshape forecast and add the average shape back
forecast_plus_mean = reconstruction + average_shape
keypoints = forecast_plus_mean.reshape(-1, 8, 3)

# Animate both sequences
hawk3d = Hawk3D("../data/mean_hawk_shape.csv")
animate_plotly_compare(hawk3d, keypoints_frames_list=[original_markers, keypoints], colours=["blue", "red"])


# ## The Mode Frequencies

# In[14]:


Lambda, Modes, bn, Psi, PhaseShifts = reorder_dmd_results(delay_optdmd, num_markers=8, nModes=n_Modes)

for i_Mode in range(n_Modes):
    print(f"Frequency of Mode {i_Mode}: {Lambda[i_Mode]}")


# ## Reconstructing the Modes

# In[15]:


import gc
gc.collect()
plt.close("all")


# In[33]:


Lambda, Modes, bn, Psi, PhaseShifts = reorder_dmd_results(delay_optdmd, num_markers=8, nModes=n_Modes)


# First, add a function to reconstruct specific modes

def reconstruct_specific_modes(times, dmd_results, mode_indices):
    """
    Reconstruct specific DMD modes using original DMD outputs.
    
    Parameters:
    - times: time points array
    - dmd_results: DMD object with eigs, modes, amplitudes
    - mode_indices: list of mode indices to include in reconstruction
    
    Returns:
    - reconstruction: reconstructed data from selected modes
    """
    # Initialize reconstruction array
    reconstruction = np.zeros((dmd_results.modes.shape[0], len(times)), dtype=complex)
    
    # Add contribution from each selected mode
    for j in mode_indices:
        # Use original (unordered) DMD components
        mode = dmd_results.modes[:, j].reshape(-1, 1)
        eig = dmd_results.eigs[j]
        amplitude = dmd_results.amplitudes[j]
        
        # Time evolution
        time_dynamics = np.exp(eig * times)
        
        # Add this mode's contribution
        reconstruction += amplitude * np.dot(mode, time_dynamics.reshape(1, -1))
    
    return reconstruction


# Now use it to create reconstructions of specific modes
# Example: Reconstruct using modes 0 and 1
# modes_0_1 = reconstruct_specific_modes(
#     times, 
#     delay_optdmd.eigs, 
#     delay_optdmd.modes, 
#     delay_optdmd.amplitudes, 
#     [0, 1]
# )

modes_0_1 = reconstruct_specific_modes(times, delay_optdmd, [0, 1])
modes_2_3 = reconstruct_specific_modes(times, delay_optdmd, [2, 3])
modes_4_5 = reconstruct_specific_modes(times, delay_optdmd, [4, 5])
modes_6_7 = reconstruct_specific_modes(times, delay_optdmd, [6, 7])

# modes_0_1_2_3 = reconstruct_specific_modes(times, delay_optdmd, [0, 1, 2, 3])

# Process reconstructions like you do with the forecast
# For modes 0-1
recon_0_1 = np.real(modes_0_1[:24,:])
recon_0_1_transposed = transpose_data(recon_0_1)
recon_0_1_plus_mean = add_average_shape(recon_0_1_transposed, average_shape)
keypoints_0_1 = reshape_data(recon_0_1_plus_mean, -1, 8, 3)
keypoints_0_1 = np.concatenate([keypoints_0_1[1:], keypoints_0_1[-1:]], axis=0)

# # For modes 2-3
recon_2_3 = np.real(modes_2_3[:24,:])
recon_2_3_transposed = transpose_data(recon_2_3)
recon_2_3_plus_mean = add_average_shape(recon_2_3_transposed, average_shape)
keypoints_2_3 = reshape_data(recon_2_3_plus_mean, -1, 8, 3)
keypoints_2_3 = np.concatenate([keypoints_2_3[1:], keypoints_2_3[-1:]], axis=0)

recon_4_5 = np.real(modes_4_5[:24,:])
recon_4_5_transposed = transpose_data(recon_4_5)
recon_4_5_plus_mean = add_average_shape(recon_4_5_transposed, average_shape)
keypoints_4_5 = reshape_data(recon_4_5_plus_mean, -1, 8, 3)
keypoints_4_5 = np.concatenate([keypoints_4_5[1:], keypoints_4_5[-1:]], axis=0)

recon_6_7 = np.real(modes_6_7[:24,:])
recon_6_7_transposed = transpose_data(recon_6_7)
recon_6_7_plus_mean = add_average_shape(recon_6_7_transposed, average_shape)
keypoints_6_7 = reshape_data(recon_6_7_plus_mean, -1, 8, 3)
keypoints_6_7 = np.concatenate([keypoints_6_7[1:], keypoints_6_7[-1:]], axis=0)

# recon_0_1_2_3 = np.real(modes_0_1_2_3[:24,:])
# recon_0_1_2_3_transposed = transpose_data(recon_0_1_2_3)
# recon_0_1_2_3_plus_mean = add_average_shape(recon_0_1_2_3_transposed, average_shape)
# keypoints_0_1_2_3 = reshape_data(recon_0_1_2_3_plus_mean, -1, 8, 3)
# keypoints_0_1_2_3 = np.concatenate([keypoints_0_1_2_3[1:], keypoints_0_1_2_3[-1:]], axis=0)

# Animate original data vs reconstructions from different modes
# animate_plotly_compare(hawk3d, keypoints_frames_list=[
#     keypoints_0_1, 
#     keypoints_2_3
# ])


plt.close("all")
animation = animate_compare(hawk3d, keypoints_frames_list=[keypoints_4_5, keypoints_2_3], az=90, el=10)
# animation.save(f"../figures/gifs/2025-03-28/{n_Modes}Total_Pair3and4.gif", writer='Pillow', fps=5, dpi=300)


# ## Load the Morphing Shape Space

# In[17]:


principal_components = np.load("../data/PCA_results/unilateral_principal_components.npy")
mu = np.load("../data/PCA_results/unilateral_mu.npy")



# ## Make the DMD Mode unilateral
# 
# Start with the flapping mode. 
# 

# In[53]:


def make_unilateral_keypoints(keypoints):
    """
    Process keypoint data to separate and prepare left and right side data
    """
    # Separate left and right keypoints
    left_reconstruction = keypoints[:,::2,:].copy()
    right_reconstruction = keypoints[:,1::2,:].copy()
    
    # Mirror the x coordinate for left side to match right
    left_reconstruction[:,:,0] = -left_reconstruction[:,:,0]
    
    # Flatten the data
    left_flat = left_reconstruction.reshape(-1, left_reconstruction.shape[1]*3)
    right_flat = right_reconstruction.reshape(-1, right_reconstruction.shape[1]*3)
    
    # Concatenate left and right data
    dmd_reconstruction_flat = np.concatenate([left_flat, right_flat], axis=0)
    
    # Create boolean array for left/right identification
    left_right_bool = np.concatenate([
        np.zeros(left_flat.shape[0]),  # 0 for left
        np.ones(right_flat.shape[0])   # 1 for right
    ])
    
    return dmd_reconstruction_flat, left_right_bool

dmd_reconstruction_flat, left_right_bool = make_unilateral_keypoints(keypoints_4_5)


# In[54]:


def project_into_pca_space(new_data, mu, principal_components):
    """
    Project new shape data into an existing PCA space.
    
    Parameters:
        new_data (np.ndarray): New data to project, shape [n_frames, n_markers*3].
        mu (np.ndarray): Mean shape, expected shape [1, n_markers, 3].
        principal_components (np.ndarray): PCA components, shape [n_components, n_markers*3].
    
    Returns:
        np.ndarray: The projected scores with shape [n_frames, n_components].
    """
    # Flatten the mean shape to match the feature dimensions
    mu_flat = mu.flatten()  # shape: [n_markers*3,]
    
    # Center the new data by subtracting the mean
    new_data_centered = new_data - mu_flat

    # if the data is bilateral 
    
    # Compute the projection (scores) by multiplying with the transpose of principal components
    scores_new = np.dot(new_data_centered, principal_components.T)
    
    return scores_new


scores_dmd = project_into_pca_space(dmd_reconstruction_flat, mu, principal_components)



# Split the scores into left and right
scores_dmd_left = scores_dmd[left_right_bool==1,:]
scores_dmd_right = scores_dmd[left_right_bool==0,:]





# In[44]:


print(seqInfo.shape)
print(scores_dmd_left.shape)


# ## Create Score DataFrame

# In[49]:


def create_scores_info_df(scores, seq_info_df, left_right_bool=None):
    # Add the scores to the dataframe. Give the column the name 'PC1' etc.
    num_components = scores.shape[1]
    PC_names = [f'PC{i:02}' for i in np.arange(1, num_components+1)]
    
    # Create a pandas dataframe from scores
    scores_df = pd.DataFrame(scores, columns=PC_names).reset_index(drop=True)
    
    # Reset index of seq_info_df as well
    seq_info_df = seq_info_df.reset_index(drop=True)
    
    # check seq has the same number of rows as scores
    if seq_info_df.shape[0] != scores.shape[0]:
        raise ValueError("The number of rows in seq_info_df and scores must be the same")
    
    # Concatenate the scores dataframe with seq_info_df
    scores_df = pd.concat([scores_df, seq_info_df], axis=1)

    return scores_df

left_scores_df = create_scores_info_df(scores_dmd_left, seqInfo)
right_scores_df = create_scores_info_df(scores_dmd_right, seqInfo)

# Merge the two dataframes with the left_right_bool
# To merge left and right dataframes, you can add a 'Side' column and concatenate vertically
left_scores_df['Left'] = 1
right_scores_df['Left'] = 0
combined_scores_df = pd.concat([left_scores_df, right_scores_df], axis=0)
combined_scores_df.head()




# ## Plotting Shape scores for the DMD Mode

# In[51]:


_ = plot_score_multi_PCs(left_scores_df,right_scores_df, x_column='time')
plt.show()
# Make matplotlib use the font Andale Mono


# In[82]:


dmd_reconstruction_flat, left_right_bool = make_unilateral_keypoints(keypoints_4_5)
scores_dmd = project_into_pca_space(dmd_reconstruction_flat, mu, principal_components)


# Split the scores into left and right
left_scores_df = create_scores_info_df(scores_dmd[left_right_bool==1,:], seqInfo)
right_scores_df = create_scores_info_df(scores_dmd[left_right_bool==0,:], seqInfo)

left_scores_df['Left'] = 1
right_scores_df['Left'] = 0

_ = plot_score_multi_PCs(left_scores_df,right_scores_df, x_column='time')
plt.show()


# ## Plot for the the 2*Frequency DMD Mode

# In[81]:


dmd_reconstruction_flat, left_right_bool = make_unilateral_keypoints(keypoints_2_3)
scores_dmd = project_into_pca_space(dmd_reconstruction_flat, mu, principal_components)


# Split the scores into left and right
left_scores_df = create_scores_info_df(scores_dmd[left_right_bool==1,:], seqInfo)
right_scores_df = create_scores_info_df(scores_dmd[left_right_bool==0,:], seqInfo)

left_scores_df['Left'] = 1
right_scores_df['Left'] = 0

_ = plot_score_multi_PCs(left_scores_df,right_scores_df, x_column='time')
plt.show()



# In[64]:


def plot_marker_positions(scores_df, figsize=(5, 5), x_column='HorzDistance'):
    """
    Plot x, y, z positions for each marker (wingtip, primary, secondary, tailtip)
    with left and right overlaid in each subplot.
    
    Parameters:
    -----------
    scores_df : pandas DataFrame
        DataFrame containing marker positions with columns like 'left_wingtip_x'
    figsize : tuple
        Figure size in inches
    """
    # Define markers and coordinates
    markers = ['wingtip', 'primary', 'secondary', 'tailtip']
    coords = ['x', 'y', 'z']
    
    # Create figure
    fig, axes = plt.subplots(len(markers), len(coords), figsize=figsize, sharex=True)
    
    # For consistent line styles
    left_style = {'color': 'green', 'linestyle': '-', 'alpha': 0.7, 'linewidth': 1.5}
    right_style = {'color': 'green', 'linestyle': '--', 'alpha': 0.7, 'linewidth': 1.5}
    
    for i, marker in enumerate(markers):
        for j, coord in enumerate(coords):
            ax = axes[i, j]
            
            # Plot left and right data
            left_col = f'left_{marker}_{coord}'
            right_col = f'right_{marker}_{coord}'
            
            # Plot lines
            ax.plot(scores_df[x_column], scores_df[left_col], 
                   label='Left' if i==0 and j==0 else None, **left_style)
            ax.plot(scores_df[x_column], scores_df[right_col], 
                   label='Right' if i==0 and j==0 else None, **right_style)
            
            # Add zero line
            ax.axhline(y=0, color='#333333', linestyle=':', linewidth=0.5)
            
            # Customize appearance
            ax.tick_params(axis='both', labelsize=8, direction='in')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add titles
            if i == 0:
                ax.set_title(f'{coord.upper()}', fontsize=10)
            if j == 0:
                ax.set_ylabel(marker, fontsize=10)
                
            # Add legend to first subplot only
            # if i == 0 and j == 0:
            #     ax.legend(loc='upper right', fontsize=8)

    # Add common x-label
    fig.text(0.5, 0.02, 'horizontal distance to perch (m)', ha='center')
    
    # Adjust layout
    fig.tight_layout()
    
    return fig

# Example usage:
fig = plot_marker_positions(left_scores_df, x_column='time')
plt.show()


# In[165]:


left_scores_df.shape


# In[133]:


plt.figure(figsize=(8,6))
plt.plot(scores_dmd_0_1_left[:, 0], scores_dmd_0_1_left[:, 1], label="Left DMD trajectory", marker='o')
plt.plot(scores_dmd_0_1_right[:, 0], scores_dmd_0_1_right[:, 1], label="Right DMD trajectory", marker='o')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("DMD Trajectory in PCA Shape Space")
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


# def mode_reconstruction(phi, omega, b, mode_index):
# """Reconstruct individual components of the DMD"""
# omega = np.atleast_2d(omega).T
# phi = np.atleast_2d(phi)

# xr = np.zeros((nx, nt), np.complex128)
# for j in mode_index:
# xr += np.linalg.multi_dot(
# [
# np.atleast_2d(phi[:, j]).T,
# np.diag(np.atleast_1d(b[j])),
# np.atleast_2d(np.exp(omega[j] * t)),
# ]
# )

# return xr


# recon_x1 = mode_reconstruction(lead_modes, lead_eigs, lead_amplitudes, [2, 3]).real
# recon_x2 = mode_reconstruction(lead_modes, lead_eigs, lead_amplitudes, [0, 1]).real


# In[ ]:
