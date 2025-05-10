#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

from BirdDMD import (load_sequence_data, 
                     remove_time_duplicates,
                     run_single_wingbeat_dmd,
                     reconstruct_dmd,
                     plot_amplitude_ranking,
                     run_forecast,
                     reorder_dmd_results,
                     reconstruct_specific_modes,
                     modify_mode_frequencies,
                     run_forecast_with_modified_modes,
                     make_unilateral_keypoints,
                     project_into_pca_space,
                     create_scores_info_df,
                     plot_2d_markers, 
                     plot_markers_overDist, 
                     plot_single_sequence, 
                     plot_score_multi_PCs)

np.set_printoptions(suppress=True, precision=3)
# Make matplotlib use the font Andale Mono
plt.rcParams['font.family'] = 'Andale Mono'




# In[2]:


column_names = np.load("../data/samples/ColumnNames2.npz")
file = np.load("../data/samples/Flapping_9mStraightTurnToothless_Bilateral.npz", allow_pickle=True)
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

# In[3]:


hawk3d = Hawk3D("../data/mean_hawk_shape.csv")

plot_plotly(hawk3d)


# ## Plot Raw Data
# 
# This is the first two wingbeats from take-off by one hawk called Toothless. There are around 175 flights at 250 fps. We can plot them over the flight to see the markers changing in x, y, and z. 
# 
# Note the x axis is the horizontal distance to the perch as measured at the centre of mass of the bird. 

# In[4]:


plot_markers_overDist(wingbeat_df, marker_column_names, x_axis='time')


# # Plot the Marker Trajectories in 2d

# In[5]:


plot_2d_markers(wingbeat_df, marker_column_names)


# ## Find Sequence Lengths

# In[5]:


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
wingbeat_df['seqID'].unique().tolist().index('04_09_038_1')



# ## Plot a Single Sequence

# In[16]:


plt.close("all")
plot_single_sequence(wingbeat_df, 37, marker_name="right_wingtip_z", x_axis="time")
plot_single_sequence(wingbeat_df, 37, marker_name="right_wingtip_x", x_axis="time")


# In[6]:


hawk3d = Hawk3D("../data/mean_hawk_shape.csv")

seqList = wingbeat_df['seqID'].unique()
seqID = seqList[60]
seqInfo = wingbeat_df[wingbeat_df['seqID'] == seqID]
markers, times = load_sequence_data(wingbeat_df, "04_09_038_1", marker_column_names)

markers = markers.reshape(-1, 8, 3)

animate_plotly_compare(
        hawk3d,
        keypoints_frames_list=[
            markers,             # Use the keypoints returned by the function
        ])


# ## Remove Duplicated Time Frames

# In[8]:


wingbeat_df = remove_time_duplicates(wingbeat_df)


# ## Test how many Modes are appropriate

# In[9]:


fig, ax, sorted_amps = plot_amplitude_ranking(
        markers,
        times,
        max_modes=20,
        d=2, # Match delay used in main analysis
        eig_constraints={"conjugate_pairs"} # Match notebook constraint
    )
plt.show() # Display the plot generated by the function


# ## DMD on Single Wingbeat Example

# In[13]:


times, markers, Lambda, Modes, bn, Psi, phase_shifts, dmd_results, keypoints = run_single_wingbeat_dmd(
    bird_name ="Toothless",
    perch_dist="9m",
    turn="Straight",
    behaviour="Flapping",
    seqID="04_09_038_1",
    eig_constraints={"conjugate_pairs"},
    n_modes=10,
    d=2,
    verbose=True
)



# ## Compare Original Sequence with DMD Reconstruction

# In[14]:


hawk3d = Hawk3D("../data/mean_hawk_shape.csv")

print("Original Markers Shape:", markers.shape)
print("DMD Keypoints Shape:", keypoints.shape)

# Take away the first frame and add an extra frame to the end
# keypoints = np.concatenate([keypoints[1:], keypoints[-1:]], axis=0)

animate_plotly_compare(
        hawk3d,
        keypoints_frames_list=[
            keypoints, # Use the reshaped original markers
            markers             # Use the keypoints returned by the function
        ])



# ## Upsample the Data
# 
# 

# In[ ]:


# Original time points
# print(times)
print(times.shape)
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
expanded_markers = expanded_markers.reshape(-1, 8, 3)

print(expanded_markers.shape)

hawk3d = Hawk3D("../data/mean_hawk_shape.csv")
average_shape = hawk3d.markers
num_markers = 8
keypoints_forecast = run_forecast(dmd_results, fake_time, average_shape, num_markers)

# Animate both sequences
animate_plotly_compare(hawk3d, keypoints_frames_list=[expanded_markers, keypoints_forecast], colours=["blue", "red"])


# ## Make the Wingbeat continue

# In[11]:


# Original time points
# print(times)
print(times.shape)
# Create a time that is 5 times longer, no interpolation
fake_time = np.linspace(times[0], times[-1]*3, len(times)*3)
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
expanded_markers = expanded_markers.reshape(-1, 8, 3)

print(expanded_markers.shape)

hawk3d = Hawk3D("../data/mean_hawk_shape.csv")
average_shape = hawk3d.markers
num_markers = 8
keypoints = run_forecast(dmd_results, fake_time, average_shape, num_markers)

# Animate both sequences
animate_plotly_compare(hawk3d, keypoints_frames_list=[expanded_markers, keypoints], colours=["blue", "red"])


# ## Find Mode Frequencies

# In[89]:


n_Modes = 10
Lambda, Modes, bn, Psi, PhaseShifts = reorder_dmd_results(dmd_results, num_markers=8, nModes=n_Modes)
for i_Mode in range(n_Modes):
    print(f"Frequency of Mode {i_Mode}: {Lambda[i_Mode]}")


# ## Reconstruct the Modes

# In[ ]:


mode_0_1_keypoints = reconstruct_specific_modes(times, dmd_results, [0,1])
mode_2_3_keypoints = reconstruct_specific_modes(times, dmd_results, [2,3])
mode_4_5_keypoints = reconstruct_specific_modes(times, dmd_results, [4,5])
mode_6_7_keypoints = reconstruct_specific_modes(times, dmd_results, [6,7])
mode_8_9_keypoints = reconstruct_specific_modes(times, dmd_results, [8,9])

hawk3d = Hawk3D("../data/mean_hawk_shape.csv")
# make markers twice as long
animate_plotly_compare(hawk3d, keypoints_frames_list=[mode_6_7_keypoints,mode_2_3_keypoints])


# ## Forecast With just 3 Modes

# In[96]:


# Original time points
print(times.shape)
# Create a time that is 3 times longer
fake_time = np.linspace(times[0], times[-1]*3, len(times)*3)

# Create expanded original data by repeating frames
expanded_markers = []
current_frame = 0

for t in fake_time:
    # Keep using same frame until we hit a time point in the original data
    while current_frame < len(times)-1 and t >= times[current_frame+1]:
        current_frame += 1
    expanded_markers.append(markers[current_frame].copy())

expanded_markers = np.array(expanded_markers)
expanded_markers = expanded_markers.reshape(-1, 8, 3)

print(expanded_markers.shape)

hawk3d = Hawk3D("../data/mean_hawk_shape.csv")
average_shape = hawk3d.markers
num_markers = 8

# Use the modified function to:
# 1. Set the first mode (index 0) to zero frequency
# 2. Only include modes 0, 2, 3, 6, and 7
keypoints = run_forecast_with_modified_modes(
    dmd_results, 
    fake_time, 
    average_shape, 
    num_markers,
    mode_indices_to_zero=[0,1],  # Set first mode to zero frequency
    selected_mode_indices=[0, 1, 2, 3, 6, 7]  # Only use these modes
)

# Animate both sequences
animate_plotly_compare(hawk3d, keypoints_frames_list=[expanded_markers, keypoints], colours=["blue", "red"])


# ## Plot the wingtip forecast in z

# In[45]:


# Have the colours switch when the time goes beyond the original time points
colours = ["blue" if t < times[-1] else "red" for t in fake_time]

fig, ax = plt.subplots(1,1, figsize=(10,5), tight_layout=True)
ax.scatter(fake_time, keypoints[:,0,2], 
    s=10, c=colours)


# ## Force the slowest mode to have 0 frequency in the forecast

# In[35]:


# Modify the first two modes (indices 0 and 1) to have zero frequency
modified_omega, modified_Psi, modified_amplitudes = modify_mode_frequencies(Lambda, Psi, bn, mode_indices_to_zero=[0, 1])

# Print modified mode information
print("\nModified modes (first two pairs set to zero frequency):")
print_mode_info(modified_omega, modified_amplitudes)




# In[36]:


# Original time points
print(times.shape)
# Create a time that is 3 times longer
fake_time = np.linspace(times[0], times[-1]*3, len(times)*3)

# Create expanded original data by repeating frames
expanded_markers = []
current_frame = 0

for t in fake_time:
    # Keep using same frame until we hit a time point in the original data
    while current_frame < len(times)-1 and t >= times[current_frame+1]:
        current_frame += 1
    expanded_markers.append(markers[current_frame].copy())

expanded_markers = np.array(expanded_markers)
expanded_markers = expanded_markers.reshape(-1, 8, 3)

print(expanded_markers.shape)

hawk3d = Hawk3D("../data/mean_hawk_shape.csv")
average_shape = hawk3d.markers
num_markers = 8

# Use the new function to forecast with modified frequencies
# This will set the first two modes (and their conjugate pairs) to zero frequency
keypoints = run_forecast_with_modified_modes(dmd_results, fake_time, average_shape, num_markers, mode_indices_to_zero=[0, 1])

# Animate both sequences
animate_plotly_compare(hawk3d, keypoints_frames_list=[expanded_markers, keypoints], colours=["blue", "red"])


# In[34]:


def print_mode_info(omega, amplitudes, n_modes=None):
    """
    Prints information about DMD modes including frequencies and amplitudes.
    
    Parameters:
        omega: Array of frequencies (imaginary parts of eigenvalues)
        amplitudes: Array of mode amplitudes
        n_modes: Optional number of modes to print. If None, prints all modes.
    """
    # Sort by amplitude magnitude
    sort_indices = np.argsort(np.abs(amplitudes))[::-1]
    omega_sorted = omega[sort_indices]
    amplitudes_sorted = np.abs(amplitudes[sort_indices])
    
    if n_modes is None:
        n_modes = len(omega)
    
    print("\nDMD Mode Information (sorted by amplitude):")
    print("-------------------------------------------")
    print("Mode Index | Frequency (Hz) | Amplitude")
    print("-------------------------------------------")
    
    for i in range(n_modes):
        print(f"{i:10d} | {omega_sorted[i]:13.3f} | {amplitudes_sorted[i]:9.3f}")
    print("-------------------------------------------")
    print("Note: Frequencies are in Hz, negative frequencies indicate conjugate pairs")


print_mode_info(Lambda, bn)


# In[39]:


# Original time points
print(times.shape)
# Create a time that is 3 times longer
fake_time = np.linspace(times[0], times[-1]*3, len(times)*3)

# Create expanded original data by repeating frames
expanded_markers = []
current_frame = 0

for t in fake_time:
    # Keep using same frame until we hit a time point in the original data
    while current_frame < len(times)-1 and t >= times[current_frame+1]:
        current_frame += 1
    expanded_markers.append(markers[current_frame].copy())

expanded_markers = np.array(expanded_markers)
expanded_markers = expanded_markers.reshape(-1, 8, 3)

print(expanded_markers.shape)

hawk3d = Hawk3D("../data/mean_hawk_shape.csv")
average_shape = hawk3d.markers
num_markers = 8

# Use the modified function to:
# 1. Set the first mode (index 0) to zero frequency
# 2. Only include modes 0, 2, 3, 6, and 7
keypoints = run_forecast_with_modified_modes(
    dmd_results, 
    fake_time, 
    average_shape, 
    num_markers,
    mode_indices_to_zero=[0, 1],  # Set first mode to zero frequency
    selected_mode_indices=[0, 1, 2, 3, 6, 7]  # Only use these modes
)

# Animate both sequences
animate_plotly_compare(hawk3d, keypoints_frames_list=[expanded_markers, keypoints], colours=["blue", "red"])


# ## Plot the Forecasted Wingtip in Z over time

# In[44]:


fig, ax = plt.subplots(1,1, figsize=(10,5), tight_layout=True)
ax.scatter(fake_time, keypoints[:,0,2], 
    s=10, c=fake_time)

