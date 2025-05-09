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
                     run_dmd_matrix,
                     plot_amplitude_ranking_matrix,
                     run_forecast,
                     reorder_dmd_results,
                     reconstruct_specific_modes,
                     make_unilateral_keypoints,
                     project_into_pca_space,
                     project_into_coordinate_space,
                     create_scores_info_df,
                     plot_2d_markers, 
                     plot_markers_overDist, 
                     plot_single_sequence, 
                     plot_score_multi_PCs)

np.set_printoptions(suppress=True, precision=3)
# Make matplotlib use the font Andale Mono
plt.rcParams['font.family'] = 'Andale Mono'




# In[18]:


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
# This is a single initial wingbeat from take-off by one hawk called Toothless. There are around 175 flights at 250 fps. We can plot them over the flight to see the markers changing in x, y, and z. 
# 
# Note the x axis is the horizontal distance to the perch as measured at the centre of mass of the bird. 

# In[96]:


plot_markers_overDist(wingbeat_df, marker_column_names, x_axis='time')


# # Plot the Marker Trajectories in 2d

# In[97]:


plot_2d_markers(wingbeat_df, marker_column_names)


# ## Find Sequence Lengths

# In[19]:


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


# ## Plot a Single Sequence

# In[20]:


plt.close("all")
plot_single_sequence(wingbeat_df, 60, marker_name="right_wingtip_z", x_axis="time")
plot_single_sequence(wingbeat_df, 60, marker_name="right_wingtip_x", x_axis="time")


# In[21]:


hawk3d = Hawk3D("../data/mean_hawk_shape.csv")

seqList = wingbeat_df['seqID'].unique()
seqID = seqList[60]
seqInfo = wingbeat_df[wingbeat_df['seqID'] == seqID]
markers, times = load_sequence_data(wingbeat_df, "04_09_051_2", marker_column_names)

markers = markers.reshape(-1, 8, 3)

animate_plotly_compare(
        hawk3d,
        keypoints_frames_list=[
            markers,             # Use the keypoints returned by the function
        ])


# ## Remove Duplicated Time Frames

# In[22]:


wingbeat_df = remove_time_duplicates(wingbeat_df)


# ## Project into Morphing Shape Space
# 
# > 2025-04-29
# 
# Here we are going to run DMD on the flight data in morphing shape space rather than raw coordinates. 
# 
# We will import the morphing shape modes (principal components) and project into them. The main difficulty is that the data is bilateral and the principal components are unilateral: as a result we will have all the PC mode scores for the right side in the first columns and then the next columns will be for the left side. It is important there is one row per timestep only. 
# 
# We will then run DMD on this matrix and once we have the spatial modes we will split them again and reconstruct back into the left and right sides. 
# 
# 

# In[23]:


# Load transformed data (no whole body rotation)
markers_with_tailpack = np.load("../data/raw/transformed_markers_with_tailpack.npy")
# Remove the tailpack
markers = markers_with_tailpack[:, :8, :]

print(f"Markers shape {markers.shape}")

# Load marker info
combined_frame_info_df = pd.read_csv("../data/raw/combined_frame_info_df.csv")

print(f"combined_frame_info_df.shape: {combined_frame_info_df.shape}")

# Load the principal components
principal_components = np.load("../data/PCA_results/unilateral_principal_components.npy")

# Load the mean of the principal components
mu = np.load("../data/PCA_results/unilateral_mu.npy")


# In[32]:


# Split the markers into left and right, which is the order right left right left
unilateral_original, right_bool = make_unilateral_keypoints(markers)

# Project the data into the principal components space
# We only want the first 9 components.
nComponents = 12
unilateral_scores = project_into_pca_space(unilateral_original, mu, principal_components[:nComponents,:])
print(f"Unilateral scores shape: {unilateral_scores.shape}")

# We now split the scores into left and right 
right_scores = unilateral_scores[right_bool]
left_scores = unilateral_scores[~right_bool]

# Concatenate horizontally into a bilateral score matrix
scores = np.concatenate([right_scores, left_scores], axis=1)
print(f"Scores shape: {scores.shape}")

# Specify which seqID we want
# We only want the first wingbeat so we will select with the time less than0.22
bool_select = (combined_frame_info_df['seqID'] == '04_09_051_2') & (combined_frame_info_df['time'] < 0.55)
original_keypoints = markers[bool_select,:,:]
# Select the scores for this seqID
scores_select = scores[bool_select, :]
times_select = combined_frame_info_df['time'][bool_select].values

# Quick plot of the scores
fig, ax = plt.subplots()
ax.plot(times_select, scores_select[:, 5])
plt.show()


# In[ ]:





# In[26]:


print(scores_select.shape)
print(times_select.shape)
print(bool_select.shape)


# ## Test how many Modes are appropriate

# In[27]:


fig, ax, sorted_amplitudes = plot_amplitude_ranking_matrix(
    scores_select,
    times = times_select,
    max_modes=20,
    d=2,
    eig_constraints={"conjugate_pairs"},
    figsize=(6, 2)
)


# ## Run DMD
# 

# In[28]:


# Run DMD
Lambda, Modes, bn, Psi, phase_shifts, dmd_results, _ = run_dmd_matrix(
    scores_select,
    times_select,
    n_modes=10,
    d=2,
    eig_constraints={"conjugate_pairs"},
    verbose=True
)

# Use phase-aware reconstruction
reconstruction = reconstruct_dmd(
    times_select,
    np.imag(dmd_results.eigs),  # frequencies
    dmd_results.modes,          # modes
    dmd_results.amplitudes      # amplitudes
)

# In[36]:


print(f"Original scores shape: {scores_select.shape}")
print(f"Reconstructed scores shape: {reconstruction.shape}")
print(f"Original times shape: {times_select.shape}")

# Now to split the scores into left and right, splitting the columns 
# right_reconstructed_scores = reconstruction[:,:9]
# left_reconstructed_scores = reconstruction[:,9:]
right_reconstructed_scores = reconstruction[:,:12]
left_reconstructed_scores = reconstruction[:,12:]

# Make sure we have mu for each from the input data
hawk3d = Hawk3D("../data/mean_hawk_shape.csv")
mu = hawk3d.right_markers

# Need to reconstruct back into keypoint xyz data using the principal components 
right_keypoints = project_into_coordinate_space(
    scores=right_reconstructed_scores,
    mu=mu,
    principal_components=principal_components[:nComponents,:],
    n_markers= 4  # adjust if needed
)
print(right_keypoints.shape)


# Need to reconstruct back into keypoint xyz data using the principal components 
left_keypoints = project_into_coordinate_space(
    scores=left_reconstructed_scores,
    mu=mu,
    principal_components=principal_components[:nComponents,:],
    n_markers= 4  # adjust if needed
)
# Reflect x so the left keypoints are correct
left_keypoints[:,:,0] = -left_keypoints[:,:,0]

# Combine the right and left keypoints [n,4,3] to [n, 8, 3] 
# so it is right, left, right, left
keypoints = np.zeros([79, 8, 3])
keypoints[:,1::2,:] = right_keypoints
keypoints[:,0::2,:] = left_keypoints

print(keypoints.shape)


# In[37]:


animate_plotly_compare(hawk3d,[keypoints, original_keypoints])


# In[ ]:


times, markers, Lambda, Modes, bn, Psi, phase_shifts, dmd_results,  keypoints = run_DMD_sequence(
    
)


times, markers, Lambda, Modes, bn, Psi, phase_shifts, dmd_results, keypoints = run_single_wingbeat_dmd(
    bird_name ="Toothless",
    perch_dist="9m",
    turn="Straight",
    behaviour="Initial",
    seqID="04_09_051_2",
    eig_constraints={"conjugate_pairs"},
    n_modes=10,
    d=2,
    verbose=True
)



# In[ ]:





# In[10]:


fig, ax, sorted_amps = plot_amplitude_ranking(
        markers,
        times,
        max_modes=20,
        d=2, # Match delay used in main analysis
        eig_constraints={"conjugate_pairs"} # Match notebook constraint
    )
plt.show() # Display the plot generated by the function


# ## DMD on Single Wingbeat Example

# In[11]:


times, markers, Lambda, Modes, bn, Psi, phase_shifts, dmd_results, keypoints = run_dmd_on_markers(
    bird_name ="Toothless",
    perch_dist="9m",
    turn="Straight",
    behaviour="Initial",
    seqID="04_09_051_2",
    eig_constraints={"conjugate_pairs"},
    n_modes=10,
    d=2,
    verbose=True
)



# ## Compare Original Sequence with DMD Reconstruction

# In[12]:


hawk3d = Hawk3D("../data/mean_hawk_shape.csv")

print("Original Markers Shape:", markers.shape)
print("DMD Keypoints Shape:", keypoints.shape)

# Take away the first frame and add an extra frame to the end
keypoints = np.concatenate([keypoints[1:], keypoints[-1:]], axis=0)

animate_plotly_compare(
        hawk3d,
        keypoints_frames_list=[
            keypoints, # Use the reshaped original markers
            markers             # Use the keypoints returned by the function
        ])



# ## Upsample the Data
# 
# 

# In[13]:


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
keypoints = run_forecast(dmd_results, fake_time, average_shape, num_markers)

# Animate both sequences
animate_plotly_compare(hawk3d, keypoints_frames_list=[expanded_markers, keypoints], colours=["blue", "red"])


# ## Make the Wingbeat continue

# In[14]:


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

# In[15]:


n_Modes = 10
Lambda, Modes, bn, Psi, PhaseShifts = reorder_dmd_results(dmd_results, num_markers=8, nModes=n_Modes)
for i_Mode in range(n_Modes):
    print(f"Frequency of Mode {i_Mode}: {Lambda[i_Mode]}")


# ## Reconstruct the Modes

# In[16]:


mode_0_1_keypoints = reconstruct_specific_modes(times, dmd_results, [0,1])
mode_2_3_keypoints = reconstruct_specific_modes(times, dmd_results, [2,3])
mode_4_5_keypoints = reconstruct_specific_modes(times, dmd_results, [4,5])
mode_6_7_keypoints = reconstruct_specific_modes(times, dmd_results, [6,7])
mode_8_9_keypoints = reconstruct_specific_modes(times, dmd_results, [8,9])

hawk3d = Hawk3D("../data/mean_hawk_shape.csv")
# make markers twice as long
animate_plotly_compare(hawk3d, keypoints_frames_list=[mode_0_1_keypoints,mode_4_5_keypoints])


# ## Reconstruct the DMD Modes into PCA Morphing Shape Space

# In[17]:


principal_components = np.load("../data/PCA_results/unilateral_principal_components.npy")
mu = np.load("../data/PCA_results/unilateral_mu.npy")

print(mode_0_1_keypoints.shape)

unilateral_0_1, left_right_0_1 = make_unilateral_keypoints(mode_0_1_keypoints)
# unilateral_2_3 = make_unilateral_keypoints(mode_2_3_keypoints)
# unilateral_4_5 = make_unilateral_keypoints(mode_4_5_keypoints)
# unilateral_6_7 = make_unilateral_keypoints(mode_6_7_keypoints)

hawk3d = Hawk3D("../data/mean_hawk_shape.csv")

scores_0_1 = project_into_pca_space(unilateral_0_1, hawk3d.right_markers, principal_components)
scores_0_1_df = create_scores_info_df(scores_0_1, seqInfo, left_right_0_1)


# In[18]:


left_scores_df = scores_0_1_df[scores_0_1_df['Left'] == 1]
right_scores_df = scores_0_1_df[scores_0_1_df['Left'] == 0]

_ = plot_score_multi_PCs(left_scores_df,right_scores_df, x_column='time')
plt.show()
# Make matplotlib use the font Andale Mono


# In[19]:


mode_2_3_keypoints = reconstruct_specific_modes(times, dmd_results, [2,3])

unilateral_2_3, left_right_2_3 = make_unilateral_keypoints(mode_2_3_keypoints)

scores_2_3 = project_into_pca_space(unilateral_2_3, hawk3d.right_markers, principal_components)
scores_2_3_df = create_scores_info_df(scores_2_3, seqInfo, left_right_2_3)


left_scores_2_3_df = scores_2_3_df[scores_2_3_df['Left'] == 1]
right_scores_2_3_df = scores_2_3_df[scores_2_3_df['Left'] == 0]

_ = plot_score_multi_PCs(left_scores_2_3_df,right_scores_2_3_df, x_column='time')
plt.show()


# In[20]:


mode_4_5_keypoints = reconstruct_specific_modes(times, dmd_results, [4,5])

unilateral_4_5, left_right_4_5 = make_unilateral_keypoints(mode_4_5_keypoints)

scores_4_5 = project_into_pca_space(unilateral_4_5, hawk3d.right_markers, principal_components)
scores_4_5_df = create_scores_info_df(scores_4_5, seqInfo, left_right_4_5)


left_scores_4_5_df = scores_4_5_df[scores_4_5_df['Left'] == 1]
right_scores_4_5_df = scores_4_5_df[scores_4_5_df['Left'] == 0]

_ = plot_score_multi_PCs(left_scores_4_5_df,right_scores_4_5_df, x_column='time')
plt.show()


# In[21]:


mode_0_1_4_5_keypoints = reconstruct_specific_modes(times, dmd_results, [0, 1, 4, 5,6,7])

# In your notebook cell [119] or similar
unilateral_0_1_4_5, left_right_0_1_4_5 = make_unilateral_keypoints(mode_0_1_4_5_keypoints)
scores_0_1_4_5 = project_into_pca_space(unilateral_0_1_4_5, mu, principal_components)
scores_0_1_4_5_df = create_scores_info_df(scores_0_1_4_5, seqInfo, left_right_0_1_4_5)

left_scores_0_1_4_5_df = scores_0_1_4_5_df[scores_0_1_4_5_df['Left'] == 1]
right_scores_0_1_4_5_df = scores_0_1_4_5_df[scores_0_1_4_5_df['Left'] == 0]

_ = plot_score_multi_PCs(left_scores_0_1_4_5_df,right_scores_0_1_4_5_df, x_column='time')
plt.show()


# In[22]:


hawk3d = Hawk3D("../data/mean_hawk_shape.csv")
animate_plotly_compare(hawk3d, keypoints_frames_list=[mode_0_1_4_5_keypoints, markers])


# ## Project wingbeat xyz coordinates into PCA space
# 
# 

# In[23]:


unilateral_original, left_right_original = make_unilateral_keypoints(markers)
scores_original = project_into_pca_space(unilateral_original, mu, principal_components)
scores_original_df = create_scores_info_df(scores_original, seqInfo, left_right_original)

left_scores_original_df = scores_original_df[scores_original_df['Left'] == 1]
right_scores_original_df = scores_original_df[scores_original_df['Left'] == 0]


_ = plot_score_multi_PCs(right_scores_original_df, right_scores_0_1_4_5_df, x_column='time')
plt.show()


# In[24]:


plt.plot(times, keypoints[:, 0, 2], label='Marker 1 Z')
plt.plot(times, keypoints[:, 1, 2], label='Marker 2 Z')
plt.xlabel("Time (s)")
plt.ylabel("Z Position")
plt.title("Forecasted DMD Z Trajectories")
plt.legend()
plt.grid(True)
plt.show()


# In[25]:


# 📊 Plot example marker trace
plt.figure(figsize=(5, 3))
plt.plot(times, markers[:, 0], label="Marker 1 (x)")
plt.plot(times, keypoints[:, 0, 0], label="DMD forecast (x)", linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Marker vs Forecast (DMD)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[26]:


original_markers = markers.reshape(-1, 8, 3)
hawk3d = Hawk3D("../data/mean_hawk_shape.csv")
animate_plotly_compare(hawk3d, keypoints_frames_list=[original_markers, keypoints], colours=["blue", "red"])

