#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'widget')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from mpl_toolkits.mplot3d import Axes3D

from pydmd import DMD
from pydmd.bopdmd import BOPDMD
from pydmd.plotter import plot_eigs
from pydmd.plotter import plot_summary
from pydmd.preprocessing.hankel import hankel_preprocessing

from morphing_birds import Hawk3D, plot, animate, animate_compare

from BirdDMD import runDMD, aggregateDMD

np.set_printoptions(suppress=True, precision=3)



# ## Load Bird Data

# In[ ]:





# In[ ]:





# ## Remove Duplicated Time Frames

# In[223]:


import os

hawk_name = "Toothless"
behaviour = "Initial"
perch_distance = "9m"
turn = "Straight"
nModes = 6

save_path = f"../data/2024-07-09_Bilateral{behaviour}{hawk_name}_{perch_distance}_DMD_results/{str(nModes)}Modes/"

# Check if exists, otherwise create
if not os.path.exists(save_path):
    os.makedirs(save_path)

runDMD.dmd_loop_seqs(hawk_name, 
                    save_path, 
                    behaviour,
                    perch_distance=perch_distance,
                    turn = turn,
                    n_Modes = nModes, 
                    min_seq_length = 20)


# In[227]:


# Load results
# DMD_results_6Modes = runDMD.load_dmd_results(wingbeat_df,save_path)
bird_name = "Toothless"
behaviour = "TwoFlaps"
turn = None
nModes = 6

save_path = f"../data/2024-07-09_Bilateral{bird_name}_{perch_distance}_DMD_results/{str(nModes)}Modes/"
print(save_path)
wingbeat_df, marker_column_names = runDMD.load_bird_data(bird_name, behaviour, perch_distance)
DMD_results_12m6Modes = runDMD.load_dmd_results(wingbeat_df,save_path)

behaviour = "Initial"
save_path = f"../data/2024-07-09_BilateralInitial{bird_name}_{perch_distance}_DMD_results/{str(nModes)}Modes/"
print(save_path)
wingbeat_df, marker_column_names = runDMD.load_bird_data(bird_name, behaviour, perch_distance)
DMD_results_Init12m6Modes = runDMD.load_dmd_results(wingbeat_df,save_path)


bird_name = "Charmander"
behaviour = "TwoFlaps"

save_path = f"../data/2024-07-09_Bilateral{bird_name}_{perch_distance}_DMD_results/{str(nModes)}Modes/"
wingbeat_df, marker_column_names = runDMD.load_bird_data(bird_name, behaviour, perch_distance)
DMD_results_Ch12m6Modes = runDMD.load_dmd_results(wingbeat_df,save_path)

bird_name = "Ruby"
save_path = f"../data/2024-07-09_Bilateral{bird_name}_{perch_distance}_DMD_results/{str(nModes)}Modes/"
wingbeat_df, marker_column_names = runDMD.load_bird_data(bird_name, behaviour, perch_distance)
DMD_results_Ru12m6Modes = runDMD.load_dmd_results(wingbeat_df,save_path)

bird_name = "Drogon"
save_path = f"../data/2024-07-09_Bilateral{bird_name}_{perch_distance}_DMD_results/{str(nModes)}Modes/"
wingbeat_df, marker_column_names = runDMD.load_bird_data(bird_name, behaviour, perch_distance)
DMD_results_Dr12m6Modes = runDMD.load_dmd_results(wingbeat_df,save_path)




# In[182]:


np.load("../data/2024-07-09_BilateralToothless_12m_DMD_results/6Modes/04_12_101_dmd_results.npz", allow_pickle=True)


# In[228]:


aggregateDMD.plot_dmd_histograms(DMD_results_12m6Modes, modes=[0, 2, 4])
aggregateDMD.plot_dmd_histograms(DMD_results_Init12m6Modes, modes=[0, 2, 4])
aggregateDMD.plot_dmd_histograms(DMD_results_Ru12m6Modes, modes=[0, 2, 4])
aggregateDMD.plot_dmd_histograms(DMD_results_Dr12m6Modes, modes=[0, 2, 4])
aggregateDMD.plot_dmd_histograms(DMD_results_Ch12m6Modes, modes=[0, 2, 4])



# In[212]:


fig, axes = plt.subplots(1, 8, figsize=(10, 2), sharey=False, tight_layout=True)
counter = 0
for ii in range(0,8,2):
    lambda_i_values = DMD_results_12m6Modes['Lambda'].apply(lambda x: x[ii])
    axes[counter].hist(abs(lambda_i_values), bins=50, alpha=0.4, color='dodgerblue', range=(0, 200))


    axes[counter].set_xticks([0, 15, 30, 60, 120])
    axes[counter].set_xlim(0, 120)
    axes[counter].grid(alpha=0.3)
    axes[counter].tick_params(axis='x', which='major', labelsize=6, rotation=90)
    axes[counter].tick_params(axis='y', which='major', labelsize=6)
    axes[counter].set_title(f"Mode {ii+1}", fontsize=8)
    axes[counter].set_xlabel("Frequency (Hz)", fontsize=8)

    counter = counter + 1

fig, axes = plt.subplots(1, 6, figsize=(10, 2), sharey=False, tight_layout=True)
counter = 0
for ii in range(0,6):
    lambda_i_values = DMD_results_Ch12m6Modes['Lambda'].apply(lambda x: x[ii])
    axes[counter].hist(abs(lambda_i_values), bins=50, alpha=0.4, color='dodgerblue', range=(0, 200))


    axes[counter].set_xticks([0, 15, 30, 60, 120])
    axes[counter].set_xlim(0, 120)
    axes[counter].grid(alpha=0.3)
    axes[counter].tick_params(axis='x', which='major', labelsize=6, rotation=90)
    axes[counter].tick_params(axis='y', which='major', labelsize=6)
    axes[counter].set_title(f"Mode {ii+1}", fontsize=8)
    axes[counter].set_xlabel("Frequency (Hz)", fontsize=8)

    counter = counter + 1

fig, axes = plt.subplots(1, 6, figsize=(10, 2), sharey=False, tight_layout=True)
counter = 0
for ii in range(0,6):
    lambda_i_values = DMD_results_Ru12m6Modes['Lambda'].apply(lambda x: x[ii])
    axes[counter].hist(abs(lambda_i_values), bins=50, alpha=0.4, color='dodgerblue', range=(0, 200))

    axes[counter].set_xticks([0, 15, 30, 60, 120])
    axes[counter].set_xlim(0, 120)
    axes[counter].grid(alpha=0.3)
    axes[counter].tick_params(axis='x', which='major', labelsize=6, rotation=90)
    axes[counter].tick_params(axis='y', which='major', labelsize=6)
    axes[counter].set_title(f"Mode {ii+1}", fontsize=8)
    axes[counter].set_xlabel("Frequency (Hz)", fontsize=8)

    counter = counter + 1

fig, axes = plt.subplots(1, 6, figsize=(10, 2), sharey=False, tight_layout=True)
counter = 0
for ii in range(0,6):
    lambda_i_values = DMD_results_Dr12m6Modes['Lambda'].apply(lambda x: x[ii])
    axes[counter].hist(abs(lambda_i_values), bins=50, alpha=0.4, color='dodgerblue', range=(0, 200))


    axes[counter].set_xticks([0, 15, 30, 60, 120])
    axes[counter].set_xlim(0, 120)
    axes[counter].grid(alpha=0.3)
    axes[counter].tick_params(axis='x', which='major', labelsize=6, rotation=90)
    axes[counter].tick_params(axis='y', which='major', labelsize=6)
    axes[counter].set_title(f"Mode {ii} & {ii+1}", fontsize=8)
    axes[counter].set_xlabel("Frequency (Hz)", fontsize=8)

    counter = counter + 1





    


# ## Running Many DMDs

# In[4]:


def load_sequence_data(df, seqID, marker_column_names):
    """Load markers and time data for a given sequence ID."""
    condition = df['seqID'] == seqID
    markers = df[condition][marker_column_names].to_numpy().astype(np.float64)
    times = df[condition]['time'].to_numpy().astype(np.float64)
    return markers, times

def normalise_data(markers, average_shape):
    """Normalise marker data by subtracting the average shape."""
    return markers - average_shape, average_shape

def perform_dmd(markers, times, n_Modes):
    """Perform Dynamic Mode Decomposition on the marker data."""
    markers = markers.T  # Transpose data
    dmd_processor = hankel_preprocessing(BOPDMD(svd_rank=n_Modes, 
                                                eig_constraints={"imag", "conjugate_pairs"}), 
                                                d=2)
    
    dmd_processor.fit(markers, t=times[1:])
    forecast = dmd_processor.forecast(times)
    return dmd_processor, forecast

def run_forecast(forecast, average_shape, num_markers):
    """Run the forecast to get the predicted markers."""
    forecast = np.real(forecast[:num_markers*3,:])
    forecast_plus_mean = forecast.T + average_shape

    keypoints = forecast_plus_mean.reshape(-1, num_markers, 3)

    # Take away the first frame and add an extra frame to the end
    keypoints = np.concatenate([keypoints[1:], keypoints[-1:]], axis=0)

    return keypoints

def reorder_dmd_results(dmd_results, num_markers, nModes):
    """Reorder DMD results based on amplitude size and extract components."""
    bn = np.argsort(dmd_results.amplitudes)[::-1]
    Psi = dmd_results.modes
    Psi = Psi[:num_markers*3, bn]
    
    Lambda = np.imag(dmd_results.eigs)[bn]
    Modes = np.real(Psi).reshape(num_markers, 3, nModes)

    return Lambda, Modes, dmd_results.amplitudes[bn], Psi

def save_results_npz(save_path, seqID, times, markers, Lambda, Modes, bn, Psi, keypoints):
    """Save DMD results to a single .npz file for a given sequence ID."""
    np.savez_compressed(f"{save_path}{seqID}_dmd_results.npz", 
                        times = times,
                        markers = markers,
                        Lambda=Lambda, 
                        Modes=Modes, 
                        bn=bn, 
                        Psi=Psi, 
                        forecast_keypoints=keypoints)

def load_results_npz(save_path, seqID):
    """Load DMD results for a specific sequence ID from a .npz file."""
    try:
        data = np.load(f"{save_path}{seqID}_dmd_results.npz")
    except FileNotFoundError:
        print(f"File {save_path}{seqID}_dmd_results.npz not found.")
        return None, None, None, None, None
    
    times = data['times']
    markers = data['markers']
    Lambda = data['Lambda']
    Modes = data['Modes']
    bn = data['bn']
    Psi = data['Psi']
    keypoints = data['forecast_keypoints']
    
    return times, markers, Lambda, Modes, bn, Psi, keypoints


# --- Main script ---

def process_and_store_sequences(df, nModes, marker_column_names, average_shape, save_path):
    """Process and store DMD results for all sequences in the DataFrame."""
    num_markers = 8

    for seqID in df['seqID'].unique():

        markers, times = load_sequence_data(df, seqID, marker_column_names)

        if markers.shape[0] <= nModes+1:
            print(f"Skipping sequence {seqID} due to insufficient timesteps.")
            continue

        normalised_markers, _ = normalise_data(markers, average_shape)
        dmd_results, forecast = perform_dmd(normalised_markers, times, nModes)
        Lambda, Modes, bn, Psi = reorder_dmd_results(dmd_results, num_markers, nModes)
        keypoints = run_forecast(forecast, average_shape, num_markers)
        markers = markers.reshape(-1, num_markers, 3)
        save_results_npz(save_path, seqID, times, markers, Lambda, Modes, bn, Psi, keypoints)



# ## Run on Flights

# In[46]:


hawk_name = "Toothless"

wingbeat_df, marker_column_names = runDMD.load_bird_data(hawk_name, perch_distance = "12m", bilateral=bilateral)
wingbeat_df = runDMD.remove_time_duplicates(wingbeat_df)
seqList = wingbeat_df['seqID'].unique()
markers, times = runDMD.load_sequence_data(wingbeat_df, seqList[4], marker_column_names)

average_shape, num_markers = runDMD.get_average_shape(markers)
normalised_markers, _ = runDMD.normalise_data(markers, average_shape)


nModes = 6
dmd_results, forecast = runDMD.perform_dmd(normalised_markers, times, nModes)
Lambda, Modes, bn, Psi = runDMD.reorder_dmd_results(dmd_results, num_markers, nModes)

keypoints = runDMD.run_forecast(forecast, average_shape, num_markers)

markers = markers.reshape(-1, num_markers, 3)



# In[45]:


hawk3d = Hawk3D("../data/mean_hawk_shape.csv")
animate_compare(hawk3d, keypoints_frames_list=[markers, keypoints])


# In[47]:


plt.close()


# In[21]:


import warnings
warnings.filterwarnings("ignore")

hawk_name = "Ruby"
# perch_distance = "9m"
# turn = "Left"
wingbeat_df, marker_column_names = load_bird_data(hawk_name, bilateral=bilateral)

wingbeat_df = handle_duplicates(wingbeat_df)

hawk3d = Hawk3D("../data/mean_hawk_shape.csv")
average_shape = hawk3d.markers.reshape(24,1).T
nModes = 6

save_path = f"../data/2024-06-12_All{hawk_name}_DMD_results/{str(nModes)}Modes/"

# Print the number of sequences
print(f"Processing {wingbeat_df['seqID'].nunique()} sequences with {nModes} modes.")
process_and_store_sequences(wingbeat_df, nModes, marker_column_names, average_shape, save_path)



# # Statistics on Multiple Wingbeat Examples

# In[50]:


import os

def load_dmd_results(wingbeat_df, save_path):

    # Initialize list to store data dictionaries
    DMD_dict = []

    # For each .npz file in the directory, load the numpy arrays inside and stack them
    for seqID in wingbeat_df['seqID'].unique():
        # Check if the file exists
        if not os.path.exists(f"{save_path}{seqID}_dmd_results.npz"):
            continue

        times, markers, Lambda, Modes, bn, Psi, keypoints = load_results_npz(save_path, seqID)

        if times is None:
            continue

        # Append a dictionary of the data to the list
        DMD_dict.append({
            "seqID": seqID,
            "times": times,
            "markers": markers,
            "Lambda": Lambda,
            "Modes": Modes,
            "bn": bn,
            "Psi": Psi,
            "keypoints": keypoints
        })

        

    # Convert the list of dictionaries to a DataFrame
    return pd.DataFrame(DMD_dict)


# Load results from 12 modes
nModes = 12
save_path = f"../data/2024-06-12_12mToothless_DMD_results/{str(nModes)}Modes/"
DMD_results_6Modes = load_dmd_results(wingbeat_df,save_path)


print(DMD_results_12Modes.head())



# In[ ]:


# # Load results from 8 modes
# nModes = 10
# save_path = f"../data/2024-06-12_12mToothless_DMD_results/{str(nModes)}Modes/"
# DMD_results_10Modes = load_dmd_results(wingbeat_df,save_path)



# # Load results from 8 modes
# nModes = 8
# save_path = f"../data/2024-06-12_12mToothless_DMD_results/{str(nModes)}Modes/"
# DMD_results_8Modes = load_dmd_results(wingbeat_df,save_path)

# # Load results from 8 modes
# nModes = 7
# save_path = f"../data/2024-06-12_12mToothless_DMD_results/{str(nModes)}Modes/"
# DMD_results_7Modes = load_dmd_results(wingbeat_df,save_path)


# # Load results from 8 modes
# nModes = 6
# save_path = f"../data/2024-06-12_12mToothless_DMD_results/{str(nModes)}Modes/"
# DMD_results_6Modes = load_dmd_results(wingbeat_df,save_path)


# # Load results from 6 modes
# hawkname = "Toothless"
# perch_distance = "5m"
# bilateral = True
# wingbeat_df, marker_column_names = load_bird_data(hawkname, perch_distance, bilateral)

# nModes = 6
# save_path = f"../data/2024-06-12_{perch_distance}{hawkname}_DMD_results/{str(nModes)}Modes/"
# DMD_results_To5m = load_dmd_results(wingbeat_df,save_path)


# # # Load results from 6 modes
# hawkname = "Toothless"
# perch_distance = "7m"
# bilateral = True
# wingbeat_df, marker_column_names = load_bird_data(hawkname, perch_distance, bilateral)

# nModes = 6
# save_path = f"../data/2024-06-12_{perch_distance}{hawkname}_DMD_results/{str(nModes)}Modes/"
# DMD_results_To7m = load_dmd_results(wingbeat_df,save_path)


# # # Load results from 6 modes
# hawkname = "Toothless"
# perch_distance = "9m"
# bilateral = True
# wingbeat_df, marker_column_names = load_bird_data(hawkname, perch_distance, bilateral)

# nModes = 6
# save_path = f"../data/2024-06-12_{perch_distance}{hawkname}_DMD_results/{str(nModes)}Modes/"
# DMD_results_To9m = load_dmd_results(wingbeat_df,save_path)




# # # Load results from 6 modes
# hawkname = "Toothless"
# perch_distance = "12m"
# bilateral = True
# wingbeat_df, marker_column_names = load_bird_data(hawkname, perch_distance, bilateral)

# nModes = 6
# save_path = f"../data/2024-06-12_{perch_distance}{hawkname}_DMD_results/{str(nModes)}Modes/"
# DMD_results_To12m = load_dmd_results(wingbeat_df,save_path)


# # # Load results from 6 modes
# hawkname = "Toothless"
# perch_distance = "9m"
# bilateral = True
# turn = "Straight"
# wingbeat_df, marker_column_names = load_bird_data(hawkname, perch_distance, bilateral, turn)

# nModes = 6
# save_path = f"../data/2024-06-12_{perch_distance}{turn}{hawkname}_DMD_results/{str(nModes)}Modes/"
# DMD_results_To9mStraight = load_dmd_results(wingbeat_df,save_path)

# # # Load results from 6 modes
# hawkname = "Toothless"
# perch_distance = "9m"
# bilateral = True
# turn = "Right"
# wingbeat_df, marker_column_names = load_bird_data(hawkname, perch_distance, bilateral, turn)

# nModes = 6
# save_path = f"../data/2024-06-12_{perch_distance}{turn}{hawkname}_DMD_results/{str(nModes)}Modes/"
# DMD_results_To9mRight = load_dmd_results(wingbeat_df,save_path)

# # # Load results from 6 modes
# hawkname = "Toothless"
# perch_distance = "9m"
# bilateral = True
# turn = "Left"
# wingbeat_df, marker_column_names = load_bird_data(hawkname, perch_distance, bilateral, turn)

# nModes = 6
# save_path = f"../data/2024-06-12_{perch_distance}{turn}{hawkname}_DMD_results/{str(nModes)}Modes/"
# DMD_results_To9mLeft = load_dmd_results(wingbeat_df,save_path)


# # # Load results from 6 modes
# hawkname = "Toothless"
# perch_distance = "9m"
# bilateral = True
# turn = "Left"
# wingbeat_df, marker_column_names = load_bird_data(hawkname, perch_distance, bilateral, turn)

# nModes = 6
# save_path = f"../data/2024-06-12_{perch_distance}{turn}{hawkname}_DMD_results/{str(nModes)}Modes/"
# DMD_results_To9mLeft = load_dmd_results(wingbeat_df,save_path)




# # # Load results from 6 modes
# hawkname = "Drogon"
# # perch_distance = "9m"
# bilateral = True
# wingbeat_df, marker_column_names = load_bird_data(hawkname, bilateral=bilateral)

# nModes = 6
# save_path = f"../data/2024-06-12_All{hawkname}_DMD_results/{str(nModes)}Modes/"
# DMD_results_Dr = load_dmd_results(wingbeat_df,save_path)

# # # Load results from 6 modes
# hawkname = "Rhaegal"
# # perch_distance = "9m"
# bilateral = True
# wingbeat_df, marker_column_names = load_bird_data(hawkname, bilateral=bilateral)

# nModes = 6
# save_path = f"../data/2024-06-12_All{hawkname}_DMD_results/{str(nModes)}Modes/"
# DMD_results_Rh = load_dmd_results(wingbeat_df,save_path)


# # # Load results from 6 modes
# hawkname = "Ruby"
# # perch_distance = "9m"
# bilateral = True
# wingbeat_df, marker_column_names = load_bird_data(hawkname, bilateral=bilateral)

# nModes = 6
# save_path = f"../data/2024-06-12_All{hawkname}_DMD_results/{str(nModes)}Modes/"
# DMD_results_Ru = load_dmd_results(wingbeat_df,save_path)


# # # Load results from 6 modes
# hawkname = "Toothless"
# # perch_distance = "9m"
# bilateral = True
# wingbeat_df, marker_column_names = load_bird_data(hawkname, bilateral=bilateral)

# nModes = 6
# save_path = f"../data/2024-06-12_All{hawkname}_DMD_results/{str(nModes)}Modes/"
# DMD_results_To = load_dmd_results(wingbeat_df,save_path)



# # # Load results from 6 modes
# hawkname = "Charmander"
# # perch_distance = "9m"
# bilateral = True
# wingbeat_df, marker_column_names = load_bird_data(hawkname, bilateral=bilateral)

# nModes = 6
# save_path = f"../data/2024-06-12_All{hawkname}_DMD_results/{str(nModes)}Modes/"
DMD_results_Ch = load_dmd_results(wingbeat_df,save_path)







# ## Histogram of each lamda (frequency)

# In[47]:





# In[36]:


# Create a figure with 12 subplots (1 row, 12 columns)
fig, axes = plt.subplots(1, 6, figsize=(10, 2), sharey=False, tight_layout=True)

# Plot each Lambda value in its own subplot
counter = 0
for ii in range(0,6):
    lambda_i_values = DMD_results_To['Lambda'].apply(lambda x: x[ii])
    axes[counter].hist(abs(lambda_i_values), bins=50, alpha=0.4, color='dodgerblue', range=(0, 200))

    lambda_i_values = DMD_results_Rh['Lambda'].apply(lambda x: x[ii])
    axes[counter].hist(abs(lambda_i_values), bins=50, alpha=0.4, color='red', range=(0, 200))

    lambda_i_values = DMD_results_Dr['Lambda'].apply(lambda x: x[ii])
    axes[counter].hist(abs(lambda_i_values), bins=50, alpha=0.4, color='seagreen', range=(0, 200))

    lambda_i_values = DMD_results_To9mRight['Lambda'].apply(lambda x: x[ii])
    axes[counter].hist(abs(lambda_i_values), bins=50, alpha=0.4, color='black', range=(0, 200))

    lambda_i_values = DMD_results_To9mLeft['Lambda'].apply(lambda x: x[ii])
    axes[counter].hist(abs(lambda_i_values), bins=50, alpha=0.4, color='yellow', range=(0, 200))
    

    # lambda_i_values = DMD_results_Ru['Lambda'].apply(lambda x: x[ii])
    # axes[counter].hist(abs(lambda_i_values), bins=50, alpha=0.4, color='purple')

    # lambda_i_values = DMD_results_Ch['Lambda'].apply(lambda x: x[ii])
    # axes[counter].hist(abs(lambda_i_values), bins=50, alpha=0.4, color='black')


    axes[counter].set_xlim(0, 200)

    # Make sure 25 and 50 are visible on the x-axis
    axes[counter].set_xticks([0, 15, 30, 60, 120, 200])
    axes[counter].grid(alpha=0.3)
    axes[counter].tick_params(axis='both', which='major', labelsize=6, rotation=90)

    
        

    axes[counter].tick_params(axis='both', which='major', labelsize=8)
    axes[counter].set_title(f"Modes {ii}", fontsize=9)

    counter += 1

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(wspace=0.3)
# fig.suptitle('Lambda for Single Wingbeat', fontsize=16)

plt.show()


# In[45]:


# Define the number of modes for each row
modes = [12, 10, 8, 6]

# Create a figure with 18 subplots (3 rows, 6 columns)
fig, axes = plt.subplots(4, 6, figsize=(10, 8), sharey=False, tight_layout=True)

# Plot each Lambda value in its own subplot for each mode set
for row, nModes in enumerate(modes):
    if nModes == 12:
        DMD_results = DMD_results_12Modes
    elif nModes == 8:
        DMD_results = DMD_results_8Modes
    elif nModes == 6:
        DMD_results = DMD_results_6Modes 
    
    counter = 0
    for ii in range(0, 12, 2):
        if counter >= nModes // 2:
            break
        lambda_i_values = DMD_results['Lambda'].apply(lambda x: x[ii])
        axes[row, counter].hist(abs(lambda_i_values), bins=100, alpha=0.2, color='dodgerblue')

        lambda_i_values = DMD_results['Lambda'].apply(lambda x: x[ii+1])
        axes[row, counter].hist(abs(lambda_i_values), bins=100, alpha=0.2, color='seagreen')

        axes[row, counter].tick_params(axis='both', which='major', labelsize=6, rotation=90)
        axes[row, counter].set_title(f"Modes {ii+1} & {ii+2}", fontsize=9)
        axes[row, counter].grid(True)

        # make grid more trasparent
        axes[row, counter].grid(alpha=0.3)

        # Make sure 25 and 50 are visible on the x-axis
        axes[row, counter].set_xticks([0, 15, 30, 60, 120, 200, 300])
        
        # axes[row, counter].set_xlim([0, 200])
        
        counter += 1
    
    # first column subplots ax xlim is set to 0, 100
    axes[row, 0].set_xlim([0, 100])
    axes[row, 1].set_xlim([0, 100])
    axes[row, 2].set_xlim([0, 200])
    axes[row, 3].set_xlim([0, 200])
    axes[row, 4].set_xlim([0, 300])
    axes[row, 5].set_xlim([0, 300])
    
    # Axis off for any remaining subplots
    for jj in range(counter, 6):
        axes[row, jj].axis('off')

    

# Set titles for the rows
axes[0, 0].set_ylabel('12 Modes', fontsize=12)
axes[1, 0].set_ylabel('10 Modes', fontsize=12)
axes[2, 0].set_ylabel('8 Modes', fontsize=12)
axes[3, 0].set_ylabel('6 Modes', fontsize=12)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.5)
# fig.suptitle('Lambda for Single Wingbeat', fontsize=16)

plt.show()


# In[94]:


# Define the number of modes for each row
modes = [12, 10, 8, 6]

# Create a figure with 3 subplots (1 row, 3 columns)
fig, axes = plt.subplots(2, 2, figsize=(5, 5), sharey=True, tight_layout=True)
axes = axes.flatten()
# Plot a histogram of all Lambda values for each mode set
for idx, nModes in enumerate(modes):
    if nModes == 12:
        DMD_results = DMD_results_12Modes
    elif nModes == 8:
        DMD_results = DMD_results_8Modes
    elif nModes == 6:
        DMD_results = DMD_results_6Modes
    
    # Combine all Lambda values into a single array
    lambda_values = np.concatenate(DMD_results['Lambda'].values)
    
    # Plot the histogram
    axes[idx].hist(abs(lambda_values), bins=50, alpha=0.4, color='dodgerblue')
    axes[idx].tick_params(axis='both', which='major', labelsize=8)
    axes[idx].set_title(f"{nModes} Modes", fontsize=12)
    axes[idx].set_xlabel('Lambda Values')

axes[0].set_ylabel('Frequency')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(wspace=0.3)
fig.suptitle('Histogram of Lambda Values for Different Modes', fontsize=16)

plt.show()


# In[83]:


nModes = 8
save_path = f"../data/2024-06-12_Toothless_DMD_results/{str(nModes)}Modes/"

seqID_list = wingbeat_df['seqID'].unique()

times, markers, Lambda, Modes, bn, Psi, keypoints = load_results_npz(save_path, seqID_list[0])

print(Lambda)


# In[18]:


Modes.shape


# In[24]:


hawk3d = Hawk3D("../data/mean_hawk_shape.csv")
currentMode = Modes[:,:,0]
currentMode = currentMode.reshape(1,8,3)
animation = animate(hawk3d,currentMode)


# In[20]:


hawk3d = Hawk3D("../data/mean_hawk_shape.csv")
animate_compare(hawk3d, keypoints_frames_list=[markers, keypoints])
plt.close()


# In[ ]:





# ## Save Toothless 5m Flights

# In[52]:


wingbeat_df, marker_column_names = load_bird_data("Toothless", "5m", bilateral)

wingbeat_df = handle_duplicates(wingbeat_df)

hawk3d = Hawk3D("../data/mean_hawk_shape.csv")
average_shape = hawk3d.markers.reshape(24,1).T
save_path = "../data/2024-05-15_DMD_results/04_05/"

process_and_store_sequences(wingbeat_df, marker_column_names, average_shape, save_path)



# ## Save Ruby 12m Flights

# In[53]:


wingbeat_df, marker_column_names = load_bird_data("Ruby", "12m", bilateral)

wingbeat_df = handle_duplicates(wingbeat_df)

hawk3d = Hawk3D("../data/mean_hawk_shape.csv")
average_shape = hawk3d.markers.reshape(24,1).T
save_path = "../data/2024-05-15_DMD_results/03_12/"

process_and_store_sequences(wingbeat_df, marker_column_names, average_shape, save_path)


# ## Saving Drogon 12m Flights

# In[25]:


import warnings
warnings.filterwarnings('ignore')

wingbeat_df, marker_column_names = load_bird_data("Drogon", "12m", bilateral)

wingbeat_df = handle_duplicates(wingbeat_df)

hawk3d = Hawk3D("../data/mean_hawk_shape.csv")
average_shape = hawk3d.markers.reshape(24,1).T
save_path = "../data/2024-05-15_DMD_results/01_12/"

process_and_store_sequences(wingbeat_df, marker_column_names, average_shape, save_path)


# ## Saving Rhaegal 12m Flights

# In[58]:


wingbeat_df, marker_column_names = load_bird_data("Rhaegal", "12m", bilateral)

wingbeat_df = handle_duplicates(wingbeat_df)

hawk3d = Hawk3D("../data/mean_hawk_shape.csv")
average_shape = hawk3d.markers.reshape(24,1).T
save_path = "../data/2024-05-15_DMD_results/02_12/"

process_and_store_sequences(wingbeat_df, marker_column_names, average_shape, save_path)


# In[63]:


save_path = "../data/2024-05-15_DMD_results/02_12/"
times, markers, Lambda, Modes, bn, Psi, keypoints = load_results_npz(save_path, "02_12_334")

print(markers.shape)
print(keypoints.shape)

hawk3d = Hawk3D("../data/mean_hawk_shape.csv")
animate_compare(hawk3d, keypoints_frames_list = [keypoints, markers])


# In[36]:






# In[ ]:






# ## Plan
# 
# Histogram of Lambdas
# 
