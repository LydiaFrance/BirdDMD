#!/usr/bin/env python
# coding: utf-8

# # Clean up Data suitable for DMD Analysis
# 
# > 2024-04-01 
# >
# > 2024-04-19 
# 
# Starting with a csv file of all data, create numpy files (`.npz`) which contains subsetted data and just the data needed for DMD analysis.
# 
# The csv files were created `2024-03-24` using matlab, though does not contain anything new should be easier to subset. 
# 
# ## Loading the Full Data
# 
# 

# In[15]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from BirdDMD import replace_rot_xyz_name, get_flight_modes, subset_by, get_column_names



# In[ ]:


unilateral_data = pd.read_csv("../data/raw/2024-03-24-FullUnilateralMarkers.csv")
bilateral_data = pd.read_csv("../data/raw/2024-03-24-FullBilateralMarkers.csv")


# ## Clean up Column Names

# In[16]:


unilateral_data = replace_rot_xyz_name(unilateral_data)
bilateral_data = replace_rot_xyz_name(bilateral_data)


# Add Turn information

# Add turn info to the frame_info_df
obstacle_df = pd.read_csv('../data/2024-06-01-ObstacleTurnsSeqList.csv')

unilateral_data = unilateral_data.merge(obstacle_df, how='left', on='seqID')
bilateral_data = bilateral_data.merge(obstacle_df, how='left', on='seqID')


bilateral_data.columns


# ## Define Flight Behaviours
# 
# Uses the distance to the perch and time from starting perch to estimate the flight behaviours. These include:
# - Initial: first wingbeat after takeoff
# - Flapping: All the wingbeats
# - Gliding: after flapping and before landing
# - Landing: last metre before the perch

# In[17]:


unilateral_data = get_flight_modes(unilateral_data)
bilateral_data = get_flight_modes(bilateral_data)

print(unilateral_data['behaviour'].unique())


# ## Save the data 
# 
# Saving csv files with just the flapping and initial behaviours for reference. 

# In[18]:


subset = bilateral_data[bilateral_data['behaviour'].str.contains("flapping")]
subset.to_csv("../data/processed/BilateralFlapping.csv")

subset = unilateral_data[unilateral_data['behaviour'].str.contains("flapping")]
subset.to_csv("../data/processed/UnilateralFlapping.csv")


subset = bilateral_data[bilateral_data['behaviour'].str.contains("initial")]
subset.to_csv("../data/processed/BilateralInitial.csv")

subset = unilateral_data[unilateral_data['behaviour'].str.contains("initial")]
subset.to_csv("../data/processed/UnilateralInitial.csv")


subset = bilateral_data[bilateral_data['behaviour'].str.contains("flapping_")]
subset.to_csv("../data/processed/BilateralTwoFlaps.csv")

subset = unilateral_data[unilateral_data['behaviour'].str.contains("flapping_")]
subset.to_csv("../data/processed/UnilateralTwoFlaps.csv")



# # Saving numpy data for DMD input
# 
# The DMD analysis will be done on the different behaviours separately. This will be saved as a numpy npz file. As numpy does not have column names, these are saved in a separate file.

# In[17]:


bilateral_data.columns


# In[ ]:





# ### Save Initial Flap Only

# In[25]:


# Save the column names
marker_column_names, info_column_names = get_column_names(bilateral_data)
np.savez("../data/samples/ColumnNames.npz", marker_column_names = marker_column_names, info_column_names = info_column_names)



birdList = ["Toothless", "Ruby", "Drogon", "Rhaegal"]
perchDistList = [5, 7, 9, 12]
turnList = ["Left", "Right", "Straight"]

for bird in birdList:
    for perchDist in perchDistList:
        if perchDist == 9:
            for turn in turnList:
                marker_data, info_data = subset_by(bilateral_data, 
                                                behaviour='initial', 
                                                bird=bird, 
                                                perchDist=perchDist, 
                                                turn=turn, 
                                                year = 2020)
                np.savez(f"../data/samples/Initial_{perchDist}m{turn}Turn{bird}_Bilateral.npz", marker_data = marker_data, info_data = info_data)
        else:
            marker_data, info_data = subset_by(bilateral_data, 
                                                behaviour='initial', 
                                                bird=bird, 
                                                perchDist=perchDist, 
                                                year = 2017)
            np.savez(f"../data/samples/Initial_{perchDist}m{bird}_Bilateral.npz", marker_data = marker_data, info_data = info_data)




# ### Save First Two Flaps Only

# In[25]:


# Save the column names
marker_column_names, info_column_names = get_column_names(bilateral_data)
np.savez("../data/samples/ColumnNames.npz", marker_column_names = marker_column_names, info_column_names = info_column_names)



birdList = ["Drogon", "Charmander", "Toothless", "Ruby"]
perchDistList = [9]
turnList = ["Straight"]

for bird in birdList:
    for perchDist in perchDistList:
        if perchDist == 9:
            for turn in turnList:
                marker_data, info_data = subset_by(bilateral_data, 
                                                behaviour='flapping_', 
                                                bird=bird, 
                                                perchDist=perchDist, 
                                                turn=turn, 
                                                year = 2020)
                np.savez(f"../data/samples/TwoFlaps_{perchDist}m{turn}Turn{bird}_Bilateral.npz", marker_data = marker_data, info_data = info_data)
        else:
            marker_data, info_data = subset_by(bilateral_data, 
                                                behaviour='flapping_', 
                                                bird=bird, 
                                                perchDist=perchDist, 
                                                year = 2017)
            np.savez(f"../data/samples/TwoFlaps_{perchDist}m{bird}_Bilateral.npz", marker_data = marker_data, info_data = info_data)



# ### Save Flapping Only

# In[7]:


# Save the column names
marker_column_names, info_column_names = get_column_names(bilateral_data)
np.savez("../data/samples/ColumnNames.npz", marker_column_names = marker_column_names, info_column_names = info_column_names)



birdList = ["Toothless"]
perchDistList = [5, 7, 9, 12]
turnList = ["Left", "Right", "Straight"]

for bird in birdList:
    for perchDist in perchDistList:
        if perchDist == 9:
            for turn in turnList:
                marker_data, info_data = subset_by(bilateral_data, 
                                                behaviour='flapping', 
                                                bird=bird, 
                                                perchDist=perchDist, 
                                                turn=turn, 
                                                year = 2020)
                np.savez(f"../data/samples/Flapping_{perchDist}m{turn}Turn{bird}_Bilateral.npz", marker_data = marker_data, info_data = info_data)
        else:
            marker_data, info_data = subset_by(bilateral_data, 
                                                behaviour='flapping', 
                                                bird=bird, 
                                                perchDist=perchDist, 
                                                year = 2017)
            np.savez(f"../data/samples/Flapping_{perchDist}m{bird}_Bilateral.npz", marker_data = marker_data, info_data = info_data)



# In[26]:


marker_data, info_data = subset_by(unilateral_data, 
                                   behaviour='intial', 
                                   bird="Toothless")
np.savez("../data/samples/Initial_Toothless_Unilateral.npz", marker_data = marker_data, info_data = info_data)

marker_data, info_data = subset_by(bilateral_data, 
                                   behaviour='initial', 
                                   bird="Toothless")
np.savez("../data/samples/Initial_Toothless_Bilateral.npz", marker_data = marker_data, info_data = info_data)


# In[20]:


marker_data, info_data = subset_by(unilateral_data, 
                                   behaviour='flapping_', 
                                   bird="Toothless")
np.savez("../data/samples/TwoFlaps_Toothless_Unilateral.npz", marker_data = marker_data, info_data = info_data)

marker_data, info_data = subset_by(bilateral_data, 
                                   behaviour='flapping_', 
                                   bird="Toothless")
np.savez("../data/samples/TwoFlaps_Toothless_Bilateral.npz", marker_data = marker_data, info_data = info_data)


# In[21]:


marker_data, info_data = subset_by(unilateral_data, 
                                   behaviour='flapping_', 
                                   bird="Ruby")
np.savez("../data/samples/TwoFlaps_Ruby_Unilateral.npz", marker_data = marker_data, info_data = info_data)

marker_data, info_data = subset_by(bilateral_data, 
                                   behaviour='flapping_', 
                                   bird="Ruby")
np.savez("../data/samples/TwoFlaps_Ruby_Bilateral.npz", marker_data = marker_data, info_data = info_data)



# In[27]:


marker_data, info_data = subset_by(unilateral_data, 
                                   behaviour='intial', 
                                   bird="Rhaegal")
np.savez("../data/samples/Initial_Rhaegal_Unilateral.npz", marker_data = marker_data, info_data = info_data)

marker_data, info_data = subset_by(bilateral_data, 
                                   behaviour='initial', 
                                   bird="Rhaegal")
np.savez("../data/samples/Initial_Rhaegal_Bilateral.npz", marker_data = marker_data, info_data = info_data)



# In[6]:


marker_data, info_data = subset_by(unilateral_data, 
                                   behaviour='intial', 
                                   bird="Ruby")
np.savez("../data/samples/Initial_Ruby_Unilateral.npz", marker_data = marker_data, info_data = info_data)

marker_data, info_data = subset_by(bilateral_data, 
                                   behaviour='initial', 
                                   bird="Ruby")
np.savez("../data/samples/Initial_Ruby_Bilateral.npz", marker_data = marker_data, info_data = info_data)



# In[10]:


marker_data, info_data = subset_by(unilateral_data, 
                                   behaviour='intial', 
                                   bird="Drogon")
np.savez("../data/samples/Initial_Drogon_Unilateral.npz", marker_data = marker_data, info_data = info_data)

marker_data, info_data = subset_by(bilateral_data, 
                                   behaviour='initial', 
                                   bird="Drogon")
np.savez("../data/samples/Initial_Drogon_Bilateral.npz", marker_data = marker_data, info_data = info_data)


