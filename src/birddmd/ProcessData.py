import numpy as np
import pandas as pd


# ------------ Process data ------------
def replace_rot_xyz_name(data):
    """
    Replace the _rot_xyz_1 etc with _x, _y, _z in the column names
    """

    for col in data.columns:
        if col.endswith('_rot_xyz_1'):
            data.rename(columns={col: col.replace('_rot_xyz_1', '_x')}, inplace=True)
        elif col.endswith('_rot_xyz_2'):
            data.rename(columns={col: col.replace('_rot_xyz_2', '_y')}, inplace=True)
        elif col.endswith('_rot_xyz_3'):
            data.rename(columns={col: col.replace('_rot_xyz_3', '_z')}, inplace=True)
    return data

def get_flight_modes(data):

    # Define the bird IDs
    birdID_dict = define_birdIDs()

    # Define the perch distances
    perchDist_dict = define_perchDists()

    # Define the flight behaviour boundaries
    flight_modes = define_behaviour_boundaries()

    # Initialise the behaviour column
    data['behaviour'] = ""

    # Loop through each bird and perch distance
    for birdID in birdID_dict.keys():
        for perchDist in perchDist_dict.keys():
            
            # Find the index for current bird and current perch distance
            index = data['BirdID'] == birdID_dict[birdID]
            index = index & (data['PerchDistance'] == perchDist_dict[perchDist])

            # Skip if no data for current bird and perch distance
            if sum(index) == 0:
                continue

            # Define the flight behaviours based on the time from 
            # start perch and distance from landing perch

            initial = index & (data['time'] < flight_modes["initial"][birdID][perchDist])
            data.loc[initial, 'behaviour'] = "flapping_initial"

            flapping = index & (data['time'] > 0) & (data['time'] < flight_modes["flapping"][birdID][perchDist]) & (data['behaviour']== "") 
            data.loc[flapping, 'behaviour'] = "flapping"

            gliding = index & (data['time'] > flight_modes["flapping"][birdID][perchDist]) & (-data['HorzDistance'] < flight_modes["landing"][birdID][perchDist])
            data.loc[gliding, 'behaviour'] = "gliding"

            landing = index & (-data['HorzDistance'] > flight_modes["landing"][birdID][perchDist])
            data.loc[landing, 'behaviour'] = "landing"

    return data

def subset_by(data, bird=None, perchDist=None, behaviour=None):

    # If no birdID specified, return all the index
    if bird is None:
        index = data.index
    else:
        # Check if birdID is a string
        if isinstance(bird, str):
            birdID_dict = define_birdIDs()
            birdID = birdID_dict[bird.lower()]

        index = data['BirdID'] == birdID

    if perchDist is not None:
        if isinstance(perchDist, str):
            perchDist_dict = define_perchDists()
            perchDist = perchDist_dict[perchDist]
        index = index & (data['PerchDistance'] == perchDist)

    if behaviour is not None:
        index = index & (data['behaviour'].str.contains(behaviour))

    marker_column_names, info_column_names = get_column_names(data)

    marker_data = data.loc[index, marker_column_names].to_numpy()
    info_data   = data.loc[index, info_column_names].to_numpy()

    return marker_data, info_data


def get_column_names(data):

    marker_column_names = [col for col in data.columns if '_' in col and 'body' not in col]

    info_column_names = [col for col in data.columns if col not in marker_column_names]

    return marker_column_names, info_column_names

# ............ Helper functions ............

def define_birdIDs():

    birdID_dict = {
                "drogon"  : 1, 
                "rhaegal" : 2, 
                   "ruby" : 3, 
              "toothless" : 4, 
             "charmander" : 5}

    return birdID_dict

def define_perchDists():

    perchDist_dict = {
                  "5m" : 5, 
                  "7m" : 7, 
                  "9m" : 9, 
                  "12m": 12}
    
    return perchDist_dict

def define_behaviour_boundaries():

    # Using Time from start perch
    initial_lims = {"drogon":     {"5m": 0.21, "7m": 0.20, "9m": 0.21, "12m": 0.22},
                    "rhaegal":    {"5m": 0.21, "7m": 0.17, "9m": 0.22, "12m": 0.21},
                    "ruby":       {"5m": 0.23, "7m": 0.22, "9m": 0.25, "12m": 0.26},
                    "toothless":  {"5m": 0.21, "7m": 0.22, "9m": 0.22, "12m": 0.20},
                    "charmander": {"9m": 0.3}}

    flap_dict =     {"drogon":    {"5m": 0.55, "7m": 0.65, "9m": 0.75, "12m": 0.80},
                    "rhaegal":   {"5m": 0.55, "7m": 0.65, "9m": 0.80, "12m": 0.85},
                    "ruby":      {"5m": 0.60, "7m": 0.75, "9m": 0.87, "12m": 1.10},
                    "toothless": {"5m": 0.55, "7m": 0.70, "9m": 0.70, "12m": 0.85},
                    "charmander":{"9m": 0.9}}

    # Distance from landing perch
    landing_lims =  {"drogon":    {"5m": -1.3, "7m": -1.1, "9m": -1.1, "12m": -1.1},
                    "rhaegal":   {"5m": -1.3, "7m": -1.1, "9m": -1.1, "12m": -1.1},
                    "ruby":      {"5m": -1.0, "7m": -1.1, "9m": -1.1, "12m": -1.1},
                    "toothless": {"5m": -1.3, "7m": -1.1, "9m": -1.1, "12m": -1.1},
                    "charmander":{"9m": -1.1}}

    # Store in a dictionary
    flight_modes = {"initial": initial_lims,
                    "flapping": flap_dict,
                    "landing": landing_lims}
    
    return flight_modes
