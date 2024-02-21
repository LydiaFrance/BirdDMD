# Experiment: Initial Wingbeat by One Bird

> 2024-02-20 Using 01_InitialWingbeatToothless12m.ipynb

## Input Data

Toothless, an inexperienced male Harris' hawk, from 177 flights of 12m between two perches. Taking the initial wingbeat from takeoff, which is selected as between 12m-10.6m from the end perch. 

Data contains `[x y z]` coordinates for four markers, relative to the central point on the bird. Markers are from the right side of the bird only, with three markers on the right wing and one on the right side of the tail. 

There were no obstacles in the flights, and no weights added to the bird. 

The inital wingbeat being compared across the different flights is the first wingbeat after the taking-off jump, and its function is maximum acceleration. 

<p align="center">
    <img src="../../docs/imgs/Markers_by_space.png" width="500" height="500" >
</p>

*Figure 1: Columns show x, y, z dimensions respectively. Rows show the different markers. From top: wingtip; primary feather; secondary feather; outer tail feather tip.Markers shown over spatial distance to the landing perch.*


## Data Processing

Each data sequence lasts roughly 0.2 seconds, with around 30-40 frames per sequence except for major dropout.

Duplicate frames, where two instances were found in the same time stamp, were removed. This was around 8% of the total frames. 

An adjusted time was calculated such that each sequence was concatenated -- with each sequence following the last in time.


<img src="../imgs/2024-02-20_WingtipToothless9m.png">

*Figure 2: Each flight sequence shown with an adjusted time variable, such that each sequence follows the previous and times do not overlap. Only right wingtip in z shown for demonstration purposes.*

## DMD Analysis

The input marker data was `12 x 4879` in size.   

The following settings were used:

```python
delay_optdmd = hankel_preprocessing(BOPDMD(svd_rank=7, num_trials=0, eig_constraints={"imag", "conjugate_pairs"}), d=2)
delay_optdmd.fit(markers, t=seqTime[1:])
plot_summary(delay_optdmd, index_modes=[1,3,5], order='F') 
```

Which produced the following plot. 

<p align="center">
    <img src="../imgs/2024-02-20_DMDFigure.png" width=500>
</p>

*Figure 3: Output of the DMD Analysis. Modes 2,4,6 are shown.*

## Dynamics Plot

Using the following settings:

```python
for dynamic in delay_optdmd.dynamics:
    plt.plot(seqTime[1:], dynamic.real)
```


Just in case the sign is arbitrary as in PCA here is the same figure with the y axis unmirrored and mirrored. 

<p align="center">
    <img src="../imgs/2024-02-20_DynamicPlot.png" width=400> <img src="../imgs/2024-02-20_NegativeDynamicPlot.png" width=400>
</p>

*Figure 4: Output of the DMD Dynamics. X axis trimmed to a similar timeframe as a single wingbeat. The right panel shows the same but the y axis mirrored.*




Looking at a plot of the raw data over time with the same vertical grid lines:

<p align="center">
    <img src="../imgs/Markers_by_time.png" width=600>
</p>




## Questions

- How do the peaks in the modes relate to the waveforms we see in the raw data?
- Are the positive/negative units arbitrary?
- How does the x axis work for the dynamics plot?
- General questions about looking at different settings for DMD, why is it so different between them?

