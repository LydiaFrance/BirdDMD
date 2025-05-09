import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from BirdDMD import (
    run_single_wingbeat_dmd,
    load_bird_data,
    remove_time_duplicates,
    reorder_dmd_results
)

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "data", "samples")
MEAN_SHAPE_PATH = os.path.join(PROJECT_ROOT, "data", "mean_hawk_shape.csv")

# Configure plotting
plt.rcParams['font.family'] = 'Andale Mono'

# Parameters for DMD analysis
DMD_CONFIG = {
    'n_modes': 8,
    'd': 2,
    'eig_constraints': {"conjugate_pairs"},
    'min_seq_length': 40,  # Will default to n_modes + 1
    'interpolate': False,
    'verbose': False  # Reduce output noise during batch processing
}

# Flight parameters
FLIGHT_CONFIG = {
    'behaviour': 'Initial',
    'bilateral': 'Bilateral'
}

# List of known bird names
BIRD_NAMES = ['Drogon', 'Charmander', 'Ruby', 'Toothless', 'Rhaegal']

def process_all_sequences():
    """
    Process DMD for all birds and sequences, returning DataFrames for both amplitude and frequency ordering.
    """
    amp_results = []
    freq_results = []
    
    for bird_name in tqdm(BIRD_NAMES, desc="Processing birds"):
        try:
            df, _ = load_bird_data(
                bird_name=bird_name,
                behaviour=FLIGHT_CONFIG['behaviour'],
                perch_distance=None,
                bilateral=FLIGHT_CONFIG['bilateral']
            )
            
            df = remove_time_duplicates(df)
            
            for seqID in tqdm(df['seqID'].unique(), desc=f"Sequences for {bird_name}", leave=False):
                try:
                    # Run DMD analysis with amplitude ordering
                    times, markers, Lambda_amp, Modes_amp, bn_amp, Psi_amp, phase_shifts_amp, dmd_results, keypoints = run_single_wingbeat_dmd(
                        bird_name=bird_name,
                        perch_dist=None,
                        behaviour=FLIGHT_CONFIG['behaviour'],
                        bilateral=FLIGHT_CONFIG['bilateral'],
                        seqID=seqID,
                        mean_shape_path=MEAN_SHAPE_PATH,
                        order_by='amplitude',
                        **DMD_CONFIG
                    )
                    
                    if times is not None:
                        # Store amplitude-ordered results
                        amp_sequence = {
                            'bird_name': bird_name,
                            'seqID': seqID,
                            'n_frames': len(times),
                            'duration': times[-1] - times[0]
                        }
                        
                        # Store frequency-ordered results (reorder the same DMD results)
                        freq_sequence = amp_sequence.copy()
                        
                        # Get frequency-ordered results
                        Lambda_freq, _, bn_freq, _, _ = reorder_dmd_results(
                            dmd_results, 
                            num_markers=8, 
                            nModes=DMD_CONFIG['n_modes'],
                            order_by='frequency'
                        )
                        
                        # Add mode frequencies and amplitudes for amplitude ordering
                        for i, (freq, amp) in enumerate(zip(Lambda_amp, bn_amp)):
                            amp_sequence[f'mode_{i}_freq'] = freq
                            amp_sequence[f'mode_{i}_amplitude'] = np.abs(amp)
                        
                        # Add mode frequencies and amplitudes for frequency ordering
                        for i, (freq, amp) in enumerate(zip(Lambda_freq, bn_freq)):
                            freq_sequence[f'mode_{i}_freq'] = freq
                            freq_sequence[f'mode_{i}_amplitude'] = np.abs(amp)
                        
                        amp_results.append(amp_sequence)
                        freq_results.append(freq_sequence)
                
                except Exception as e:
                    print(f"Error processing {bird_name} sequence {seqID}: {str(e)}")
                    continue
        except Exception as e:
            print(f"Error loading data for {bird_name}: {str(e)}")
            continue
    
    # Convert to DataFrames
    amp_df = pd.DataFrame(amp_results)
    freq_df = pd.DataFrame(freq_results)
    return amp_df, freq_df

def plot_mode_frequencies(results_df):
    """Plot histograms of mode frequencies."""
    plt.figure(figsize=(15, 10))
    n_modes = DMD_CONFIG['n_modes']
    for i in range(n_modes):
        plt.subplot(2, 4, i+1)
        plt.hist(results_df[f'mode_{i}_freq'], bins=30)
        plt.title(f'Mode {i} Frequencies')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def plot_mode_amplitudes(results_df):
    """Plot histograms of mode amplitudes."""
    plt.figure(figsize=(15, 10))
    n_modes = DMD_CONFIG['n_modes']
    for i in range(n_modes):
        plt.subplot(2, 4, i+1)
        plt.hist(results_df[f'mode_{i}_amplitude'], bins=30)
        plt.title(f'Mode {i} Amplitudes')
        plt.xlabel('Amplitude')
        plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def print_summary_statistics(results_df):
    """Print summary statistics for the results."""
    # Summary by bird
    summary_by_bird = results_df.groupby('bird_name').agg({
        'n_frames': ['mean', 'std', 'count'],
        'duration': ['mean', 'std']
    }).round(3)
    print("\nSummary by bird:")
    print(summary_by_bird)

    # Mode frequency statistics
    freq_cols = [col for col in results_df.columns if 'freq' in col]
    freq_stats = results_df[freq_cols].agg(['mean', 'std', 'min', 'max']).round(3)
    print("\nMode frequency statistics:")
    print(freq_stats)

if __name__ == "__main__":
    print(f"Processing data for birds: {', '.join(BIRD_NAMES)}")
    print(f"Using mean shape file: {MEAN_SHAPE_PATH}")
    print(f"Using samples directory: {SAMPLES_DIR}")

    # Run the batch processing
    amp_results_df, freq_results_df = process_all_sequences()

    # Save results
    amp_output_file = os.path.join(PROJECT_ROOT, 'data', 'dmd_batch_results_amp_ordered.csv')
    freq_output_file = os.path.join(PROJECT_ROOT, 'data', 'dmd_batch_results_freq_ordered.csv')
    
    amp_results_df.to_csv(amp_output_file, index=False)
    freq_results_df.to_csv(freq_output_file, index=False)
    
    print(f"Processed {len(amp_results_df)} sequences successfully")
    print(f"Results saved to:")
    print(f"  Amplitude ordered: {amp_output_file}")
    print(f"  Frequency ordered: {freq_output_file}")

    # Generate plots and statistics for both orderings
    print("\nAmplitude-ordered results:")
    plot_mode_frequencies(amp_results_df)
    plot_mode_amplitudes(amp_results_df)
    print_summary_statistics(amp_results_df)

    print("\nFrequency-ordered results:")
    plot_mode_frequencies(freq_results_df)
    plot_mode_amplitudes(freq_results_df)
    print_summary_statistics(freq_results_df) 