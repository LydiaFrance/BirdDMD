import os
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from BirdDMD import (
    # Data handling
    load_bird_data,
    load_sequence_data,
    remove_time_duplicates,
    normalise_data,
    
    # Core DMD
    run_dmd,
    run_sequence_dmd,
    run_marker_dmd,
    run_timeseries_dmd,
    reorder_dmd_results,
    
    # Reconstruction
    reconstruct_dmd,
    reconstruct_specific_modes,
    run_forecast,
    run_forecast_with_modified_modes,
    modify_mode_frequencies,
    
    # PCA analysis
    make_unilateral_keypoints,
    project_into_pca_space,
    project_into_coordinate_space,
    create_scores_info_df,
    
    # Visualization
    plot_amplitude_ranking,
    plot_2d_markers,
    plot_markers_overDist,
    plot_single_sequence,
    plot_score_multi_PCs
)

# Test data paths
DATA_DIR = Path(__file__).parent.parent / "data"
SAMPLES_DIR = DATA_DIR / "samples"

@pytest.fixture
def sample_data():
    """Load sample data for testing."""
    # Load data exactly as in AllFlapping.py
    column_names = np.load(SAMPLES_DIR / "ColumnNames.npz")
    file = np.load(SAMPLES_DIR / "Flapping_9mStraightTurnToothless_Bilateral.npz", allow_pickle=True)
    
    marker_data = file["marker_data"]
    info_data = file["info_data"]
    
    marker_column_names = column_names["marker_column_names"]
    info_column_names = column_names["info_column_names"]
    
    # Create dataframes
    marker_df = pd.DataFrame(marker_data, columns=marker_column_names)
    info_df = pd.DataFrame(info_data, columns=info_column_names)
    
    # Concatenate
    wingbeat_df = pd.concat([info_df, marker_df], axis=1)
    
    return {
        'wingbeat_df': wingbeat_df,
        'marker_column_names': marker_column_names,
        'info_column_names': info_column_names
    }

@pytest.fixture(scope="session")
def dmd_results():
    """Run DMD on a specific sequence and return results."""
    times, markers, Lambda, Modes, bn, Psi, phase_shifts, dmd_results, keypoints = run_sequence_dmd(
        bird_name="Toothless",
        perch_dist="9m",
        turn="Straight",
        behaviour="Flapping",
        seqID="04_09_038_1",
        eig_constraints={"conjugate_pairs"},
        n_modes=10,
        d=2,
        verbose=False
    )
    
    return {
        'times': times,
        'markers': markers,
        'Lambda': Lambda,
        'Modes': Modes,
        'amplitudes': bn,
        'Psi': Psi,
        'phase_shifts': phase_shifts,
        'dmd_results': dmd_results,
        'keypoints': keypoints
    }

def test_data_loading(sample_data):
    """Test that data is loaded with correct shapes and types."""
    wingbeat_df = sample_data['wingbeat_df']
    marker_column_names = sample_data['marker_column_names']
    
    # Check dataframe shape
    assert isinstance(wingbeat_df, pd.DataFrame)
    assert len(wingbeat_df) > 0
    
    # Check column names
    assert len(marker_column_names) > 0
    assert all(col in wingbeat_df.columns for col in marker_column_names)
    
    # Check sequence IDs
    n_sequences = wingbeat_df["seqID"].nunique()
    assert n_sequences > 0

def test_dmd_results_shape(dmd_results):
    """Test that DMD results have the expected shapes."""
    assert len(dmd_results['Lambda']) == 10  # n_modes
    assert dmd_results['Modes'].shape[2] == 10  # n_modes
    assert len(dmd_results['amplitudes']) == 10  # n_modes
    assert dmd_results['Psi'].shape[1] == 10  # n_modes
    assert dmd_results['keypoints'].shape[0] == len(dmd_results['times'])
    assert dmd_results['keypoints'].shape[1] == 8  # 8 markers
    assert dmd_results['keypoints'].shape[2] == 3  # 3D coordinates

def test_marker_dmd(sample_data):
    """Test running DMD directly on marker data."""
    # Extract marker data for a specific sequence
    seqID = sample_data['wingbeat_df']['seqID'].iloc[0]
    markers, times = load_sequence_data(
        sample_data['wingbeat_df'],
        seqID,
        sample_data['marker_column_names']
    )
    
    # Reshape markers to (n_frames, n_markers, 3)
    markers = markers.reshape(-1, 8, 3)
    
    # Run DMD on marker data
    Lambda, Modes, bn, Psi, phase_shifts, dmd_results, reconstruction = run_marker_dmd(
        markers=markers,
        times=times,
        n_modes=10,
        d=2,
        eig_constraints={"conjugate_pairs"},
        verbose=False
    )
    
    # Check results
    assert len(Lambda) == 10
    assert Modes.shape == (8, 3, 10)
    assert len(bn) == 10
    assert Psi.shape[1] == 10
    assert reconstruction.shape == markers.shape

def test_timeseries_dmd(sample_data):
    """Test running DMD on generic time series data using real marker data."""
    # Extract marker data for a specific sequence
    seqID = sample_data['wingbeat_df']['seqID'].iloc[0]
    markers, times = load_sequence_data(
        sample_data['wingbeat_df'],
        seqID,
        sample_data['marker_column_names']
    )
    
    # Use more variables - take first 4 coordinates (x,y of first two markers)
    # This gives us more information for the DMD algorithm
    data = markers[:, [0, 1, 2, 3, 4]]  # First 4 coordinates
    
    # Transpose data to (n_variables, n_timesteps) as expected by DMD
    data = data.T
    
    print("\nData preparation debug:")
    print(f"Original data shape: {markers[:, [0, 1, 2, 3, 4]].shape}")
    print(f"Transposed data shape: {data.shape}")
    print(f"Times shape: {times.shape}")
    
    # Run DMD on time series data
    # With d=2 and 4 variables, we can compute more modes
    Lambda, Modes, bn, Psi, phase_shifts, dmd_results, reconstruction = run_timeseries_dmd(
        data=data,
        times=times,
        n_modes=2,  # Request 2 modes
        d=2,
        eig_constraints={"conjugate_pairs"},
        verbose=False
    )
    
    # Get actual number of modes computed
    n_modes = len(Lambda)
    print(f"\nDMD results debug:")
    print(f"Number of modes computed: {n_modes}")
    print(f"Eigenvalues: {Lambda}")
    
    # Check results
    assert n_modes > 0, "Should compute at least one mode"
    assert n_modes == 2, "Should compute exactly 2 modes as requested"
    
    # Account for Hankel matrix preprocessing effect
    # With d=2, the number of variables is doubled
    expected_vars = data.shape[0] *2 # Original variables * delay embedding
    assert Modes.shape == (expected_vars, 2), f"Expected Modes shape ({expected_vars}, 2), got {Modes.shape}"
    assert Psi.shape == (expected_vars, 2), f"Expected Psi shape ({expected_vars}, 2), got {Psi.shape}"
    assert len(bn) == 2
    assert reconstruction.shape == data.T.shape  # Reconstruction should match input shape
    
    # Check that reconstruction is close to original data
    # DMD should capture the main dynamics
    reconstruction_error = np.mean(np.abs(reconstruction - data.T))
    print(f"Reconstruction error: {reconstruction_error}")
    assert reconstruction_error < 1.0  # Error should be reasonably small

# ------------------------------------------------------------------------------------------------
# ONLY UNCOMMENT WHEN YOU WANT TO SAVE THE RESULTS FOR INSPECTION
# ------------------------------------------------------------------------------------------------


# def test_save_dmd_results_for_inspection(dmd_results):
#     """Test that saves DMD results for manual inspection of values."""
#     # Create a directory for test data if it doesn't exist
#     test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
#     os.makedirs(test_data_dir, exist_ok=True)
    
#     # Save to numpy file
#     baseline_file = os.path.join(test_data_dir, 'dmd_baseline_results.npz')
#     np.savez_compressed(baseline_file, **dmd_results)
    
#     print(f"\nDMD Results saved to: {baseline_file}")
#     print("\nKey DMD Results Summary:")
#     print(f"Number of modes: {len(dmd_results['Lambda'])}")
#     print("\nFrequencies (Lambda):")
#     print(dmd_results['Lambda'])
#     print("\nAmplitudes (bn):")
#     print(dmd_results['amplitudes'])
#     print("\nMode shapes (first mode):")
#     print(dmd_results['Modes'][:,:,0])  # First mode shape

# ------------------------------------------------------------------------------------------------

def test_compare_dmd_results(dmd_results):
    """Test that compares new DMD results against saved baseline."""
    # Load baseline results
    test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    baseline_file = os.path.join(test_data_dir, 'dmd_baseline_results.npz')
    
    if not os.path.exists(baseline_file):
        pytest.skip("Baseline results not found. Run test_save_dmd_results_for_inspection first.")
    
    baseline = np.load(baseline_file)
    
    # Compare results with baseline
    # Use a small tolerance for floating point comparisons
    rtol = 1e-5
    atol = 1e-8
    
    # Print summary of phase shift differences
    print("\nPhase Shift Comparison Summary:")
    print(f"Current shape: {dmd_results['phase_shifts'].shape}")
    print(f"Baseline shape: {baseline['phase_shifts'].shape}")
    
    # Calculate differences
    abs_diff = np.abs(dmd_results['phase_shifts'] - baseline['phase_shifts'])
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    
    print(f"\nMax absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")
    
    # Find indices of largest differences
    if max_diff > atol:
        print("\nLargest differences found at:")
        threshold = max_diff * 0.9  # Show differences within 90% of max
        large_diffs = np.where(abs_diff > threshold)
        for i, j in zip(*large_diffs):
            print(f"Position [{i},{j}]:")
            print(f"  Current: {dmd_results['phase_shifts'][i,j]:.6f}")
            print(f"  Baseline: {baseline['phase_shifts'][i,j]:.6f}")
            print(f"  Difference: {abs_diff[i,j]:.6f}")
    
    # Compare each component
    assert np.allclose(dmd_results['times'], baseline['times'], rtol=rtol, atol=atol), "Times don't match baseline"
    assert np.allclose(dmd_results['markers'], baseline['markers'], rtol=rtol, atol=atol), "Markers don't match baseline"
    assert np.allclose(dmd_results['Lambda'], baseline['Lambda'], rtol=rtol, atol=atol), "Lambda doesn't match baseline"
    assert np.allclose(dmd_results['Modes'], baseline['Modes'], rtol=rtol, atol=atol), "Modes don't match baseline"
    assert np.allclose(dmd_results['amplitudes'], baseline['amplitudes'], rtol=rtol, atol=atol), "Amplitudes don't match baseline"
    assert np.allclose(dmd_results['Psi'], baseline['Psi'], rtol=rtol, atol=atol), "Psi doesn't match baseline"
    assert np.allclose(dmd_results['phase_shifts'], baseline['phase_shifts'], rtol=rtol, atol=atol), "Phase shifts don't match baseline"
    assert np.allclose(dmd_results['keypoints'], baseline['keypoints'], rtol=rtol, atol=atol), "Keypoints don't match baseline"

def test_remove_time_duplicates(sample_data):
    """Test removal of duplicate time frames."""
    df = sample_data['wingbeat_df']
    original_len = len(df)
    
    df_cleaned = remove_time_duplicates(df)
    
    # Check that we have a dataframe
    assert isinstance(df_cleaned, pd.DataFrame)
    
    # Check that we have no duplicates in frameID
    assert not df_cleaned.duplicated(subset=['frameID']).any()
    
    # Check that we haven't lost all data
    assert len(df_cleaned) > 0
    assert len(df_cleaned) <= original_len

def test_reorder_dmd_results(dmd_results):
    """Test DMD results reordering with different methods."""
    # Test amplitude ordering
    Lambda_amp, Modes_amp, bn_amp, Psi_amp, phase_amp = reorder_dmd_results(
        dmd_results['dmd_results'], 
        num_markers=8,  # Changed from n_markers to num_markers
        n_modes=10, 
        order_by='amplitude'
    )
    assert len(Lambda_amp) == 10
    assert Modes_amp.shape == (8, 3, 10)
    
    # Test frequency ordering
    Lambda_freq, Modes_freq, bn_freq, Psi_freq, phase_freq = reorder_dmd_results(
        dmd_results['dmd_results'], 
        num_markers=8,  # Changed from n_markers to num_markers
        n_modes=10, 
        order_by='frequency'
    )
    assert len(Lambda_freq) == 10
    assert Modes_freq.shape == (8, 3, 10)

def test_pca_projection(dmd_results):
    """Test PCA projection and reconstruction using real data."""
    # Use the keypoints from DMD results
    keypoints = dmd_results['keypoints']
    n_samples = keypoints.shape[0]
    n_features = keypoints.shape[1] * keypoints.shape[2]  # 8 markers * 3 coordinates
    
    # Flatten the keypoints
    data = keypoints.reshape(n_samples, n_features)
    mu = np.mean(data, axis=0)
    centered_data = data - mu
    
    # Create synthetic PCA components (since we don't have real PCA components)
    n_components = 3
    principal_components = np.random.rand(n_components, n_features)
    principal_components = principal_components / np.linalg.norm(principal_components, axis=1)[:, np.newaxis]
    
    # Test projection
    scores = project_into_pca_space(centered_data, mu, principal_components)
    assert scores.shape == (n_samples, n_components)
    
    # Test reconstruction
    keypoints_reconstructed = project_into_coordinate_space(scores, mu, principal_components)
    assert keypoints_reconstructed.shape == (n_samples, 8, 3)

def test_forecast_modification(dmd_results):
    """Test DMD forecast with modified modes."""
    # Create longer time vector for forecast (3x longer as in example)
    fake_time = np.linspace(dmd_results['times'][0], dmd_results['times'][-1]*3, len(dmd_results['times'])*3)
    
    # Test forecast with modified modes
    modified_keypoints = run_forecast_with_modified_modes(
        dmd_results['dmd_results'],
        fake_time,
        average_shape=dmd_results['keypoints'][0],  # Use first frame as average shape
        num_markers=8,
        mode_indices_to_zero=[0,1],  # Set first mode to zero frequency
        selected_mode_indices=[0,1,2,3,6,7]  # Only use these modes
    )
    
    assert isinstance(modified_keypoints, np.ndarray)
    assert modified_keypoints.shape[0] == len(fake_time)
    assert modified_keypoints.shape[1] == 8  # 8 markers
    assert modified_keypoints.shape[2] == 3  # 3D coordinates

def test_unilateral_processing(dmd_results):
    """Test unilateral keypoint processing."""
    # Test unilateral processing
    dmd_reconstruction_flat, left_right_bool = make_unilateral_keypoints(dmd_results['keypoints'])
    
    assert isinstance(dmd_reconstruction_flat, np.ndarray)
    assert isinstance(left_right_bool, np.ndarray)
    assert len(left_right_bool) == dmd_reconstruction_flat.shape[0]
    assert dmd_reconstruction_flat.shape[0] == 2 * len(dmd_results['times'])  # Should be doubled for left/right
    assert dmd_reconstruction_flat.shape[1] == 12  # 4 markers * 3 coordinates

def test_amplitude_ranking(dmd_results):
    """Test amplitude ranking plot generation."""
    fig, ax, sorted_amps = plot_amplitude_ranking(
        dmd_results['markers'],
        dmd_results['times'],
        max_modes=20,
        d=2,
        eig_constraints={"conjugate_pairs"}
    )
    
    # Check outputs
    assert fig is not None
    assert ax is not None
    assert isinstance(sorted_amps, np.ndarray)
    assert len(sorted_amps) > 0

def test_dmd_reconstruction(dmd_results):
    """Test DMD reconstruction functionality."""
    # Test specific mode reconstruction
    mode_0_1_keypoints = reconstruct_specific_modes(dmd_results['times'], dmd_results['dmd_results'], [0,1])
    mode_2_3_keypoints = reconstruct_specific_modes(dmd_results['times'], dmd_results['dmd_results'], [2,3])
    mode_4_5_keypoints = reconstruct_specific_modes(dmd_results['times'], dmd_results['dmd_results'], [4,5])
    mode_6_7_keypoints = reconstruct_specific_modes(dmd_results['times'], dmd_results['dmd_results'], [6,7])
    mode_8_9_keypoints = reconstruct_specific_modes(dmd_results['times'], dmd_results['dmd_results'], [8,9])
    
    # Check all reconstructions
    for mode_keypoints in [mode_0_1_keypoints, mode_2_3_keypoints, mode_4_5_keypoints, 
                          mode_6_7_keypoints, mode_8_9_keypoints]:
        assert isinstance(mode_keypoints, np.ndarray)
        assert mode_keypoints.shape[0] == len(dmd_results['times'])
        assert mode_keypoints.shape[1] == 8  # 8 markers
        assert mode_keypoints.shape[2] == 3  # 3D coordinates 