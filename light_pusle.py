import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import helpers as hp


def detect_light_pulses(worm_df, pulse_column='Light_Pulse', pulse_length_min=8, pulse_length_max=15):
    """
    Detect light pulses in the dataset and return the start and end frames of valid pulses.

    Args:
        worm_df (pd.DataFrame): DataFrame containing worm data.
        pulse_column (str): Column indicating light pulses (0 for no pulse, 1 for pulse).
        pulse_length_min (int): Minimum acceptable length of a light pulse.
        pulse_length_max (int): Maximum acceptable length of a light pulse.

    Returns:
        list: List of tuples (start_frame, end_frame) for valid light pulses.
    """
    pulse_sequences = []
    pulse_active = False
    start_frame = None

    for i, pulse in enumerate(worm_df[pulse_column]):
        if pulse == 1 and not pulse_active:  # Start of a new pulse
            pulse_active = True
            start_frame = worm_df['Frame'].iloc[i]
        elif pulse == 0 and pulse_active:  # End of the current pulse
            pulse_active = False
            end_frame = worm_df['Frame'].iloc[i - 1]
            pulse_length = end_frame - start_frame + 1
            if pulse_length_min <= pulse_length <= pulse_length_max:
                pulse_sequences.append((start_frame, end_frame))

    return pulse_sequences



def compare_behavior_before_during(worm_df, light_pulses, window=15):
    """
    Compare worm behavior before and during light pulses.

    Args:
        worm_df (pd.DataFrame): DataFrame containing worm data.
        light_pulses (list): List of tuples (start_frame, end_frame) for light pulses.
        window (int): Number of frames before the pulse to consider for comparison.

    Returns:
        pd.DataFrame: Summary of behavior comparisons.
    """
    comparisons = []

    for start_frame, end_frame in light_pulses:
        during_pulse = worm_df[(worm_df['Frame'] >= start_frame) & (worm_df['Frame'] <= end_frame)]
        
        before_pulse = worm_df[
            (worm_df['Frame'] >= start_frame - window) & (worm_df['Frame'] < start_frame)
        ]

        # Ensure enough data is available for both before and during windows
        if not before_pulse.empty and not during_pulse.empty:
            comparison = {
                'Pulse Start': start_frame,
                'Pulse End': end_frame,
                'Speed Before': before_pulse['Speed'].mean(),
                'Speed During': during_pulse['Speed'].mean(),
                'Changed Pixels Before': before_pulse['Changed Pixels'].mean(),
                'Changed Pixels During': during_pulse['Changed Pixels'].mean(),
                'Distance Before': np.sqrt(np.diff(before_pulse['X']).mean()**2 + np.diff(before_pulse['Y']).mean()**2),
                'Distance During': np.sqrt(np.diff(during_pulse['X']).mean()**2 + np.diff(during_pulse['Y']).mean()**2)
            }
            comparisons.append(comparison)

    return pd.DataFrame(comparisons)



def plot_behavior_changes(behavior_comparisons, feature, worm_type):
    """
    Plot boxplots comparing a feature (e.g., Speed, Changed Pixels) before and during light pulses.

    Args:
        behavior_comparisons (pd.DataFrame): DataFrame with behavior comparison metrics.
        feature (str): The feature to plot ('Speed' or 'Changed Pixels').
        worm_type (str): Indicates whether the data is from 'ATR+' or 'ATR-' worms.

    Returns:
        None
    """
    feature_column_before = f"{feature} Before"
    feature_column_during = f"{feature} During"

    if feature_column_before not in behavior_comparisons.columns or feature_column_during not in behavior_comparisons.columns:
        print(f"Columns {feature_column_before} and {feature_column_during} are missing in behavior comparisons.")
        return

    # Prepare data for visualization
    melted_data = pd.melt(
        behavior_comparisons,
        id_vars=['Pulse Start', 'Pulse End'],
        value_vars=[feature_column_before, feature_column_during],
        var_name='Condition',
        value_name=feature
    )

    # Rename conditions for better visualization
    melted_data['Condition'] = melted_data['Condition'].str.replace(' Before', ' Before Pulse')
    melted_data['Condition'] = melted_data['Condition'].str.replace(' During', ' During Pulse')

    # Plot using Seaborn
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=melted_data, x='Condition', y=feature)
    plt.title(f"{feature} Before and During Light Pulses ({worm_type} Worms)", fontsize=14)
    plt.ylabel(feature, fontsize=12)
    plt.xlabel("Condition", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot
    repo_root = hp.find_repo_root()
    output_dir = os.path.join(repo_root, 'Data', 'Plots', 'light_pulses', 'behavior_changes')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{feature}_comparison_{worm_type}.png")
    plt.savefig(output_path)

    plt.close()