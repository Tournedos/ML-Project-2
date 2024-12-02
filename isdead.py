from IPython.display import display
import pandas as pd
import numpy as np

def calculate_absolute_time(absolute_frame):
    session_frames = 900
    session_duration_sec = 1800 # seconds
    gap_duration_sec = 19800 # 5.5 hours = 19800 seconds
    total_session_duration_sec = session_duration_sec + gap_duration_sec # 1800 seconds + gap_duration
    session_per_segment = 12
    segment_duration_sec = session_per_segment * total_session_duration_sec # 12 * (1800+19800) = 259200 seconds
    segment_frames= 10800 # frames in a segment = 900 * 12 = 10800

    
    segment_number = absolute_frame // segment_frames # Step 1: Calculate which segment the absolute frame belongs to

    frame_within_segment = absolute_frame % segment_frames # Step 2: Calculate frame offset within the current segment

    session_number_within_segment = frame_within_segment // session_frames # Step 3: Calculate which session within the segment the frame belongs to

    frame_offset_within_session = frame_within_segment % session_frames # Step 4: Calculate frame offset within the session

    time_from_segments_sec = segment_number * segment_duration_sec # Step 5: Compute the total time
    time_from_sessions_sec = session_number_within_segment * total_session_duration_sec
    time_from_frames_sec = frame_offset_within_session * 2  # Each frame is 2 seconds

    total_time_sec = time_from_segments_sec + time_from_sessions_sec + time_from_frames_sec # Total time in seconds

    total_time_hr = total_time_sec / 3600  # Convert seconds to hours

    return total_time_hr


def estimate_dying_time(df, movement_threshold) : #, inactivity_threshold_frames=900, movement_threshold=1.0):
    """
    Estimate the dying time of a worm based on the last active frame.

    Args:
        df (pd.DataFrame): Worm data containing 'X', 'Y', and other columns.
        movement_threshold (float): Threshold for negligible movement in X and Y.
        inactivity_frames (int): Minimum length of inactivity to consider the worm dead.

    Returns:
        tuple: Detailed output about dying frame, time, and segment.
    """
    # Prepare a copy of the dataframe
    df_temp = df.copy() # Create a temporary copy of dataframe
    df_temp = df_temp.drop(['Speed','Changed Pixels'], axis=1)

    # Add a column for frame resets and absolute frames
    df_temp['Absolute_Frame'] = df.index + 1
    df_temp['Segment'] = df_temp['Absolute_Frame'] // 10800

    # Calculate changes in X and Y
    df_temp['Delta_X'] = df_temp['X'].diff().abs().fillna(0) # Fill first frame with 0, because no movement before that
    df_temp['Delta_Y'] = df_temp['Y'].diff().abs().fillna(0)

    # Identify frames with simultaneous negligible movement in both X and Y
    df_temp['Inactivity'] = ( # so = 1 if BOTH are < threshold , means the worm is inactive
    (df_temp['Delta_X'] < movement_threshold) &  # Check if Delta_X is below the threshold
    (df_temp['Delta_Y'] < movement_threshold)    # Check if Delta_Y is below the threshold
    ).astype(int)  # Convert the resulting boolean series to integers (1 for True, 0 for False)

    ###################

    # Reverse iteration to find the last activity
    for i in range(len(df_temp) - 1, -1, -1):
        if df_temp.loc[i, 'Inactivity'] == 0:  # Last active frame
            dying_index = i + 1  # Start of inactivity
            if dying_index >= len(df_temp):  # Edge case: no inactivity
                return None, None, None

            dying_frame = df_temp.loc[dying_index, 'Frame']
            absolute_frame = df_temp.loc[dying_index, 'Absolute_Frame']
            segment_number = df_temp.loc[dying_index, 'Segment']
            dying_time_hours = calculate_absolute_time(absolute_frame)
            
          
            # Add detailed explanation
            print("\nDetailed Output:")
            print(f"- The worm most probably died at the Absolute Frame {absolute_frame}.")
            print(f"- This correspond to the frame {dying_frame} of the {segment_number}'th segment(starts at segment 0).")
            print(f"- The estimated time of death is {dying_time_hours:.2f} hours since the start of recording.\n")

            return dying_frame,absolute_frame,dying_time_hours,segment_number

    # No activity detected at all
    print("No activity detected; worm inactive throughout.")
    return None, None, None



 
