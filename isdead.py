from IPython.display import display
import pandas as pd
import numpy as np

def calculate_absolute_time(absolute_frame):
    """
    Calculate the total time in hours for a given absolute frame number 

    Parameters:
    - absolute_frame (int): The frame number to calculate the absolute time for.

    Returns:
    - float: The total time in hours corresponding to the given frame.

    Explanation:
    - Each frames has a duration of 2 sec
    - The dataset is divided into segments, each containing 12 sessions.
    - Each session consists of 900 frames and spans 1800 seconds (30 minutes).
    - A gap of 19800 seconds (5.5 hours) exists between sessions.
    - Each segment spans 259200 seconds (72 hours) and contains 10800 frames.

    Steps:
    1. Determine which segment the absolute frame belongs to by dividing the
       frame number by the total number of frames in a segment.
    2. Calculate the frame offset within the segment using the modulus operator.
    3. Determine the session number within the segment by dividing the frame
       offset by the number of frames in a session.
    4. Calculate the frame offset within the session using the modulus operator.
    5. Compute the time contributions from segments, sessions, and frames:
       - Time from segments: segment_number * segment_duration_sec
       - Time from sessions: session_number_within_segment * total_session_duration_sec
       - Time from frames: frame_offset_within_session * 2 (each frame = 2 seconds)
    6. Add these time contributions to get the total time in seconds.
    7. Convert the total time from seconds to hours and return.

    Note:
    - The time calculation method is based on the dataset's specific segmentation 
      and session structure, which may seem unusual but is consistent with the data design.
    """
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
    Estimate the dying time of a worm based on its last active frame.

    Args:
        df (pd.DataFrame): DataFrame containing worm behavioral data with columns 'Speed', 'X', 'Y', 'changed pixels'
        movement_threshold (float): Threshold below which movement in 'X' and 'Y' is considered negligible.

    Returns:
        tuple: 
            - dying_frame (int): Frame index where the worm is estimated to have died (in the current 'segment')
            - absolute_frame (int): Absolute frame number corresponding to the dying frame.
            - dying_time_hours (float): Estimated time of death in hours from the start of the recording.
            - segment_number (int): Segment number where the estimated death occurs.

    Methodology:
    1. Compute the absolute frame number and the corresponding segment for each row.
    2. Calculate frame-to-frame differences in 'X' and 'Y' to track movement.
    3. Identify inactive frames where movement in both 'X' and 'Y' falls below the threshold.
    4. Iterate backwards through the data to find the last active frame.
    5. Use the `calculate_absolute_time` function to estimate the time of death in hours.
    """
    # Step 1: Prepare the DataFrame by creating a temporary copy and dropping unnecessary columns
    df_temp = df.copy() # Create a temporary copy of dataframe
    df_temp = df_temp.drop(['Speed','Changed Pixels'], axis=1) # drop columns not needed for this calculation

    # Step 2: Add columns for absolute frame numbers and segment identification
    df_temp['Absolute_Frame'] = df.index + 1  # Absolute frame starts from 1
    df_temp['Segment'] = df_temp['Absolute_Frame'] // 10800 # Each segment has 10800 frames

    # Step 3: Calculate movement (delta) in 'X' and 'Y' coordinates
    df_temp['Delta_X'] = df_temp['X'].diff().abs().fillna(0) # Fill first frame with 0, because no movement before that
    df_temp['Delta_Y'] = df_temp['Y'].diff().abs().fillna(0) 

    # Step 4: Identify inactivity frames where BOTH 'X' AND 'Y' movement is negligible
    df_temp['Inactivity'] = ( # so = 1 if BOTH are < threshold , means the worm is inactive
    (df_temp['Delta_X'] < movement_threshold) &  # Check if Delta_X is below the threshold
    (df_temp['Delta_Y'] < movement_threshold)    # Check if Delta_Y is below the threshold
    ).astype(int)  # Convert the resulting boolean series to integers (1 = True = INACTIVE, 0 = False = ACTIVE)

    # Step 5: Iterate backwards to find the last active frame
    for i in range(len(df_temp) - 1, -1, -1): # Reverse iteration
        if df_temp.loc[i, 'Inactivity'] == 0:  # Found the last active frame
            dying_index = i + 1  # Start of inactivity

            # Edge case: If dying_index exceeds the DataFrame length, return None
            if dying_index >= len(df_temp): 
                return None, None, None, None

            # Extract information about the dying frame
            dying_frame = df_temp.loc[dying_index, 'Frame']
            absolute_frame = df_temp.loc[dying_index, 'Absolute_Frame']
            segment_number = df_temp.loc[dying_index, 'Segment']
            dying_time_hours = calculate_absolute_time(absolute_frame)
            
          
            # Optional : Add detailed explanation

            # print("\nDetailed Output:")
            # print(f"- The worm most probably died at the Absolute Frame {absolute_frame}.")
            # print(f"- This correspond to the frame {dying_frame} of the {segment_number}'th segment(starts at segment).")
            # print(f"- The estimated time of death is {dying_time_hours:.2f} hours since the start of recording.\n")

            return dying_frame,absolute_frame,dying_time_hours,segment_number

    # Step 6: Handle case where no activity is detected throughout the DataFrame
    print("No activity detected; worm inactive throughout.")
    return None, None, None, None



 
