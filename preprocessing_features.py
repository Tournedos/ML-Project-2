import numpy as np
from scipy.signal import find_peaks

def summarize_feature(series):
    """
    Summarize a time series with statistical features.

    Parameters:
        series (numpy.ndarray): Input time series.

    Returns:
        dict: Statistical summaries.
    """
    return {
        'mean': np.mean(series),
        'std': np.std(series),
        'min': np.min(series),
        'max': np.max(series),
        'range': np.max(series) - np.min(series),
        'skewness': np.mean((series - np.mean(series))**3) / (np.std(series)**3 + 1e-8),
        'kurtosis': np.mean((series - np.mean(series))**4) / (np.std(series)**4 + 1e-8) - 3
    }

def compute_direction_changes(x, y):
    """
    Compute variance in direction changes based on X and Y.

    Parameters:
        x, y (numpy.ndarray): X and Y coordinates.

    Returns:
        float: Variance of angular changes in directions.
    """
    dx = np.diff(x)
    dy = np.diff(y)
    angles = np.arctan2(dy, dx)
    return np.var(np.diff(angles))

def compute_speed(x, y):
    """
    Compute speed from X and Y coordinates.

    Parameters:
        x, y (numpy.ndarray): X and Y coordinates.

    Returns:
        numpy.ndarray: Speed time series.
    """
    speed = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    return np.pad(speed, (1, 0), mode='constant', constant_values=0)

def summarize_speed(speed):
    """
    Summarize speed with additional temporal statistics.

    Parameters:
        speed (numpy.ndarray): Speed time series.

    Returns:
        dict: Speed summaries including peaks and inactivity.
    """
    peaks, _ = find_peaks(speed)
    return {
        'peak_count': len(peaks),
        'time_to_max_speed': np.argmax(speed),
        'inactivity_duration': np.sum(speed < 0.01)  # Speed below threshold
    }


# 1. Higher-Order Statistical Features
def compute_entropy(values):
    """
    Compute the entropy of a series of values.

    Parameters:
        values (numpy.ndarray): Input values.

    Returns:
        float: Entropy of the values.
    """
    probabilities = np.histogram(values, bins=10, density=True)[0]
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log(probabilities))

def compute_coefficient_of_variation(values):
    """
    Compute the coefficient of variation (std/mean).

    Parameters:
        values (numpy.ndarray): Input values.

    Returns:
        float: Coefficient of variation.
    """
    mean = np.mean(values)
    std = np.std(values)
    return std / mean if mean != 0 else 0

# 2. Temporal Dynamics
def compute_acceleration_stats(speed):
    """
    Compute the standard deviation of acceleration.

    Parameters:
        speed (numpy.ndarray): Speed values.

    Returns:
        float: Standard deviation of acceleration.
    """
    acceleration = np.diff(speed)
    return np.std(acceleration)

# 3. Spatial Features
def compute_convex_hull_area(x, y):
    """
    Compute the convex hull area of the trajectory.

    Parameters:
        x, y (numpy.ndarray): X and Y coordinates.

    Returns:
        float: Area of the convex hull.
    """
    from scipy.spatial import ConvexHull
    points = np.column_stack((x, y))
    hull = ConvexHull(points)
    return hull.area

def compute_path_efficiency(x, y):
    """
    Compute path efficiency as total displacement / cumulative distance.

    Parameters:
        x, y (numpy.ndarray): X and Y coordinates.

    Returns:
        float: Path efficiency.
    """
    displacement = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    total_distance = np.sum(distances)
    return displacement / total_distance if total_distance != 0 else 0

# 4. Frequency-Domain Features
def compute_spectral_entropy(values):
    """
    Compute the spectral entropy of a signal.

    Parameters:
        values (numpy.ndarray): Input signal.

    Returns:
        float: Spectral entropy.
    """
    fft_vals = np.abs(np.fft.fft(values))
    probabilities = fft_vals / np.sum(fft_vals)
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log(probabilities))

def compute_dominant_frequency(values):
    """
    Compute the dominant frequency in the signal.

    Parameters:
        values (numpy.ndarray): Input signal.

    Returns:
        float: Dominant frequency.
    """
    fft_vals = np.abs(np.fft.fft(values))
    freqs = np.fft.fftfreq(len(values))
    dominant_freq = freqs[np.argmax(fft_vals)]
    return dominant_freq

# 5. Event-Based Features
def compute_burst_count(speed, threshold=0.5):
    """
    Compute the number of bursts of high speed.

    Parameters:
        speed (numpy.ndarray): Speed values.
        threshold (float): Speed threshold for bursts.

    Returns:
        int: Number of bursts.
    """
    return np.sum(speed > threshold)

def compute_inactivity_events(speed, threshold=0.1):
    """
    Compute the number of inactivity events (speed below threshold).

    Parameters:
        speed (numpy.ndarray): Speed values.
        threshold (float): Speed threshold for inactivity.

    Returns:
        int: Number of inactivity events.
    """
    return np.sum(speed < threshold)

# 6. Behavioral Patterns
def compute_asymmetry_index(x, y):
    """
    Compute the asymmetry index (ratio of movement in X vs Y).

    Parameters:
        x, y (numpy.ndarray): X and Y coordinates.

    Returns:
        float: Asymmetry index.
    """
    return np.sum(np.abs(np.diff(x))) / np.sum(np.abs(np.diff(y))) if np.sum(np.abs(np.diff(y))) != 0 else 0

def compute_mean_turning_angle(x, y):
    """
    Compute the mean turning angle between consecutive movement vectors.

    Parameters:
        x, y (numpy.ndarray): X and Y coordinates.

    Returns:
        float: Mean turning angle.
    """
    dx = np.diff(x)
    dy = np.diff(y)
    angles = np.arctan2(dy, dx)
    return np.mean(np.abs(np.diff(angles)))


def create_aug(x,y,seed=42):
    #x_opp = np.max(x) - x
    #y_opp = np.max(y) - y
    np.random.seed(seed)
    noise_x = np.random.normal(0,1,len(x))
    noise_y = np.random.normal(0,1,len(y))
    return x + noise_x, y + noise_y


def preprocess_dataset(samples):
    """
    Process the dataset and extract summarized features for each sample.

    Parameters:
        samples (list[numpy.ndarray]): List of arrays of shape (frames, 4) for each sample.

    Returns:
        numpy.ndarray: Array of feature vectors for all samples.
    """
    all_features = []

    for sample in samples:
        # Extract X, Y, and speed
        x = sample[:, 0]
        y = sample[:, 1]
        speed = sample[:, 2]  # Assuming speed is precomputed, else use compute_speed(x, y)
        cspeed = compute_speed(x,y)
        if (speed != cspeed).any():
            speed = cspeed

        # Summarize X and Y
        x_features = summarize_feature(x)
        y_features = summarize_feature(y)

        # Compute direction changes
        direction_change_variance = compute_direction_changes(x, y)
        entropy_x = compute_entropy(x)
        entropy_y = compute_entropy(y)
        entropy_v = compute_entropy(speed)
        cofvar_x = compute_coefficient_of_variation(x)
        cofvar_y = compute_coefficient_of_variation(y)
        cofvar_v = compute_coefficient_of_variation(speed)
        acc_std = compute_acceleration_stats(speed)
        h_ar = compute_convex_hull_area(x,y)
        patheff = compute_path_efficiency(x,y)
        #entrioy....
        #...


        # Summarize speed
        speed_features = summarize_feature(speed)
        speed_temporal_features = summarize_speed(speed)

        # Combine all features
        features = {
            **{f'x_{k}': v for k, v in x_features.items()},
            **{f'y_{k}': v for k, v in y_features.items()},
            'direction_change_variance': direction_change_variance,
            'entropy_x' : entropy_x,
            'entropy_y' : entropy_y,
            'entropy_speed' : entropy_v,
            'ceff_var_x' : cofvar_x,
            'coeff_var_y' : cofvar_y,
            'coeff_var_v' : cofvar_v,
            'acceleration_std' : acc_std,
            'hull_convex_area' : h_ar,
            'path_eff' : patheff,
            **{f'speed_{k}': v for k, v in speed_features.items()},
            **{f'speed_temporal_{k}': v for k, v in speed_temporal_features.items()}
        }

        all_features.append(features)

    return np.array(all_features)
