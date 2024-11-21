import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics.pairwise import haversine_distances

from navi import plane_sailing_course_speed, plane_sailing_next_position, haversine


def geo_to_cartesian(df):
    """
    Transforms latitude and longitude into Cartesian coordinates (x, y, z).

    Parameters:
        df (pandas.DataFrame): DataFrame with 'lat' and 'lon' columns.

    Returns:
        pandas.DataFrame: DataFrame with added 'x', 'y', 'z' columns.
    """
    # Convert lat/lon from degrees to radians
    lat_rad = np.radians(df['lat'])
    lon_rad = np.radians(df['lon'])

    # Compute x, y, z using spherical to Cartesian transformation
    df = df.assign(x=np.cos(lat_rad) * np.cos(lon_rad))
    df = df.assign(y=np.cos(lat_rad) * np.sin(lon_rad))
    df = df.assign(z=np.sin(lat_rad))

    return df


def cartesian_to_geo(x, y, z):
    """
    Transforms Cartesian coordinates (x, y, z) back to latitude and longitude.

    Parameters:
        df (pandas.DataFrame): DataFrame with 'x', 'y', 'z' columns.

    Returns:
        pandas.DataFrame: DataFrame with added 'lat' and 'lon' columns.
    """
    # Calculate latitude (in radians)
    lat = np.round(np.degrees(np.arcsin(z)), 1)

    # Calculate longitude (in radians), using arctan2 to handle all quadrants
    lon = np.round(np.degrees(np.arctan2(y, x)), 1)

    return lat, lon


def calculate_velocity_direction(df):
    """
    Calculates velocity (kn) and direction (bearing in degrees) between consecutive points and haversine distance.
    """
    df = df.copy()

    velocity = np.zeros(len(df))
    direction = np.zeros(len(df))
    # distance = np.zeros(len(df))

    for i in range(len(df)):
        # velocity and direction for the first record
        if i == 0:
            velocity[i] = 0
            direction[i] = 0
            continue

        # check previous name is different from current name
        previous_name = df.iloc[i - 1]['name']
        current_name = df.iloc[i]['name']

        if current_name != previous_name:
            velocity[i] = 0
            direction[i] = 0
            continue

            # Get the previous and current coordinates
        prev_point = (df.iloc[i - 1]['lat'], df.iloc[i - 1]['lon'])
        curr_point = (df.iloc[i]['lat'], df.iloc[i]['lon'])

        # Calculate time difference in hours
        time_diff = (df.index[i] - df.index[i - 1]).total_seconds() / 3600

        # check time difference more than 6 hrs
        if time_diff > 6 or time_diff == 0:
            velocity[i] = 0
            direction[i] = 0
            continue

        R = 3440.065  # earth radius in NM

        # Calculate direction and velocity
        direction[i], velocity[i] = plane_sailing_course_speed(prev_point, curr_point, time_interval=time_diff)
        # distance[i] = (haversine_distances([np.radians(prev_point), np.radians(curr_point)]) * R)[0][1]

    df['velocity_kn'] = velocity
    df['direction_deg'] = direction
    # df['hav_distance'] = distance

    return df


def convert_direction_to_sin_cosin(df):
    direction_rad = np.deg2rad(df['direction_deg'])

    # Create sine and cosine components
    df = df.assign(direction_sin=np.sin(direction_rad))
    df = df.assign(direction_cos=np.cos(direction_rad))

    return df


def get_direction_from_sin_cos(sin_val, cos_val):
    """
    Reversed bearing in degrees from sin and cosin values.

    Usage:
    df['direction_deg_reversed'] = df.apply(lambda row: get_direction_from_sin_cos(row['direction_sin'], row['direction_cos']), axis=1)
    """
    # Get the angle in radians
    angle_rad = np.arctan2(sin_val, cos_val)

    # Convert radians to degrees
    angle_deg = np.degrees(angle_rad)

    # Make sure the angle is within the range 0° to 360°
    if angle_deg < 0:
        angle_deg += 360

    return int(angle_deg)


def generate_training_dataframe(df):
    def shift_group(df_group):
        """
        Shift features to create target variables for the next observation within each group (TD).
        """
        df_group['next_x'] = df_group['x'].shift(-1)
        df_group['next_y'] = df_group['y'].shift(-1)
        df_group['next_z'] = df_group['z'].shift(-1)
        df_group['next_max_wind_kn'] = df_group['max_wind_kn'].shift(-1)
        df_group['next_min_pressure_mBar'] = df_group['min_pressure_mBar'].shift(-1)
        # df_group['next_velocity_kn'] = df_group['velocity_kn'].shift(-1)
        # df_group['next_direction_sin'] = df_group['direction_sin'].shift(-1)
        # df_group['next_direction_cos'] = df_group['direction_cos'].shift(-1)
        # df_group['next_hav_distance'] = df_group['hav_distance'].shift(-1)
        return df_group

    # Shift features within each group to create targets
    df = df.groupby('group').apply(shift_group, include_groups=False)

    # Reset index and drop unnecessary columns
    df = df.reset_index()
    df.index = pd.to_datetime(df['date'])
    df = df.drop(columns=['date'])

    # drop NaN containing lines
    df = df.dropna()

    return df


def prepare_dataframe(df):
    """
    Prepares the input DataFrame for further analysis and machine learning modeling by performing several transformations:

    1. Transforms the index to a `datetime` format.
    2. Reads the ENSO (El Niño Southern Oscillation) phase data from a CSV file and adds an 'enso' feature to the DataFrame.
    3. Fills missing tropical depression names with 'UNNAMED' where applicable.
    4. Keeps only relevant columns: 'name', 'lat', 'lon', 'max_wind_kn', 'min_pressure_mBar', and 'enso'.
    5. Calculates velocity and direction based on latitude and longitude.
    6. Converts directional degrees into sine and cosine components for better model representation.
    7. Converts geographical coordinates (latitude, longitude) into 3D Cartesian coordinates (x, y, z).
    8. Generates training data by lagging one observation.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing tropical depression data with columns including latitude, longitude, wind speed, pressure, and name.

    Returns:
    --------
    pd.DataFrame
        A cleaned and transformed DataFrame with the following columns:
        - 'name': The name of the tropical depression.
        - 'lat': Latitude of the tropical depression.
        - 'lon': Longitude of the tropical depression.
        - 'max_wind_kn': Maximum wind speed (in knots).
        - 'min_pressure_mBar': Minimum central pressure (in millibars).
        - 'enso': The ENSO phase (1 for El Niño, -1 for La Niña, and 0 for Neutral).
        - 'velocity_kn': Velocity of the tropical depression (calculated from lat/lon).
        - 'direction_sin': Sine of the direction angle.
        - 'direction_cos': Cosine of the direction angle.
        - 'x', 'y', 'z': Cartesian coordinates of the tropical depression based on latitude and longitude.
    """

    def import_enso_to_df(row, enso_df):
        """
        Retrieves the ENSO (El Niño Southern Oscillation) phase for a given row based on its year.

        This function is designed to be applied row-wise to a DataFrame, and it assigns an ENSO phase (El Niño, La Niña, or Neutral)
        to each record based on the year of the tropical depression. ENSO phase values are drawn from a separate DataFrame containing
        the ENSO data for each year.

        Parameters:
        ----------
        row : pd.Series
            A row of the DataFrame being processed. The index of the row should contain datetime values from which the year can be extracted.

        enso_df : pd.DataFrame
            A DataFrame containing ENSO phase information, where each row corresponds to a specific year and includes an 'enso' column
            that holds the ENSO phase (-1 for La Niña, 0 for Neutral, 1 for El Niño).

        Returns:
        --------
        int
            The ENSO phase for the corresponding year of the input row. The function returns:
            - -1 for La Niña,
            - 0 for Neutral,
            - 1 for El Niño.
            If the year is 1949 (for specific handling), the function returns -1 (La Niña as a placeholder).

        Usage:
        ------
        df['enso'] = df.apply(import_enso_to_df, axis=1, enso_df=enso_df)

        Notes:
        ------
        - This function assumes that the DataFrame's index is of `datetime` type, allowing access to the year using `row.name.year`.
        """
        year = row.name.year
        if year == 1949 or year == 2024:
            return -1
        return enso_df.loc[enso_df['year'] == year].enso.values[0]

    def shift_velocity_direction(df_group):
        """
        Shifts velocity and direction and hav distance within each TD group (id), ensuring that the last observation has 0 velocity and 0 direction.
        """
        # Shift velocity and direction for each group
        df_group['velocity_kn'] = df_group['velocity_kn'].shift(-1)
        df_group['direction_deg'] = df_group['direction_deg'].shift(-1)
        # df_group['hav_distance'] = df_group['hav_distance'].shift(-1)

        # Replace NaN values in the last observation of each group with 0 (because TD dissipates)
        df_group[['velocity_kn', 'direction_deg']] = df_group[
            ['velocity_kn', 'direction_deg']].fillna(0)

        return df_group

    # Check if dataframe already has been transformed:
    if not all(col in df.columns for col in ['id', 'velocity_kn', 'direction_deg', 'enso']):
        # transform index column to datetime
        df.index = pd.to_datetime(df.index)

        # read enso data and add the feature to df
        enso_df = pd.read_csv('data/csv_ready/enso_years.csv')
        df.loc[:, 'enso'] = df.apply(import_enso_to_df, axis=1, enso_df=enso_df)

        # fix the NaN of the TDs name
        df.loc[:, 'name'] = df['name'].apply(lambda x: 'UNNAMED' if pd.isna(x) else x)

        df = df[['name', 'lat', 'lon', 'max_wind_kn', 'min_pressure_mBar', 'enso']]

        df = calculate_velocity_direction(df)

        # adding consecutive count of the TDs in order to be able to group and split the datasets
        new_td_starts = (df['velocity_kn'] == 0) & (df['direction_deg'] == 0)
        ids = new_td_starts.cumsum()
        df['group'] = ids

        df = df.groupby('group').apply(shift_velocity_direction)

        # Drop the 'group' column temporarily to avoid conflict when resetting the index
        df = df.drop(columns=['group'])

        # Reset index and drop unnecessary columns
        df = df.reset_index()
        df.index = pd.to_datetime(df['date'])
        df = df.drop(columns=['date'])

        # Add back the 'group' column after resetting the index
        df['group'] = ids

    df = convert_direction_to_sin_cosin(df)
    df = geo_to_cartesian(df)

    return df


def split_dataframe(df, splitter='gss', n_splits=1, test_size=0.2, random_state=97):
    # Define the features and target columns
    X = df[['x', 'y', 'z', 'max_wind_kn', 'min_pressure_mBar', 'velocity_kn',
            'direction_sin', 'direction_cos', 'enso']]
    y = df[['next_x', 'next_y', 'next_z', 'next_max_wind_kn', 'next_min_pressure_mBar', ]]
    # 'next_velocity_kn', 'next_direction_sin', 'next_direction_cos']]

    # Use the specified splitter
    if splitter == 'gkf':
        splitter = GroupKFold(n_splits=n_splits)
    elif splitter == 'gss':
        splitter = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    else:
        raise ValueError("Invalid splitter type. Use 'gkf' for GroupKFold or 'gss' for GroupShuffleSplit.")

    groups = df['group']

    # Perform the split and yield train/test sets
    for train_idx, test_idx in splitter.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        yield X_train, X_test, y_train, y_test


def manage_prediction(df, model):
    """
        Predict the next 6-hour position based on the current dataframe `df`.
    """

    predicted_values = model.predict(df)
    x, y, z, max_wind_kn, min_pressure_mBar = predicted_values[0]
    # x, y, z, max_wind_kn, min_pressure_mBar, velocity_kn, direction_sin, direction_cos = predicted_values[0]

    lat, lon = cartesian_to_geo(x, y, z)
    min_pressure_mBar = int(min_pressure_mBar)
    max_wind_kn = np.round(max_wind_kn, 1)
    # velocity_kn = np.round(velocity_kn, 1)
    # direction_deg = get_direction_from_sin_cos(direction_sin, direction_cos)
    # hav_distance = np.round(hav_distance, 1)

    # return lat, lon, max_wind_kn, min_pressure_mBar, velocity_kn, direction_deg
    return lat, lon, max_wind_kn, min_pressure_mBar


def create_real_pred_df(df, df_plot, model, n_samples, random_state, intervals=1):
    groups = get_groups(df, df_plot, n_samples, random_state)

    data = []

    # Loop through each group and collect the predicted and real data
    for idx, group in enumerate(groups):
        # Get the group data
        pred = df[df.group == group].copy()

        # Shift velocity and direction to match the method
        pred.loc[:, 'velocity_kn'] = pred.velocity_kn.shift(1)
        pred.loc[:, 'direction_deg'] = pred.direction_deg.shift(1)
        # pred.loc[:, 'hav_distance'] = pred.hav_distance.shift(1)
        pred = pred.dropna()

        if intervals == 1:
            # Loop through each point in the group to generate predictions and collect real values
            for i in range(1, len(pred) + 1):
                # Get the predicted values
                predicted = manage_prediction(pred[i - 1: i], model)

                # Real values
                real_row = pred.iloc[i - 1]

                # Values that predictions are based upon
                initial_row = pred.iloc[i - 2]

                pred_direction_deg, pred_velocity_kn = plane_sailing_course_speed(
                    [initial_row['lat'], initial_row['lon']],
                    [predicted[0], predicted[1]])

                # Create a dictionary to store both real and predicted values
                data_row = {
                    'date': real_row.name,
                    'group': real_row['group'],
                    'name': real_row['name'],

                    # Real values
                    'real_lat': real_row['lat'],
                    'real_lon': real_row['lon'],
                    'real_max_wind_kn': real_row['max_wind_kn'],
                    'real_min_pressure_mBar': real_row['min_pressure_mBar'],
                    'real_velocity_kn': real_row['velocity_kn'],
                    'real_direction_deg': real_row['direction_deg'],
                    # 'real_hav_distance': real_row['hav_distance'],

                    # Predicted values
                    'pred_lat': predicted[0],
                    'pred_lon': predicted[1],
                    'pred_max_wind_kn': predicted[2],
                    'pred_min_pressure_mBar': predicted[3],
                    'expected_velocity_kn': pred_velocity_kn,
                    'expected_direction_deg': pred_direction_deg,
                }

                # Append the dictionary to the data list
                data.append(data_row)

        # if intervals are more than one
        else:
            # Multi-step prediction: Generate predictions for multiple intervals
            initial_row = pred.sample(n=1, random_state=random_state)
            initial_idx = pred.reset_index().sample(n=1, random_state=random_state).index[0]

            # Get initial row and consecutive rows for real data
            if initial_idx + intervals <= len(pred):
                real_df = pred.reset_index().loc[initial_idx:initial_idx + intervals - 1].copy()
            else:
                # If not enough rows after initial_idx, get the last possible consecutive rows
                real_df = pred.reset_index().iloc[-intervals:].copy()

            real_df = real_df.reset_index(drop=True)

            # Generate predictions for the next intervals
            for i in range(intervals):
                predicted = manage_prediction(initial_row, model)

                # Update real_df with predicted data for the current interval
                real_df.loc[i, 'pred_lat'] = predicted[0]
                real_df.loc[i, 'pred_lon'] = predicted[1]
                real_df.loc[i, 'pred_max_wind_kn'] = predicted[2]
                real_df.loc[i, 'pred_min_pressure_mBar'] = predicted[3]

                calc_direction_deg, calc_velocity_kn = plane_sailing_course_speed(
                    [initial_row['lat'].values[0], initial_row['lon'].values[0]],
                    [predicted[0], predicted[1]])
                calc_hav_distance = haversine(
                    [initial_row['lat'].values[0], initial_row['lon'].values[0]],
                    [predicted[0], predicted[1]])

                real_df.loc[i, 'pred_velocity_kn'] = calc_velocity_kn
                real_df.loc[i, 'pred_direction_deg'] = calc_direction_deg
                # real_df.loc[i, 'pred_hav_distance'] = calc_hav_distance

                # Generate df based on predictions to pass for next interval prediction
                initial_row = pd.DataFrame({
                    'date': initial_row.name,
                    'lat': [predicted[0]],
                    'lon': [predicted[1]],
                    'max_wind_kn': [predicted[2]],
                    'min_pressure_mBar': [predicted[3]],
                    'velocity_kn': calc_velocity_kn,
                    'direction_deg': calc_direction_deg,
                    # 'hav_distance': calc_hav_distance,
                    'group': [initial_row['group'].values[0]],
                    'name': [initial_row['name'].values[0]]
                })

            # Append the real_df to the data list
            data.append(real_df)

    if intervals == 1:
        predicted_vs_real_df = pd.DataFrame(data)

        columns = ['pred_lat',
                   'pred_lon',
                   'pred_max_wind_kn',
                   'pred_min_pressure_mBar',
                   'expected_velocity_kn',
                   'expected_direction_deg']
        for col in columns:
            predicted_vs_real_df[col] = shift_row(predicted_vs_real_df.groupby('group'), col, 1)

        predicted_vs_real_df = predicted_vs_real_df.dropna()
    else:
        # Convert the list of dictionaries into a DataFrame
        predicted_vs_real_df = pd.concat(data, ignore_index=True)

    predicted_vs_real_df.index = predicted_vs_real_df.date
    predicted_vs_real_df = predicted_vs_real_df.drop(columns='date')

    return predicted_vs_real_df


def shift_row(df, col_name, direction):
    return df[col_name].shift(direction)


def get_groups(df, df_plot, n_samples, random_state):
    df.index = pd.to_datetime(df.index)
    samples = df_plot.sample(n=n_samples, random_state=random_state)
    groups = df[df.index.isin(samples.index)].group.values
    return groups


def plot_tds(df, df_plot, model, n_samples, random_state):
    groups = get_groups(df, df_plot, n_samples, random_state)
    # Number of groups
    num_groups = len(groups)

    # Set up subplots (3 plots per row): 3 plots for each group (Track, Wind, and Pressure)
    num_cols = 3  # One column for each: Track, Wind, Pressure
    num_rows = num_groups  # One row per TD

    # Create subplots for lat/lon comparison, wind comparison, and pressure comparison
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, num_rows * 6))

    # Flatten axes to easily iterate
    axes = axes.reshape(num_rows, num_cols)

    # Loop through each group and make the plots
    for idx, group in enumerate(groups):
        # Select subplot axes for the current group: Track, Wind, Pressure
        track_ax = axes[idx, 0]  # Track (Lat/Lon)
        wind_ax = axes[idx, 1]  # Wind
        pressure_ax = axes[idx, 2]  # Pressure

        # Get the group data
        pred = df[df.group == group]

        # Shift velocity and direction
        pred.loc[:, 'velocity_kn'] = pred.velocity_kn.shift(1)
        pred.loc[:, 'direction_deg'] = pred.direction_deg.shift(1)
        # pred.loc[:, 'hav_distance'] = pred.hav_distance.shift(1)
        pred = pred.dropna()

        # Plot Latitude/Longitude track comparison (Real vs Predicted)
        for i in range(1, len(pred) + 1):
            predicted = manage_prediction(pred[i - 1: i], model)
            pred_lon, pred_lat = predicted[1], predicted[0]
            if pred_lon < 0:
                pred_lon += 360
            if i == 1:  # Only add label for the first point to avoid duplication in the legend
                track_ax.scatter(pred_lon, predicted[0], label="Predicted Position", c='b')
            else:
                track_ax.scatter(pred_lon, pred_lat, c='b')

        # Plot real values for latitude/longitude in red
        pred['lon'] = pred['lon'].apply(lambda x: x if x >= 0 else 360 + x)
        track_ax.plot(pred.lon, pred.lat, c='r', label="Real Track")

        # Set labels and title for the track plot
        track_ax.set_title(f'Track (TD {pred.name.values[0]} ID: {group})')
        track_ax.set_xlabel('Longitude')
        track_ax.set_ylabel('Latitude')
        track_ax.legend()

        # Plot predicted and real wind speed
        for i in range(1, len(pred) + 1):
            predicted = manage_prediction(pred[i - 1: i], model)
            if i == 1:
                wind_ax.scatter(i, predicted[2], label="Predicted Wind", c='b')
            else:
                wind_ax.scatter(i, predicted[2], c='b')
        wind_ax.plot(range(len(pred)), pred.max_wind_kn, c='r', label="Real Wind")
        wind_ax.set_title(f'Wind Speed Comparison (TD {pred.name.values[0]} ID: {group})')
        wind_ax.set_xlabel('Time Step')
        wind_ax.set_ylabel('Wind Speed (kn)')
        wind_ax.legend()

        # Plot predicted and real pressure
        for i in range(1, len(pred) + 1):
            predicted = manage_prediction(pred[i - 1: i], model)
            if i == 1:
                pressure_ax.scatter(i, predicted[3], label="Predicted Pressure", c='b')
            else:
                pressure_ax.scatter(i, predicted[3], c='b')
        pressure_ax.plot(range(len(pred)), pred.min_pressure_mBar, c='r', label="Real Pressure")
        pressure_ax.set_title(f'Pressure Comparison (TD {pred.name.values[0]} ID: {group})')
        pressure_ax.set_xlabel('Time Step')
        pressure_ax.set_ylabel('Pressure (mBar)')
        pressure_ax.legend()

    # Adjust layout for better readability
    plt.tight_layout()
    plt.show()


def probability_within_radius(df, radius_nm):
    """
    Calculate the probability that the real position is within a given radius (in nautical miles)
    of the predicted position, using both the mean (mu) and standard deviation (sigma)
    of the error distribution.

    Parameters:
    - mu: Mean of the distance errors (to account for bias).
    - sigma: Standard deviation of the distance errors (fitted from the error distribution).
    - distance_errors: The actual calculated distance errors between predicted and real positions.
    - radius_nm: The radius in nautical miles within which we want to calculate the probability.

    Returns:
    - A series of probabilities representing the likelihood that the real position is within
      the specified radius for each distance error.
      """

    def calculate_errors(df):
        """
        Calculate the errors between the predicted and real positions (lat, lon) and add them to the DataFrame.
        """

        df['lat_error'] = df['real_lat'] - df['pred_lat']
        df['lon_error'] = df['real_lon'] - df['pred_lon']

        df['distance_error'] = df.apply(
            lambda row: haversine(
                [row['real_lat'], row['real_lon']],
                [row['pred_lat'], row['pred_lon']]), axis=1)

        return df

    def fit_error_distribution(errors):
        """
        Fit a normal distribution to the distance errors to estimate the standard deviation (sigma).
        """
        mu, sigma = norm.fit(errors)
        return mu, sigma

    # Step 1: Calculate errors (using the DataFrame generated from the real vs predicted values)
    df = calculate_errors(df)
    distance_errors = df['distance_error']

    # Step 2: Fit a normal distribution to the distance errors (for sigma)
    mu, sigma = fit_error_distribution(distance_errors)

    # Calculate probability for each distance error to be within the specified radius
    df['probability_within_radius'] = norm.cdf(radius_nm - (distance_errors - mu), loc=0, scale=sigma)

    # df = df.drop(columns=['lat_error', 'lon_error', 'distance_error'])

    return df
