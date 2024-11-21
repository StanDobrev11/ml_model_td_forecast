import numpy as np


# Mercator latitudes
def mercator_latitude(lat):
    return np.log(np.tan(np.pi / 4 + lat / 2))


def mercator_conversion(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)

    delta_phi = mercator_latitude(lat2) - mercator_latitude(lat1)

    # Difference in longitudes
    delta_lambda = lon2 - lon1

    return delta_phi, delta_lambda


def rumbline_distance(start_point, end_point):
    """
    Calculates rumbline distance between 2 points located on the earth surface

    :param start_point: lat, lon of starting position
    :param end_point: lat, lon of ending position
    :return: distance in NM
    """
    lat1, lon1 = start_point
    lat2, lon2 = end_point
    delta_phi, delta_lambda = mercator_conversion(lat1, lon1, lat2, lon2)

    # Calculate distance using the Mercator Sailing formula
    return np.sqrt((delta_lambda * np.cos(np.radians(lat1))) ** 2 + delta_phi ** 2) * 3440.065


def bearing_to_waypoint(start_point, end_point):
    lat1, lon1 = start_point
    lat2, lon2 = end_point
    delta_phi, delta_lambda = mercator_conversion(lat1, lon1, lat2, lon2)

    # Calculate the bearing using atan2
    bearing_rad = np.arctan2(delta_lambda, delta_phi)

    # Convert from radians to degrees
    bearing_deg = np.degrees(bearing_rad)

    # Normalize to 0-360 degrees
    bearing_deg = (bearing_deg + 360) % 360

    return bearing_deg


def mercator_sailing_future_position(lat, lon, speed, bearing, time_interval):
    """
    Calculate future position given current lat, lon, speed (knots), bearing, and time interval (hours).

    Parameters:
    - lat (float): Current latitude in degrees
    - lon (float): Current longitude in degrees
    - speed (float): Speed in knots (nautical miles per hour)
    - bearing (float): Bearing in degrees (from north)
    - time_interval (float): Time interval in hours

    Returns:
    - new_lat (float): New latitude in degrees
    - new_lon (float): New longitude in degrees
    """
    # Convert degrees to radians
    lat = np.radians(lat)
    lon = np.radians(lon)
    bearing = np.radians(bearing)

    # Earth radius in nautical miles
    R = 3440.065

    # Distance traveled in nautical miles
    distance = speed * time_interval

    # Delta latitude (in radians)
    delta_lat = distance * np.cos(bearing) / R

    # Update latitude
    new_lat = lat + delta_lat

    # Delta longitude (in radians)
    if np.cos(new_lat) != 0:
        delta_lon = (distance * np.sin(bearing)) / (R * np.cos(new_lat))
    else:
        delta_lon = 0

    # Update longitude
    new_lon = lon + delta_lon

    # Convert radians back to degrees
    new_lat = np.degrees(new_lat)
    new_lon = np.degrees(new_lon)

    return np.array([new_lat, new_lon])

def great_circle_future_position(lat, lon, speed, bearing, time_interval):
    """
    Calculate future position given current lat, lon, speed (knots), bearing, and time interval (hours)
    using great-circle navigation.

    Parameters:
    - lat (float): Current latitude in degrees
    - lon (float): Current longitude in degrees
    - speed (float): Speed in knots (nautical miles per hour)
    - bearing (float): Bearing in degrees (from north)
    - time_interval (float): Time interval in hours

    Returns:
    - new_lat (float): New latitude in degrees
    - new_lon (float): New longitude in degrees
    """
    # Convert degrees to radians
    lat = np.radians(lat)
    lon = np.radians(lon)
    bearing = np.radians(bearing)

    # Earth radius in nautical miles
    R = 3440.065

    # Distance traveled in nautical miles
    distance = speed * time_interval

    # Update latitude using great circle navigation
    new_lat = np.arcsin(np.sin(lat) * np.cos(distance / R) +
                        np.cos(lat) * np.sin(distance / R) * np.cos(bearing))

    # Update longitude using great circle navigation
    new_lon = lon + np.arctan2(np.sin(bearing) * np.sin(distance / R) * np.cos(lat),
                               np.cos(distance / R) - np.sin(lat) * np.sin(new_lat))

    # Convert radians back to degrees
    new_lat = np.degrees(new_lat)
    new_lon = np.degrees(new_lon)

    return np.array([new_lat, new_lon])


def plane_sailing_course_speed(start_point, end_point, time_interval=6):
    """
    Calculates course and distance between 2 points using plane sailing approximation.

    :param start_point: tuple (lat1, lon1) in degrees
    :param end_point: tuple (lat2, lon2) in degrees
    :return: course (degrees), distance (nautical miles)
    """
    # Convert lat/lon to radians
    start_point = np.radians(np.array(start_point))
    end_point = np.radians(np.array(end_point))

    # Calculate delta lat (in radians)
    delta_lat = end_point[0] - start_point[0]

    # Calculate delta lon (in radians)
    delta_lon = end_point[1] - start_point[1]

    # Calculate the mean latitude (in radians)
    mean_lat = np.mean([start_point[0], end_point[0]])

    # Calculate the departure (in radians)
    dep = delta_lon * np.cos(mean_lat)

    # Calculate the course (in degrees)
    course = round(np.degrees(np.arctan2(dep, delta_lat)))
    if course < 0:
        course = 360 + course

    # Calculate the distance (in nautical miles)
    distance = np.sqrt(delta_lat ** 2 + dep ** 2) * 60 * 180 / np.pi  # Converting from radians to nautical miles
    speed = np.round(distance / time_interval, 1)

    return np.array([course, speed])


def plane_sailing_next_position(start_point, course, speed, time_interval=6):
    """
    Calculates the next position based on a starting point, course, and distance using plane sailing approximation.

    :param start_point: tuple (lat, lon) in degrees
    :param course: course (bearing) in degrees
    :param distance: distance in nautical miles
    :return: tuple (new_lat, new_lon) in degrees
    """
    # Convert lat/lon and course to radians
    lat1, lon1 = np.radians(start_point)
    course = np.radians(course)

    # Calculate distance
    distance = speed * time_interval

    # Convert distance in nautical miles to degrees of latitude/longitude
    distance_rad = np.radians(distance / 60)  # Distance in radians (1 degree = 60 NM)

    # Calculate the change in latitude (delta_lat)
    delta_lat = distance_rad * np.cos(course)

    # Calculate the new latitude
    new_lat = lat1 + delta_lat

    # Calculate the mean latitude (average of the original and new latitude)
    mean_lat = (lat1 + new_lat) / 2

    # Calculate the change in longitude (delta_lon)
    if np.cos(mean_lat) != 0:
        delta_lon = distance_rad * np.sin(course) / np.cos(mean_lat)
    else:
        delta_lon = 0

    # Calculate the new longitude
    new_lon = lon1 + delta_lon

    # Convert the new latitude and longitude back to degrees
    new_lat = np.round(np.degrees(new_lat), 1)
    new_lon = np.round(np.degrees(new_lon), 1)

    return np.array([new_lat, new_lon])


def haversine(start_point, end_point):
    """
    Calculate the great-circle distance between two points on the Earth (specified in decimal degrees).
    """
    # Convert decimal degrees to radians
    lat1, lon1 = map(np.radians, start_point)
    lat2, lon2 = map(np.radians, end_point)

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 3440.065  # Radius of Earth in nautical miles
    return c * r

if __name__ == '__main__':
    lat1 = 18.6
    lon1 = 139.3
    lat2 = 19
    lon2 = 140
    print(rumbline_distance([lat1, lon1], [lat2, lon2]))
    # print(bearing_to_waypoint([lat1, lon1], [lat2, lon2]))
    # print(mercator_sailing_future_position(lat1, lon1, 46.768623, 48.146357, 6))
    # print(great_circle_future_position(lat1, lon1, 46.768623, 48.146357, 6))
    print(plane_sailing_course_speed([lat1, lon1], [lat2, lon2]))
    print(plane_sailing_next_position([lat1, lon1], 274, 14.8))
