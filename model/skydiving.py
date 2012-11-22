import math
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

g = 9.81 # m / s2
initial_height = 39045.0 # m
parachute_release_start_time = 4.0 * 60.0 + 16.0 # s
parachute_release_end_time = 4.0 * 60.0 + 20.0 # s
end_time = 9.0 * 60.0 # s
total_mass = 118 #kg
c_d_man = 1.15 # unitless
c_d_parachute = 1.5 # unitless
cross_area_man = 0.73 # m2
area_parachute = 25.0 # m2
# Assuming parachute is a half-sphere
cross_area_parachute = 4.0 * area_parachute / math.pi ** 2
M = 0.0289644 # kg/mol, air molar mass
R = 8.31447 # J/(mol * K), universal gas constant

# Changes of air temperature depending on altitude, (m, K)
temp_alt = ((0.0, 15.0 + 273), (11.0e3, -56.5 + 273), (20.1e3, -56.5 + 273), (32.2e3, -44.5 + 273), (47.3e3, -2.5 + 273))

# Changes of air pressure depending on altitude, (m, Pa)
pres_alt = ((0.0, 101325.0), (11.0e3, 22630.5), (20.1e3, 5474.27), (32.2e3, 867.849), (47.3e3, 110.874))

h = 0.1 # s
num_steps = int(end_time / h)
times = h * np.array(range(num_steps + 1))

def air_temperature_by_altitude(altitude):
    ''' Returns the air temperature at given altitude '''
    return get_approximation_value(altitude, temp_alt)

def air_pressure_by_altitude(altitude):
    ''' Returns the air pressure at given altitude '''
    return get_approximation_value(altitude, pres_alt)

def get_approximation_value(point, key_points):
    ''' Returns the approximation value for the given point. Approximation represented by list of key points and their values '''
    if point < key_points[0][0] or point > key_points[-1][0]:
        message = 'Point {0} does not fall into the acceptable range between {1} and {2}'.format(point, key_points[0][0], key_points[-1][0])
        raise ValueError(message)

    for left, right in zip(key_points[:-1], key_points[1:]):
        if left[0] <= point <= right[0]:
            return left[1] + ((point - left[0])/(right[0] - left[0])) * (right[1] - left[1])

def air_density(temperature, pressure):
    ''' Given pressure and temperature, calculates the air density '''
    return pressure * M / (temperature * R)

def air_density_by_altitude(altitude):
    ''' Returns the air density at given altitude '''
    temperature = air_temperature_by_altitude(altitude)
    pressure = air_pressure_by_altitude(altitude)
    return air_density(temperature, pressure)

def get_environment_conditions(altitude):
    return air_density_by_altitude(altitude), air_temperature_by_altitude(altitude), air_pressure_by_altitude(altitude)
    
def skydiving():
    x = np.zeros(num_steps + 1)
    v = np.zeros(num_steps + 1)
    air_density_array = np.zeros(num_steps + 1)
    air_temperature_array = np.zeros(num_steps + 1)
    air_pressure_array = np.zeros(num_steps + 1)

    c_d = c_d_man
    cross_area = cross_area_man

    x[0] = initial_height
    v[0] = 0.0
    air_density_array[0], air_temperature_array[0], air_pressure_array[0] = get_environment_conditions(x[0])

    for step in xrange(num_steps):
        # Air density at current altitude
        air_dens = air_density_by_altitude(x[step])

        # Air resistance at current altitude
        air_resistance = 0.5 * air_dens * c_d * cross_area * v[step] ** 2

        x[step + 1] = x[step] + v[step] * h
        v[step + 1] = v[step] + (air_resistance / total_mass - g) * h

        current_time = h * step
        if parachute_release_start_time <= current_time <= parachute_release_end_time:
            parachute_opening_fraction = (current_time - parachute_release_start_time) / (parachute_release_end_time - parachute_release_start_time)

            c_d = c_d_man + (c_d_parachute - c_d_man) * parachute_opening_fraction
            cross_area = cross_area_man + (cross_area_parachute - cross_area_man) * parachute_opening_fraction

        air_density_array[step + 1], air_temperature_array[step + 1], air_pressure_array[step + 1] = get_environment_conditions(x[step + 1])
    
    return x, v, air_density_array, air_temperature_array, air_pressure_array

def plot_graphs():
    x, v, d, t, p = skydiving()

    plt.figure(1)

    plt.subplot(511)
    plt.plot(times, x)
    plt.xlabel("Time, s")
    plt.ylabel("Altitude, m")

    plt.subplot(512)
    plt.plot(times, v)
    plt.xlabel('Time, s')
    plt.ylabel("Velocity, m/s")

    plt.subplot(513)
    plt.plot(times, d)
    plt.xlabel('Time, s')
    plt.ylabel("Air density, kg/m3")
    
    plt.subplot(514)
    plt.plot(times, t)
    plt.xlabel('Time, s')
    plt.ylabel("Air temperature, K")
    
    plt.subplot(515)
    plt.plot(times, p)
    plt.xlabel('Time, s')
    plt.ylabel("Air pressure, m/s")

    plt.show()

plot_graphs()