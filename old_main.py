import matplotlib.pyplot as plt
import astropy.units as units
import scipy.constants as const
import numpy as np

from typing import List, Tuple
from scipy.integrate import solve_ivp

def projectile(t, Y):
    y, v = Y
    return [v, -const.g]

def self_destruction():
    pass

def drange2(start, stop, step):
    numelements = int((stop-start)/float(step))
    for i in range(numelements+1):
            yield start + i*step

def main():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    initial_time = [0]
    max_time = 60 
    initial_conditions = [0, 400] # (m, m/s)

    solution = solve_ivp(
                         projectile, 
                         [0, max_time], 
                         initial_conditions, 
                         t_eval=np.linspace(0, max_time, 100)
                         )

    max_n = 5 
    for i in drange2(0, max_n, 0.5):
        print(f"Time: {i:.2f} s, Position: {solution.y[0][int(i*100/max_time)]:.2f} m, Velocity: {solution.y[1][int(i*100/max_time)]:.2f} m/s")

    plt.figure(figsize=(10, 5))
    plt.plot(solution.t, solution.y[0], label='Position')
    plt.xlabel('Time (s)')
    plt.ylabel('Height (m)')
    plt.show()

if __name__ == "__main__":
    main()
