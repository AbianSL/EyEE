import numpy as np
import scipy.constants as const
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def projectile_ode(time, state, param_n):
    height, velocity = state
    return [velocity, -const.g]


def self_destruction(time, state, param_n):
    height, velocity = state
    return height


self_destruction.terminal = True
self_destruction.direction = -1


def main():
    max_time = 100.0
    max_n = 5.0
    n_step = 0.5
    impact_records = []

    plt.figure(figsize=(10, 5))

    n_values = np.arange(0.0, max_n + n_step, n_step)
    time_grid = np.linspace(0.0, max_time, 500)

    for n in n_values:
        initial_velocity = 100.0 + n * 60.0
        initial_state = [0.0, initial_velocity]

        solution = solve_ivp(
            projectile_ode,
            [0.0, max_time],
            initial_state,
            t_eval=time_grid,
            events=self_destruction,
            args=(n,)
        )

        plot_time = solution.t
        plot_height = solution.y[0]

        if solution.t_events[0].size > 0:
            impact_time = solution.t_events[0][0]
            impact_height = solution.y_events[0][0][0]

            impact_records.append((n, impact_time))
            plot_time = np.append(plot_time, impact_time)
            plot_height = np.append(plot_height, impact_height)

        plt.plot(plot_time, plot_height, label=f"n = {n:.1f}")

    for n, impact_time in impact_records:
        print(f"n = {n:.1f} -> Impact time: {impact_time:.2f} s")

    plt.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Height (m)")
    plt.title("Projectile Motion with Ground Impact Event")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
