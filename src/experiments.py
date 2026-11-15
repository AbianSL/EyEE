import numpy as np
from scipy import constants as const
from astropy import constants as astro_const
from src.lame_emdem import Polytrope

M_SUN = astro_const.M_sun.value  # kg
R_SUN = astro_const.R_sun.value  # m

def sweep_n_values(max_xi, start_n, max_n, n_step):
    """
    Sweep through a range of polytropic indices n and simulate the Lane-Emden equation for each value.
    """
    n_values = np.arange(start_n, max_n + n_step / 2.0, n_step)
    time_grid = np.linspace(0.0, max_xi, 500)

    trajectories = []
    for n in n_values:
        polytrope = Polytrope(n, max_xi)
        polytrope.simulate_single(time_grid)
        polytrope.compute_constants()
        trajectories.append(polytrope)

    return trajectories

def k_non_relativistic_degenerate(mu_e=2.0):
    """
    Compute the polytropic constant K for a non-relativistic degenerate electron gas.
    """
    k_prime_1 = ((3.0 * np.pi**2) ** (2.0 / 3.0) * const.hbar**2) / (5.0 * const.m_e * const.m_u ** (5.0 / 3.0))
    return k_prime_1 * (mu_e ** (-5.0 / 3.0))

def model_white_dwarfs(max_xi, central_densities, mu_e=2.0):
    """
    Simulate a non-relativistic white dwarf (n = 3/2) and compute physical radii and masses
    for a list of central densities.
    """
    dwarf = Polytrope(3.0 / 2.0, max_xi)
    xi_grid = np.concatenate([
        np.linspace(0.0, 3.64, 5000, endpoint=False),
        np.linspace(3.64, 3.66, 20000)
    ])
    dwarf.simulate_single(xi_grid)
    dwarf.compute_constants()

    k_wd = k_non_relativistic_degenerate(mu_e=mu_e)
    results = []
    for rho_c in central_densities:
        radius, mass = dwarf.compute_physical_star(rho_c, k_wd)
        results.append((rho_c, radius, mass))

    return dwarf, k_wd, results

def model_solar_remnant_wd(max_xi, target_mass_solar=0.5405, mu_e=2.0):
    """
    Model a white dwarf remnant of given mass in solar units (n = 3/2)
    and return the Polytrope instance, K, central density rho_c [kg/m^3], and radius R [m].
    """
    dwarf = Polytrope(3.0 / 2.0, max_xi)
    xi_grid = np.concatenate([
        np.linspace(0.0, 3.64, 5000, endpoint=False),
        np.linspace(3.64, 3.66, 20000)
    ])
    dwarf.simulate_single(xi_grid)
    dwarf.compute_constants()

    k_wd = k_non_relativistic_degenerate(mu_e=mu_e)
    target_mass_kg = target_mass_solar * M_SUN
    rho_c, radius = dwarf.compute_central_density_from_mass(target_mass_kg, k_wd)

    return dwarf, k_wd, rho_c, radius
