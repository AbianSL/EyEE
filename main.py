from src.experiments import sweep_n_values, model_white_dwarfs, R_SUN, M_SUN, model_solar_remnant_wd
from src.plots import plot_lane_emden_solutions, extract_first_zeros, plot_white_dwarf_profiles

def main():
    max_xi = 200.0
    max_n = 5.0
    n_step = 0.5
    start_n = 0.0
    
    # Polytrope sweep
    trajectories = sweep_n_values(max_xi, start_n, max_n, n_step)
    plot_lane_emden_solutions(trajectories, 20)
    first_zeros = extract_first_zeros(trajectories)
    for n, xi_1 in first_zeros:
        print(f"n = {n:.1f} -> at xi_1 = {xi_1:.2f}")
    for polytrope in trajectories:
        dn, mn, rn, bn = polytrope.compute_constants()
        print(f"n = {polytrope.n:.1f} -> dn = {dn:.4f}, mn = {mn:.4f}, rn = {rn:.4f}, bn = {bn:.4f}")
    
    # White Dwarf modeling
    central_densities = [5e8, 1e9, 5e9]
    dwarf, k_wd, wd_results = model_white_dwarfs(max_xi, central_densities, mu_e=2.0)

    print("\n--- Non-relativistic White Dwarfs (n = 3/2, mu_e = 2) ---")
    for rho_c, radius, mass in wd_results:
        print(
            f"rho_c = {rho_c:.1e} kg/m^3 -> "
            f"R = {radius:.3e} m ({radius / R_SUN:.4f} R_sun), "
            f"M = {mass:.3e} kg ({mass / M_SUN:.3f} M_sun)"
        )

    plot_white_dwarf_profiles(dwarf, central_densities, k_wd)
    
    # Solar remnant white dwarf modeling
    solar_dwarf, k_wd, rho_c_solar, r_solar_wd = model_solar_remnant_wd(max_xi, target_mass_solar=0.5405, mu_e=2.0)
    print("\n--- Solar Remnant White Dwarf (M = 0.5405 M_sun, mu_e = 2) ---")
    print(
        f"rho_c = {rho_c_solar:.3e} kg/m^3, "
        f"R = {r_solar_wd:.3e} m ({r_solar_wd / R_SUN:.4f} R_sun)"
    )
    plot_white_dwarf_profiles(solar_dwarf, [rho_c_solar], k_wd, filename="solar_wd_remnant_profiles.pdf")


if __name__ == "__main__":
    main()
