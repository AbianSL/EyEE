import os
import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as astro_const


M_SUN = astro_const.M_sun.value
R_SUN = astro_const.R_sun.value


def _configure_astro_style():
    """
    Configure matplotlib typography and font sizes to match astrophysics publication standards.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })


def _format_scientific_latex(value, decimals=2):
    """
    Format a float into proper LaTeX scientific notation (e.g., 1.19e9 -> '1.19 \\times 10^{9}').
    """
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / (10.0 ** exponent)
    if np.isclose(mantissa, 1.0):
        return rf"10^{{{exponent}}}"
    if np.isclose(mantissa, round(mantissa)):
        return rf"{int(round(mantissa))} \times 10^{{{exponent}}}"
    return rf"{mantissa:.{decimals}f} \times 10^{{{exponent}}}"

def extract_first_zeros(trajectories):
    first_zeros = []
    for polytrope in trajectories:
        n, _, _, xi_1 = polytrope.get_trajectory()
        if xi_1 is not None:
            first_zeros.append((n, xi_1))
    return first_zeros

def plot_lane_emden_solutions(trajectories, max_xi=20.0, output_dir="figures"):
    _configure_astro_style()
    os.makedirs(output_dir, exist_ok=True)

    linestyles = ["-", "--", "-.", ":"]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(trajectories)))

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, polytrope in enumerate(trajectories):
        n, xi, theta, _ = polytrope.get_trajectory()
        style = linestyles[i % len(linestyles)]
        ax.plot(xi, theta, label=rf"$n = {n:.1f}$", color=colors[i], linestyle=style, linewidth=2.0)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlim(0, max_xi)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$\theta(\xi)$")
    ax.legend(ncol=2, frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "lane_emden_solutions.pdf"), format="pdf", bbox_inches="tight")
    plt.show()

def plot_white_dwarf_profiles(dwarf, central_densities, k_poly, output_dir="figures", filename="white_dwarf_profiles.pdf"):
    """
    Plot log(rho / kg m^-3) and enclosed mass m / M_sun as a function of r / R_sun.
    """
    _configure_astro_style()
    os.makedirs(output_dir, exist_ok=True)

    linestyles = ["-", "--", "-."]
    colors = plt.cm.viridis([0.15, 0.50, 0.85]) if len(central_densities) > 1 else ["#1b4f72"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    for i, rho_c in enumerate(central_densities):
        r, rho, m_enc = dwarf.compute_radial_profiles(rho_c, k_poly)
        r_solar = r / R_SUN
        m_solar = m_enc / M_SUN
        style = linestyles[i % len(linestyles)]
        rho_latex = _format_scientific_latex(rho_c)
        label = rf"$\rho_{{\rm c}} = {rho_latex}\,\mathrm{{kg\,m^{{-3}}}}$"

        mask = rho > 0.0
        ax1.plot(r_solar[mask], np.log10(rho[mask]), color=colors[i], linestyle=style, linewidth=2.0, label=label)
        ax2.plot(r_solar, m_solar, color=colors[i], linestyle=style, linewidth=2.0, label=label)

    for ax in (ax1, ax2):
        ax.set_xlim(-0.001, 0.0195)
        ax.set_xticks([0.0, 0.005, 0.010, 0.015])
        ax.set_xlabel(r"$r / R_\odot$")
        ax.minorticks_on()
        ax.tick_params(which="minor", direction="in", top=True, right=True)

    ax1.set_ylim(4.0, 10.0)
    ax1.set_ylabel(r"$\log(\rho / \mathrm{kg\,m^{-3}})$")
    ax1.legend(frameon=False, loc="lower left")

    ax2.set_ylim(-0.05, 1.2)
    ax2.set_ylabel(r"$m / M_\odot$")

    fig.tight_layout(w_pad=2.5)
    fig.savefig(os.path.join(output_dir, filename), format="pdf", bbox_inches="tight")
    plt.show()
