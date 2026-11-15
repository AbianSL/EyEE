import numpy as np
from scipy.integrate import solve_ivp
from scipy import constants as const

class Polytrope:
    """
    Class to simulate the Lane-Emden equation for a given polytropic index n
    """
    
    def __init__(self, n, max_xi) -> None:
        self.n = n
        self.max_xi = max_xi
        self.xi = None    # Xi values (dimensionless radius)
        self.theta = None  # Theta values (dimensionless density)
        self.dtheta_dxi = None
        self.xi_1 = None
        self.dtheta_dxi_1 = None

        self.dn = None
        self.mn = None
        self.rn = None
        self.bn = None

    def lane_emdem(self, xi, state) -> list:
        """
        Lane-Emden equation for a given polytropic index n.
        It is defined as:
            2/xi * dtheta/dxi + d^2theta/dxi^2 + theta^n = 0
        """
        theta, dtheta_dxi = state
        theta = max(theta, 0.0)
        d2theta_dxi2 = -(1.0 / 3.0) # when xi = 0, the second derivative is defined as -1/3 for the Lane-Emden equation 
        if xi != 0:
            d2theta_dxi2 = -(theta ** self.n) - (2.0 / xi) * dtheta_dxi
        return [dtheta_dxi, d2theta_dxi2]

    @staticmethod
    def self_destruction(xi, state) -> float:
        """
        Event funciton to detect when the theta becomes negative.
        """
        theta, dtheta_dxi = state
        return theta

    def compute_alpha(self, rho_c, k_poly):
        """
        Compute the radial scaling factor alpha [m] for a given central density rho_c and polytropic constant K.
        """
        return np.sqrt(((self.n + 1.0) * k_poly) / (4.0 * np.pi * const.G * (rho_c ** ((self.n - 1.0) / self.n))))

    def compute_physical_star(self, rho_c, k_poly):
        """
        Compute physical radius R [m] and mass M [kg] for any polytrope with n < 5.
        """
        if self.n >= 5.0:
            raise ValueError("Physical radius and mass are not finite for n >= 5.")

        if self.rn is None or self.mn is None:
            self.compute_constants()

        alpha = self.compute_alpha(rho_c, k_poly)
        radius = alpha * self.rn
        mass = 4.0 * np.pi * (alpha ** 3) * rho_c * self.mn

        return radius, mass

    def compute_radial_profiles(self, rho_c, k_poly):
        """
        Compute radial profiles r [m], rho(r) [kg/m^3], and enclosed mass m(r) [kg].
        """
        if self.xi is None or self.theta is None or self.dtheta_dxi is None:
            raise ValueError("Run simulate_single before computing radial profiles.")

        alpha = self.compute_alpha(rho_c, k_poly)
        r = alpha * self.xi
        theta_clipped = np.maximum(self.theta, 0.0)
        rho = rho_c * (theta_clipped ** self.n)
        m_enclosed = -4.0 * np.pi * (alpha ** 3) * rho_c * (self.xi ** 2) * self.dtheta_dxi

        return r, rho, m_enclosed
    
    def simulate_single(self, time_grid) -> "Polytrope":
        """
        Simulate the Lane-Emden equation for a given polytropic index n and return the time, height, and impact time.
        """
        initial_state = [1.0, 0.0] # theta, dtheta/dxi

        self.self_destruction.terminal = True
        self.self_destruction.direction = -1

        solution = solve_ivp(
            self.lane_emdem,
            [0.0, self.max_xi],
            initial_state,
            t_eval=time_grid,
            events=self.self_destruction
        )

        self.xi = solution.t
        self.theta = solution.y[0]
        self.dtheta_dxi = solution.y[1]
        self.xi_1 = None

        if solution.t_events[0].size > 0:
            self.xi_1 = solution.t_events[0][0]
            impact_height = solution.y_events[0][0][0]
            self.dtheta_dxi_1 = solution.y_events[0][0][1]

            self.xi = np.append(self.xi, self.xi_1)
            self.theta = np.append(self.theta, impact_height)
            self.dtheta_dxi = np.append(self.dtheta_dxi, self.dtheta_dxi_1)

        return self
    
    def compute_constants(self):
        """
        Compute Dn, Mn, Rn, and Bn for this polytropic index n.
        Handles the special case n = 5 (infinite radius, finite mass).
        """
        if np.isclose(self.n, 5.0):
            self.rn = np.inf
            self.mn = np.sqrt(3.0)
            self.dn = np.inf
            self.bn = (1.0 / 6.0) * (self.mn ** (-2.0 / 3.0))
            return self.dn, self.mn, self.rn, self.bn

        if self.n > 5.0:
            raise ValueError(f"Polytropes with n > 5 (got n = {self.n}) have infinite radius and infinite mass.")

        if self.xi_1 is None or self.dtheta_dxi_1 is None:
            raise ValueError(f"Surface xi_1 not reached for n = {self.n}. Increase max_xi.")

        self.rn = self.xi_1
        self.mn = -(self.xi_1 ** 2) * self.dtheta_dxi_1
        self.dn = (self.rn ** 3 / 3.0) * (1.0 / self.mn)
        self.bn = (1.0 / (self.n + 1.0)) * (self.mn ** (-2.0 / 3.0))

        return self.dn, self.mn, self.rn, self.bn
    
    def compute_central_density_from_mass(self, mass, k_poly):
        """
        Compute central density rho_c [kg/m^3] and physical radius R [m]
        for a given total mass M [kg] and polytropic constant K (valid for n < 5, n != 3).
        """
        if self.n >= 5.0 or np.isclose(self.n, 3.0):
            raise ValueError(f"Mass does not uniquely determine central density for n = {self.n}.")

        if self.rn is None or self.mn is None:
            self.compute_constants()

        scale_factor = ((self.n + 1.0) * k_poly) / (4.0 * np.pi * const.G)
        base = mass / (4.0 * np.pi * self.mn * (scale_factor ** 1.5))
        exponent = (2.0 * self.n) / (3.0 - self.n)
        rho_c = base ** exponent

        alpha = self.compute_alpha(rho_c, k_poly)
        radius = alpha * self.rn

        return rho_c, radius
    
    def get_trajectory(self):
        """
        Return the core trajectory data (n, xi, theta, xi_1).
        """
        return self.n, self.xi, self.theta, self.xi_1

    def get_constants(slef) -> tuple:
        """
        Return the constants dn, mn, rn, and bn for the polytrope.
        """
        return self.dn, self.mn, self.rn, self.bn
