import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class CometOrbit:
    def __init__(self, a, e, i, omega, Omega, M0, mu=1.327e11):
        """
        Initialize comet orbit with Keplerian elements

        Parameters:
        a: Semi-major axis (km)
        e: Eccentricity (0 < e < 1 for elliptical orbit)
        i: Inclination (degrees)
        omega: Argument of periapsis (degrees)
        Omega: Longitude of ascending node (degrees)
        M0: Mean anomaly at epoch (degrees)
        mu: Standard gravitational parameter (km³/s² for Sun)
        """
        self.a = a
        self.e = e
        self.i = np.radians(i)
        self.omega = np.radians(omega)
        self.Omega = np.radians(Omega)
        self.M0 = np.radians(M0)
        self.mu = mu
        self.n = np.sqrt(mu / a**3)  # Mean motion

    def solve_kepler_equation(self, M, tolerance=1e-10):
        """Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E"""
        E = M  # Initial guess
        for _ in range(50):  # Max iterations
            dE = (M - E + self.e * np.sin(E)) / (1 - self.e * np.cos(E))
            E += dE
            if abs(dE) < tolerance:
                break
        return E

    def get_position_velocity(self, t):
        """
        Calculate position and velocity at time t (seconds from epoch)
        Returns position (x,y,z) and velocity (vx,vy,vz) in km and km/s
        """
        # Mean anomaly at time t
        M = self.M0 + self.n * t

        # Solve for eccentric anomaly
        E = self.solve_kepler_equation(M)

        # True anomaly
        nu = 2 * np.arctan2(np.sqrt(1 + self.e) * np.sin(E/2),
                            np.sqrt(1 - self.e) * np.cos(E/2))

        # Distance from focus
        r = self.a * (1 - self.e * np.cos(E))

        # Position in orbital plane
        x_orb = r * np.cos(nu)
        y_orb = r * np.sin(nu)
        z_orb = 0

        # Velocity in orbital plane
        # Specific angular momentum
        h = np.sqrt(self.mu * self.a * (1 - self.e**2))
        vx_orb = -(self.mu / h) * np.sin(nu)
        vy_orb = (self.mu / h) * (self.e + np.cos(nu))
        vz_orb = 0

        # Rotation matrices for 3D transformation
        cos_Omega, sin_Omega = np.cos(self.Omega), np.sin(self.Omega)
        cos_omega, sin_omega = np.cos(self.omega), np.sin(self.omega)
        cos_i, sin_i = np.cos(self.i), np.sin(self.i)

        # Combined rotation matrix (Rz(Omega) * Rx(i) * Rz(omega))
        R11 = cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i
        R12 = -cos_Omega * sin_omega - sin_Omega * cos_omega * cos_i
        R13 = sin_Omega * sin_i

        R21 = sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i
        R22 = -sin_Omega * sin_omega + cos_Omega * cos_omega * cos_i
        R23 = -cos_Omega * sin_i

        R31 = sin_omega * sin_i
        R32 = cos_omega * sin_i
        R33 = cos_i

        # Transform to 3D space
        x = R11 * x_orb + R12 * y_orb + R13 * z_orb
        y = R21 * x_orb + R22 * y_orb + R23 * z_orb
        z = R31 * x_orb + R32 * y_orb + R33 * z_orb

        vx = R11 * vx_orb + R12 * vy_orb + R13 * vz_orb
        vy = R21 * vx_orb + R22 * vy_orb + R23 * vz_orb
        vz = R31 * vx_orb + R32 * vy_orb + R33 * vz_orb

        return np.array([x, y, z]), np.array([vx, vy, vz])

# Example: Halley's Comet-like orbit


def main():
    # Keplerian elements for a comet with highly eccentric orbit
    comet = CometOrbit(
        a=2.668e9,      # Semi-major axis (km) - roughly 17.8 AU
        e=0.967,        # High eccentricity (very elliptical)
        i=162.3,        # Inclination (degrees) - retrograde orbit
        omega=111.3,    # Argument of periapsis (degrees)
        Omega=58.4,     # Longitude of ascending node (degrees)
        M0=0.0          # Mean anomaly at epoch (degrees)
    )

    # Calculate orbital period
    T = 2 * np.pi * np.sqrt(comet.a**3 / comet.mu)
    print(f"Orbital period: {T / (365.25 * 24 * 3600):.1f} years")
    print(f"Perihelion distance: {comet.a * (1 - comet.e) / 1.496e8:.2f} AU")
    print(f"Aphelion distance: {comet.a * (1 + comet.e) / 1.496e8:.1f} AU")

    # Generate orbit points
    time_points = np.linspace(0, T, 1000)
    positions = []
    velocities = []

    for t in time_points:
        pos, vel = comet.get_position_velocity(t)
        positions.append(pos)
        velocities.append(vel)

    positions = np.array(positions)
    velocities = np.array(velocities)

    # Convert to AU for plotting
    positions_AU = positions / 1.496e8

    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot orbit
    ax.plot(positions_AU[:, 0], positions_AU[:, 1], positions_AU[:, 2],
            'b-', linewidth=2, label='Comet Orbit')

    # Mark perihelion and aphelion
    perihelion_idx = np.argmin(np.linalg.norm(positions, axis=1))
    aphelion_idx = np.argmax(np.linalg.norm(positions, axis=1))

    ax.scatter(*positions_AU[perihelion_idx], color='red', s=100,
               label='Perihelion', marker='o')
    ax.scatter(*positions_AU[aphelion_idx], color='orange', s=100,
               label='Aphelion', marker='s')

    # Mark Sun at origin
    ax.scatter(0, 0, 0, color='yellow', s=200, label='Sun', marker='*')

    # Add planet orbits for reference (circular approximations)
    theta = np.linspace(0, 2*np.pi, 100)

    # Earth orbit
    earth_r = 1.0  # 1 AU
    ax.plot(earth_r * np.cos(theta), earth_r * np.sin(theta),
            np.zeros_like(theta), 'g--', alpha=0.5, label='Earth Orbit')

    # Jupiter orbit
    jupiter_r = 5.2  # 5.2 AU
    ax.plot(jupiter_r * np.cos(theta), jupiter_r * np.sin(theta),
            np.zeros_like(theta), 'brown', alpha=0.3, linestyle='--', label='Jupiter Orbit')

    ax.set_xlabel('X (AU)')
    ax.set_ylabel('Y (AU)')
    ax.set_zlabel('Z (AU)')
    ax.set_title('Comet Orbit in 3D Space')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set equal aspect ratio
    max_range = np.max(np.abs(positions_AU))
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range/2, max_range/2)

    plt.tight_layout()
    plt.show()

    # Plot velocity magnitude over time
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    vel_mag = np.linalg.norm(velocities, axis=1)
    time_years = time_points / (365.25 * 24 * 3600)

    ax2.plot(time_years, vel_mag, 'r-', linewidth=2)
    ax2.set_xlabel('Time (years)')
    ax2.set_ylabel('Velocity (km/s)')
    ax2.set_title('Comet Velocity Magnitude Over Time')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
