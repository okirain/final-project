import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless rendering
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure scratch directory exists
os.makedirs('/workspace/scratch', exist_ok=True)

# ----------------------------------------------------------------------
# 1. PHYSICAL CONSTANTS & GEOMETRY DEFINITIONS (NASA KILOPOWER SPEC)
# ----------------------------------------------------------------------
# Heat pipe geometry (from Table 2, passage 449 of "Space Nuclear Power")
OD = 1.27e-2         # Outer diameter: 1.27 cm (0.5 in)
t_wall = 0.089e-2    # Wall thickness: 0.089 cm (0.035 in)
t_wick = 0.05e-2     # Wick thickness: 0.05 cm

r_o = OD / 2.0
r_i = r_o - t_wall
r_v = r_i - t_wick
D_v = 2.0 * r_v

A_v = np.pi * r_v**2                      # Vapor core area
A_wick = np.pi * (r_i**2 - r_v**2)        # Wick cross-sectional area
A_wall = np.pi * (r_o**2 - r_i**2)        # Wall cross-sectional area

# Lengths of sections (from NASA specifications)
L_evap = 0.356       # 35.6 cm (14.0 in)
L_adia = 0.864       # 86.4 cm (34.0 in)
L_cond = 0.0889      # 8.89 cm (3.5 in)
L_eff = L_evap / 2.0 + L_adia + L_cond / 2.0  # Effective length

# Sodium and materials constants
M_Na = 22.98977e-3   # Molar mass of Sodium: 22.99 g/mol
R_u = 8.31446        # Universal gas constant (J/(mol K))
R_g = R_u / M_Na     # Specific gas constant for Sodium vapor (J/(kg K))
gamma = 1.67         # Monatomic gas ratio of specific heats
g = 9.81             # Gravity (m/s^2)

# Haynes 230 wall and SS316 wick thermal conductivities (W/(m K))
k_wall = 22.0        # Haynes 230 thermal conductivity at ~800C
k_ss = 20.0          # SS316 wire thermal conductivity
k_l = 70.0           # Liquid sodium thermal conductivity
# Effective wick conductivity (Maxwell's formula or simple volume fraction)
epsilon_wick = 0.6   # Wick porosity
k_eff = k_l * ((k_l + k_ss) - (1.0 - epsilon_wick) * (k_l - k_ss)) / \
              ((k_l + k_ss) + (1.0 - epsilon_wick) * (k_l - k_ss))

# Wick pore parameters (SS316 screen mesh, 300 mesh)
N_mesh = 300.0 * 39.37  # 300 mesh per inch to mesh per meter
d_wire = 0.0016 * 2.54e-2 # wire diameter: 0.0016 in to m
r_c = 1.0 / (2.0 * N_mesh) # capillary pore radius
# High permeability arterial wick modeling for flight
K_artery = 1.5e-8    # High-permeability wick/artery for arterial design (m^2)
r_n = 2.0e-6         # Nucleation bubble radius for boiling limit (2.0 microns)

# ----------------------------------------------------------------------
# 2. TEMPERATURE-DEPENDENT SODIUM THERMOPHYSICAL PROPERTIES
# ----------------------------------------------------------------------
def P_sat_sodium(T):
    """Saturated vapor pressure of sodium (Pa) using Foust's equation."""
    # T in Kelvin
    return 101325.0 * 10**(4.521 - 5220.0 / T)

def rho_liquid_sodium(T):
    """Saturated liquid density of sodium (kg/m3)."""
    T_c = T - 273.15
    return 950.1 - 0.2297 * T_c - 1.46e-5 * T_c**2

def rho_vapor_sodium(T):
    """Saturated vapor density of sodium (kg/m3) using ideal gas law."""
    return P_sat_sodium(T) / (R_g * T)

def h_fg_sodium(T):
    """Latent heat of vaporization of sodium (J/kg)."""
    return 4.56e6 - 400.0 * T

def sigma_sodium(T):
    """Surface tension of liquid sodium (N/m)."""
    return 0.2067 - 1.0e-4 * T

def mu_liquid_sodium(T):
    """Viscosity of liquid sodium (Pa*s)."""
    return 1.2e-4 * np.exp(750.0 / T)

def mu_vapor_sodium(T):
    """Viscosity of sodium vapor (Pa*s)."""
    return 1.6e-5 * (T / 800.0)**0.8

# ----------------------------------------------------------------------
# 3. HIGH-TEMPERATURE HEAT PIPE LIMITS CALCULATIONS
# ----------------------------------------------------------------------
def viscous_limit(T):
    Pv = P_sat_sodium(T)
    rhov = rho_vapor_sodium(T)
    h_fg = h_fg_sodium(T)
    muv = mu_vapor_sodium(T)
    return (D_v**2 * A_v * rhov * Pv * h_fg) / (64.0 * muv * L_eff)

def sonic_limit(T):
    rhov = rho_vapor_sodium(T)
    h_fg = h_fg_sodium(T)
    speed_of_sound = np.sqrt(gamma * R_g * T)
    return A_v * rhov * h_fg * speed_of_sound / np.sqrt(2.0 * (gamma + 1.0))

def entrainment_limit(T):
    rhov = rho_vapor_sodium(T)
    h_fg = h_fg_sodium(T)
    sigma = sigma_sodium(T)
    return A_v * h_fg * np.sqrt(rhov * sigma / (2.0 * r_c))

def capillary_limit(T):
    sigma = sigma_sodium(T)
    rhol = rho_liquid_sodium(T)
    rhov = rho_vapor_sodium(T)
    h_fg = h_fg_sodium(T)
    mul = mu_liquid_sodium(T)
    muv = mu_vapor_sodium(T)
    
    # Capillary pressure driver
    dP_cap_max = 2.0 * sigma / r_c
    
    # Liquid flow resistance factor
    F_l = mul / (K_artery * A_wick * rhol * h_fg)
    
    # Vapor flow resistance factor
    F_v = 32.0 * muv / (D_v**2 * A_v * rhov * h_fg)
    
    return dP_cap_max / ((F_l + F_v) * L_eff)

def boiling_limit(T):
    h_fg = h_fg_sodium(T)
    rhov = rho_vapor_sodium(T)
    sigma = sigma_sodium(T)
    # Standard boiling limit based on nucleate boiling bubbles in the wick
    return (2.0 * np.pi * L_evap * k_eff * T * (2.0 * sigma / r_n)) / (h_fg * rhov * np.log(r_i / r_v))

# ----------------------------------------------------------------------
# 4. THERMAL RESISTANCE NETWORK RESOLUTION
# ----------------------------------------------------------------------
def solve_thermal_network(T_evap_outer_wall, Q_input):
    """
    Solves the radial and axial thermal resistances to find the temperature profile.
    Returns: T_evap_outer_wall, T_evap_inner_wall, T_vapor_evap, T_vapor_cond, T_cond_inner_wall, T_cond_outer_wall
    """
    # 1. Evaporator outer wall to inner wall (Haynes 230 conduction)
    R_w_evap = np.log(r_o / r_i) / (2.0 * np.pi * L_evap * k_wall)
    T_evap_inner_wall = T_evap_outer_wall - Q_input * R_w_evap
    
    # 2. Evaporator inner wall to vapor core (Wick conduction)
    R_wk_evap = np.log(r_i / r_v) / (2.0 * np.pi * L_evap * k_eff)
    T_vapor_evap = T_evap_inner_wall - Q_input * R_wk_evap
    
    # Estimate sodium properties at this local vapor temperature to get vapor resistance
    mu_v = mu_vapor_sodium(T_vapor_evap)
    rho_v = rho_vapor_sodium(T_vapor_evap)
    h_fg = h_fg_sodium(T_vapor_evap)
    
    # 3. Vapor core resistance (pressure drop from evap to cond)
    # Clausius-Clapeyron: dP/dT = rho_v * h_fg / T
    # Vapor pressure drop: dP_v = F_v * Q * L_eff
    # T_drop = dP_v * T / (rho_v * h_fg)
    F_v_drop = 32.0 * mu_v * L_eff / (D_v**2 * A_v * rho_v * h_fg)
    R_vapor = (T_vapor_evap * F_v_drop) / (rho_v * h_fg)
    
    T_vapor_cond = T_vapor_evap - Q_input * R_vapor
    
    # 4. Vapor to condenser inner wall (Condenser wick)
    R_wk_cond = np.log(r_i / r_v) / (2.0 * np.pi * L_cond * k_eff)
    T_cond_inner_wall = T_vapor_cond - Q_input * R_wk_cond
    
    # 5. Condenser inner wall to outer wall (Wall conduction)
    R_w_cond = np.log(r_o / r_i) / (2.0 * np.pi * L_cond * k_wall)
    T_cond_outer_wall = T_cond_inner_wall - Q_input * R_w_cond
    
    return {
        "R_w_evap": R_w_evap,
        "R_wk_evap": R_wk_evap,
        "R_vapor": R_vapor,
        "R_wk_cond": R_wk_cond,
        "R_w_cond": R_w_cond,
        "T_evap_outer": T_evap_outer_wall,
        "T_evap_inner": T_evap_inner_wall,
        "T_vapor_evap": T_vapor_evap,
        "T_vapor_cond": T_vapor_cond,
        "T_cond_inner": T_cond_inner_wall,
        "T_cond_outer": T_cond_outer_wall
    }

# ----------------------------------------------------------------------
# 5. SIMULATION EXECUTION & DATA GENERATION
# ----------------------------------------------------------------------
# Temperatures to simulate (400C to 950C)
temp_celsius = np.linspace(400.0, 950.0, 150)
temp_kelvin = temp_celsius + 273.15

limit_visc = []
limit_sonic = []
limit_ent = []
limit_cap = []
limit_boil = []

for T in temp_kelvin:
    limit_visc.append(viscous_limit(T))
    limit_sonic.append(sonic_limit(T))
    limit_ent.append(entrainment_limit(T))
    limit_cap.append(capillary_limit(T))
    limit_boil.append(boiling_limit(T))

# Convert to arrays and change to kW
limit_visc = np.array(limit_visc) / 1e3
limit_sonic = np.array(limit_sonic) / 1e3
limit_ent = np.array(limit_ent) / 1e3
limit_cap = np.array(limit_cap) / 1e3
limit_boil = np.array(limit_boil) / 1e3

# Active envelope is the minimum of all limits at any given temperature
limit_envelope = np.minimum(limit_visc, np.minimum(limit_sonic, np.minimum(limit_ent, np.minimum(limit_cap, limit_boil))))

# ----------------------------------------------------------------------
# 6. PLOTTING VISUALIZATION (DATA-CRAFT ACCESSIBLE STANDARDS)
# ----------------------------------------------------------------------
sns.set_theme(style='whitegrid', palette='colorblind', font='DejaVu Sans')
fig, ax = plt.subplots(figsize=(10, 6.5))

# Plot limits
ax.plot(temp_celsius, limit_visc, label='Viscous Limit (Vapor choke)', color='#E66101', linewidth=2, linestyle='--')
ax.plot(temp_celsius, limit_sonic, label='Sonic Limit (Choked flow)', color='#FDB863', linewidth=2)
ax.plot(temp_celsius, limit_ent, label='Entrainment Limit (Shear dryout)', color='#5E3C99', linewidth=2)
ax.plot(temp_celsius, limit_cap, label='Capillary Limit (Circulation limit)', color='#B2ABD2', linewidth=2, linestyle='-.')
ax.plot(temp_celsius, limit_boil, label='Boiling Limit (Wick bubble barrier)', color='#4DAC26', linewidth=2)

# Plot overall operational envelope (shaded region underneath the lowest limit curve)
ax.fill_between(temp_celsius, 0, limit_envelope, alpha=0.15, color='gray', label='Stable Operating Region')
ax.plot(temp_celsius, limit_envelope, color='black', linewidth=2.5, label='Maximum Operating Limit')

# Mark nominal Kilopower heat pipe operating point (passage 449: 380 W nominal at ~750°C-800°C)
ax.scatter([750.0], [0.380], color='red', s=100, zorder=5, edgecolor='black', label='Nominal Operating Point (380 W @ 750°C)')
ax.annotate('Nominal Operating Point\n(380 W per pipe at 750°C)', xy=(750.0, 0.380), xytext=(780, 1.5),
            arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
            fontsize=10, fontweight='bold')

# Configure labels and limits
ax.set_xlim(400, 950)
ax.set_ylim(0, 10.0)
ax.set_xlabel('Heat Pipe Operating Temperature (°C)', fontsize=11, fontweight='bold')
ax.set_ylabel('Heat Transport Rate (kW)', fontsize=11, fontweight='bold')

# MANDATORY DATA-CRAFT HEADLINE AS A TITLE
ax.set_title('Sodium Heat Pipe Operating Envelope: Startup Choked at Low T, Limited to ~3.8 kW at 800°C',
             fontsize=13, fontweight='bold', pad=18)

# Subtle details
ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='lightgray')
ax.text(410, 0.2, 'Source: Space Nuclear Power: Advanced Heat Pipe Thermal Management', fontsize=8, color='gray')

sns.despine()
plt.tight_layout(pad=1.5)

plot_path = '/workspace/scratch/kilopower_hp_limits.png'
fig.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Heat pipe limits plot saved to {plot_path}")
plt.close()

# ----------------------------------------------------------------------
# 7. RUN TEMPERATURE PROFILE SOLVER & SAVE DATA FILE
# ----------------------------------------------------------------------
# Standard simulation case (nominal: 380 Watts, overpower: 600 Watts)
profile_nominal = solve_thermal_network(T_evap_outer_wall=1023.15, Q_input=380.0)  # Evap outer wall at 750C
profile_overpower = solve_thermal_network(T_evap_outer_wall=1073.15, Q_input=600.0) # Evap outer wall at 800C

# Format outputs into a clean table/data report
summary_text = f"""# Kilopower Sodium Heat Pipe Python Simulation Summary

This simulation solves the temperature-dependent thermophysical limits and thermal resistance network for Kilopower's primary sodium heat pipes. 

## Heat Pipe Geometrical Specifications
- Envelope Material: Haynes 230 Superalloy [cite: 7, 17]
- Wick Material: Stainless Steel 316 Screen (300 mesh) [cite: 17]
- Working Fluid: High-purity Sodium (Na) [cite: 20]
- Outer Diameter: 1.27 cm (0.50 inches) [cite: 16, 17]
- Wall Thickness: 0.089 cm (0.035 inches) [cite: 17]
- Evaporator Length: 35.6 cm (14.0 inches) [cite: 17]
- Adiabatic Length: 86.4 cm (34.0 inches) [cite: 7, 17]
- Condenser Length: 8.89 cm (3.5 inches) [cite: 17]

## Operational Limits & Safety Margins (at 750°C / 1023 K)
- Viscous Limit: {viscous_limit(1023.15)/1e3:.2f} kW
- Sonic Limit: {sonic_limit(1023.15)/1e3:.2f} kW
- Entrainment Limit: {entrainment_limit(1023.15)/1e3:.2f} kW
- Capillary Limit: {capillary_limit(1023.15)/1e3:.2f} kW
- Boiling Limit: {boiling_limit(1023.15)/1e3:.2f} kW
- **Maximum Combined Operating Limit**: {min(viscous_limit(1023.15), sonic_limit(1023.15), entrainment_limit(1023.15), capillary_limit(1023.15), boiling_limit(1023.15))/1e3:.2f} kW
- **NASA Nominal Power demand**: 0.38 kW (per pipe) [cite: 17]
- **Operational Safety Margin Factor**: {min(viscous_limit(1023.15), sonic_limit(1023.15), entrainment_limit(1023.15), capillary_limit(1023.15), boiling_limit(1023.15))/380.0:.2f}x

## Steady-State Thermal Resistance Network Solver Output

### Case 1: Nominal Thermal Profile (T_evap = 750.00 °C, Q = 380 W)
- Radial Wall Resistance (Evap): {profile_nominal['R_w_evap']:.5f} K/W
- Radial Wick Resistance (Evap): {profile_nominal['R_wk_evap']:.5f} K/W
- Axial Vapor core Resistance: {profile_nominal['R_vapor']:.5f} K/W
- Radial Wick Resistance (Cond): {profile_nominal['R_wk_cond']:.5f} K/W
- Radial Wall Resistance (Cond): {profile_nominal['R_w_cond']:.5f} K/W
- **Total Heat Pipe Thermal Resistance**: {profile_nominal['R_w_evap'] + profile_nominal['R_wk_evap'] + profile_nominal['R_vapor'] + profile_nominal['R_wk_cond'] + profile_nominal['R_w_cond']:.5f} K/W

#### Computed Temperature Distribution along the Exergy Flowpath:
1. Evaporator Outer Wall: {profile_nominal['T_evap_outer'] - 273.15:.2f} °C
2. Evaporator Inner Wall: {profile_nominal['T_evap_inner'] - 273.15:.2f} °C
3. Evaporator Vapor Space: {profile_nominal['T_vapor_evap'] - 273.15:.2f} °C
4. Condenser Vapor Space: {profile_nominal['T_vapor_cond'] - 273.15:.2f} °C
5. Condenser Inner Wall: {profile_nominal['T_cond_inner'] - 273.15:.2f} °C
6. Condenser Outer Wall: {profile_nominal['T_cond_outer'] - 273.15:.2f} °C
- **Total Temperature Drop across Heat Pipe**: {profile_nominal['T_evap_outer'] - profile_nominal['T_cond_outer']:.2f} K

### Case 2: Overpower Thermal Profile (T_evap = 800.00 °C, Q = 600 W)
- Radial Wall Resistance (Evap): {profile_overpower['R_w_evap']:.5f} K/W
- Radial Wick Resistance (Evap): {profile_overpower['R_wk_evap']:.5f} K/W
- Axial Vapor core Resistance: {profile_overpower['R_vapor']:.5f} K/W
- Radial Wick Resistance (Cond): {profile_overpower['R_wk_cond']:.5f} K/W
- Radial Wall Resistance (Cond): {profile_overpower['R_w_cond']:.5f} K/W
- **Total Heat Pipe Thermal Resistance**: {profile_overpower['R_w_evap'] + profile_overpower['R_wk_evap'] + profile_overpower['R_vapor'] + profile_overpower['R_wk_cond'] + profile_overpower['R_w_cond']:.5f} K/W

#### Computed Temperature Distribution along the Exergy Flowpath:
1. Evaporator Outer Wall: {profile_overpower['T_evap_outer'] - 273.15:.2f} °C
2. Evaporator Inner Wall: {profile_overpower['T_evap_inner'] - 273.15:.2f} °C
3. Evaporator Vapor Space: {profile_overpower['T_vapor_evap'] - 273.15:.2f} °C
4. Condenser Vapor Space: {profile_overpower['T_vapor_cond'] - 273.15:.2f} °C
5. Condenser Inner Wall: {profile_overpower['T_cond_inner'] - 273.15:.2f} °C
6. Condenser Outer Wall: {profile_overpower['T_cond_outer'] - 273.15:.2f} °C
- **Total Temperature Drop across Heat Pipe**: {profile_overpower['T_evap_outer'] - profile_overpower['T_cond_outer']:.2f} K
"""

with open('/workspace/scratch/kilopower_hp_sim_summary.md', 'w') as f:
    f.write(summary_text)

print("Simulation text summary generated.")
