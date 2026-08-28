"""Reference values for the dimensionless normalization used by the solvers.

These are not SI constants. Length, time, charge, and mass are scaled so that
the electron charge magnitude, electron mass, speed of light, and vacuum
permittivity are unity. A run may choose different thermal speeds explicitly.
"""

q_e = -1.0
q_i = 1.0
m_e = 1.0
m_i = 1836.0
v_te = 1.0e-6
v_ti = 5.0e-7
eps_0 = 1.0
c = 1.0
mu_0 = 1.0 / (eps_0 * c**2)

# Using these units we have the following other important units
# Length L: 1 = 3.54112824901163e-14 m  = mu_0 * e^2 / m_e
# Time t: 1 = 1.181193240362181e-22 s = mu_0 * e^2 / (m_e * c)
# Energy E: 1 = 8.187105775475753e-14 J (kg m^2 s^-2) = c * c * m_e
