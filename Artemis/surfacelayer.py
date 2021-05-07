import numpy as np

from cm4twc.components import SurfaceLayerComponent
from cm4twc.settings import dtype_float


class Artemis(SurfaceLayerComponent):
    _inputs_info = {
        'precipitation_flux': {
            'units': 'kg m-2 s-1',
            'kind': 'dynamic'
        },
        'specific_humidity': {
            'units': '1',
            'kind': 'dynamic'
        },
        'surface_downwelling_shortwave_flux_in_air': {
            'units': 'W m-2',
            'kind': 'dynamic'
        },
        'surface_downwelling_longwave_flux_in_air': {
            'units': 'W m-2',
            'kind': 'dynamic'
        },
        'air_temperature': {
            'units': 'K',
            'kind': 'dynamic'
        },
        'surface_albedo': {
            'units': '1',
            'kind': 'static'
        },
        'wind_speed': {
            'units': 'm s-1',
            'kind': 'dynamic'
        },
        'vegetation_height': {
            'units': 'm',
            'kind': 'static'
        },
        'leaf_area_index': {
            'units': '1',
            'kind': 'climatologic',
            'frequency': 'monthly'
        }
    }
    _parameters_info = {}
    _states_info = {
        'canopy_store': {
            'units': 'm'
        },
        'snowpack_store': {
            'units': 'm'
        }
    }
    _constants_info = {
        'sigma_sb': {
            'description': 'Stefan-Boltzman constant',
            'units': 'W m-2 K-4',
            'default_value': 5.67e-8
        },
        'kappa': {
            'description': 'von Karman constant',
            'units': '1',
            'default_value': 0.41
        },
        'lambda_lh': {
            'description': 'latent heat of vaporisation '
                           '(water vapour @ 288K) - M&U90 p.268',
            'units': 'J kg-1',
            'default_value': 2.48e6
        },
        'rho_a': {
            'description': 'specific mass of (dry) air @ 288K - M&U90 p.268',
            'units': 'kg m-3',
            'default_value': 1.22
        },
        'rho_lw': {
            'description': 'specific mass of liquid water',
            'units': 'kg m-3',
            'default_value': 1e3
        },
        'c_p': {
            'description': 'specific heat of air - M&U90 p.267',
            'units': 'J kg-1 K-1',
            'default_value': 1.01e3
        },
        'R_v': {
            'description': 'gas constant for water vapour - P&P77 p.23',
            'units': 'J kg-1 K-1',
            'default_value': 461.5
        },
        't_freeze': {
            'description': 'freezing point of water',
            'units': 'K',
            'default_value': 273.
        },
        'gamma': {
            'description': 'psychrometric constant [kPa/K] M&U90 p.181',
            'units': 'kPa K-1',
            'default_value': 0.066
        },
        'D_s': {
            'description': 'drainage rate when canopy is full - '
                           'Shuttleworth, 2007. p. 218; orig Rutter',
            'units': 'm s-1',
            'default_value': 6e-9
        },
        'b': {
            'description': 'canopy drainage decay term - '
                           'Shuttleworth, 2007. p. 218',
            'units': 's-1',
            'default_value': 0.062
        },
        'F': {
            'description': 'degree-day melting factor - '
                           'Beven 2000; 0.004 m/K/day',
            'units': 'm K-1 s-1',
            'default_value': 4.6e-8
        },
        'z': {
            'description': 'height measurement for wind speed',
            'units': 'm',
            'default_value': 10
        }
    }
    _outputs_info = {
        'evaporation_canopy': {
            'description': 'evaporation of the water on the leaf surface',
            'units': 'kg m-2 s-1'
        }
    }

    def initialise(self,
                   # component states
                   canopy_store, snowpack_store,
                   **kwargs):

        canopy_store[-1][:] = 0
        snowpack_store[-1][:] = 0

    def run(self,
            # from exchanger
            # component inputs
            precipitation_flux, air_temperature, wind_speed, specific_humidity,
            surface_downwelling_shortwave_flux_in_air,
            surface_downwelling_longwave_flux_in_air, surface_albedo,
            leaf_area_index, vegetation_height,
            # component parameters
            # component states
            canopy_store, snowpack_store,
            # component constants
            sigma_sb, kappa, lambda_lh, rho_a, rho_lw, c_p, t_freeze,
            R_v, gamma, D_s, b, F, z,
            **kwargs):

        # /!\__RENAMING_CM4TWC__________________________________________
        dt = self.timedelta_in_seconds
        datetime = self.current_datetime

        pr = precipitation_flux
        huss = specific_humidity
        tas = air_temperature
        rsds = surface_downwelling_shortwave_flux_in_air
        rlds = surface_downwelling_longwave_flux_in_air

        h = vegetation_height
        ancil_L = leaf_area_index

        canopy_prev = canopy_store[-1]
        snowpack_prev = snowpack_store[-1]
        # ______________________________________________________________

        # Compute derived parameters
        # vegetation roughness length [m] (3e-4 is for bare soil)
        z0 = np.ma.where(h == 0, 3e-4, 0.1 * h)
        # zero-plane displacement [m]
        d = 0.75 * h
        # temperature for snowfall [K]
        t_snow = t_freeze
        # snowmelt threshold [K]
        T_F = t_freeze

        with np.errstate(over='ignore'):
            # Update LAI and derived terms
            month = datetime.month
            L = ancil_L[month-1, ...]
            # Canopy storage capacity [m] (Hough and Jones, MORECS)
            C_t = 0.002 * L
            # throughfall coefficient [-]
            phi_t = np.ma.where(L == 0, 1, 1 - 0.5 ** L)
            # Canopy resistance (calc from LAI) [s/m] (Beven 2000 p. 76)
            r_c = 40.

            # Compute derived meteorological variables
            T_degC = tas - 273.
            # Scale to 2m wind speed
            wind_speed = wind_speed * 4.87 / (np.log(67.8 * z - 5.42))

            # Hendrik; kPa
            e_sat = 0.6108 * np.exp(T_degC * 17.27 / (237.3 + T_degC))
            # kg/kg to kPa; Parish and Putnam 1977)
            e_abs = huss * rho_a * R_v * tas / 1000
            # slope of SVP curve kPa / degC
            Delta_e = 4098. * e_sat / (237.3 + T_degC) ** 2
            with np.errstate(invalid='ignore'):
                # aerodynamic resistance to evaporation
                r_a = np.log((z - d) / z0) ** 2 / (kappa ** 2 * wind_speed)
            # Prevent negative r_a
            r_a = np.ma.where(r_a <= 0., 1, r_a)

            # /!\__ADJUSTMENT_CM4TWC____________________________________
            # calculate upwelling shortwave "rsus" with albedo
            rsus = surface_albedo * rsds

            # calculate upwelling longwave "rlus" with air temperature
            # (assuming surface temperature is air temperature)
            # (ignoring emissivity)
            rlus = air_temperature ** 4 * sigma_sb

            # total net radiation in W m-2
            R_n = rsds + rlds - rsus - rlus
            # __________________________________________________________

        # Partition precipitation between canopy, snow, and soil
        pr = pr / rho_lw  # convert to m/s
        p_snow = np.ma.where(tas <= t_snow, pr, 0.)
        p_soil = np.ma.where(tas > t_snow, phi_t * pr, 0.)
        p_can = np.ma.where(tas > t_snow, pr - p_soil, 0.)

        # CANOPY

        # Add canopy rainfall to store from previous timestep
        canopy = canopy_prev + p_can * dt

        # Calculate canopy evaporation, e_can
        # vapour pressure gradient (kPa)
        D = e_sat - e_abs
        D = np.ma.where(D >= 0, D, 0.)
        e_can = (Delta_e * R_n + (rho_a * c_p * D) / r_a) / (Delta_e + gamma)
        # Penman-Monteith (rc = 0; Rutter/Gash)
        # convert to m/s
        e_can = e_can / lambda_lh / 1000.
        # scale canopy evaporation by canopy wetted fraction
        e_can = e_can * (canopy / C_t)
        # limit e_can to available canopy water
        e_can = np.minimum(canopy / dt, e_can)
        # update
        canopy = canopy - e_can * dt
        # avoid small roundoff values
        canopy = np.ma.where(canopy < 1.e-11, 0., canopy)

        # Calculate canopy throughfall, q_t
        q_t = np.ma.where(canopy < C_t, 0., D_s * np.exp(b * (canopy - C_t)))

        # limit throughfall if it would deplete more than available
        q_t = np.minimum(canopy / dt, q_t)
        # update
        canopy = canopy - q_t * dt
        # avoid small roundoff values
        canopy = np.ma.where(canopy < 1.e-11, 0., canopy)

        # Ensure canopy storage doesnt exceed capacity
        q_t = np.ma.where(canopy > C_t, q_t + (canopy - C_t) / dt, q_t)
        canopy = np.ma.where(canopy > C_t, C_t, canopy)

        # SNOW PACK

        # Calculate snowmelt, q_m
        # Degree-day melting model
        q_m = F * np.maximum(0., tas - T_F)

        # Update snowpack
        # add new snow to store from previous timestep
        snowpack = snowpack_prev + p_snow * dt
        # limit snowmelt to available snowpack
        q_m = np.minimum(snowpack / dt, q_m)
        # update
        snowpack = snowpack - q_m * dt
        # avoid small roundoff values
        snowpack = np.ma.where(snowpack < 1.e-11, 0., snowpack)

        # SURFACE

        # surface store
        # add new rain, snowmelt and throughfall
        surface = (p_soil + q_m + q_t) * dt

        # Surface evaporation
        # Penman-Monteith
        e_surf = ((Delta_e * R_n + (rho_a * c_p * D) / r_a)
                  / (Delta_e + gamma * (1 + r_c / r_a)))
        # convert to m/s
        e_surf = e_surf / lambda_lh / 1000.
        # subtract canopy evaporation if possible, to conserve energy
        e_surf = np.maximum(e_surf - e_can, 0.)
        # limit e_surf to available water
        e_surf = np.minimum(surface / dt, e_surf)

        # /!\__UPDATE_STATES_CM4TWC_____________________________________
        canopy_store[0][:] = canopy
        snowpack_store[0][:] = snowpack
        # ______________________________________________________________

        return (
            # to exchanger
            {
                'throughfall':
                    (q_t + p_soil) * rho_lw,
                'snowmelt':
                    q_m * rho_lw,
                'transpiration':
                    np.zeros(self.spaceshape, dtype_float()),
                'evaporation_soil_surface':
                    e_surf * rho_lw,
                'evaporation_ponded_water':
                    np.zeros(self.spaceshape, dtype_float()),
                'evaporation_openwater':
                    np.zeros(self.spaceshape, dtype_float())
            },
            # component outputs
            {
                'evaporation_canopy':
                    e_can * rho_lw
            }
        )

    def finalise(self, **kwargs):

        pass
