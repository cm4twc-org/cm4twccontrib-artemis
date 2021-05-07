import numpy as np

from cm4twc.components import SubSurfaceComponent


class Artemis(SubSurfaceComponent):
    _inputs_info = {
        'topmodel_saturation_capacity': {
            'units': 'mm m-1',
            'kind': 'static'
        },
        'saturated_hydraulic_conductivity': {
            'units': 'm s-1',
            'kind': 'static'
        },
        'topographic_index': {
            'units': '1',
            'kind': 'static'
        }
    }
    _parameters_info = {}
    _states_info = {
        'subsurface_store': {
            'units': 'm'
        }
    }
    _constants_info = {
        'm': {
            'description': 'K_sat decay constant - Marthews et al',
            'units': '1',
            'default_value': 0.02
        },
        'rho_lw': {
            'description': 'specific mass of liquid water',
            'units': 'kg m-3',
            'default_value': 1e3
        },
        'S_top': {
            'description': 'soil depth over which to apply topmodel',
            'units': 'm',
            'default_value': 3.
        }
    }

    def initialise(self,
                   # component states
                   subsurface_store,
                   **kwargs):

        subsurface_store[-1][:] = 0

    def run(self,
            # from exchanger
            throughfall, snowmelt, transpiration, evaporation_soil_surface,
            evaporation_ponded_water,
            # component inputs
            topmodel_saturation_capacity,
            saturated_hydraulic_conductivity,
            topographic_index,
            # component parameters
            # component states
            subsurface_store,
            # component constants
            m, rho_lw, S_top,
            **kwargs):

        # /!\__RENAMING_CM4TWC__________________________________________
        dt = self.timedelta_in_seconds

        q_m = snowmelt / rho_lw
        q_t = throughfall / rho_lw
        e_surf = (transpiration + evaporation_soil_surface
                  + evaporation_ponded_water) / rho_lw

        K_s = saturated_hydraulic_conductivity
        S_max = topmodel_saturation_capacity
        Lambda = topographic_index

        subsurface_prev = subsurface_store[-1]
        # ______________________________________________________________

        # SURFACE

        # surface store
        # add new rain, snowmelt and throughfall
        surface = (q_t + q_m) * dt
        # update
        surface = surface - e_surf * dt
        # avoid small roundoff values
        surface = np.ma.where(surface < 1.e-11, 0., surface)

        # Infiltration
        # use JULES formulation rather than full Green-Ampt
        q_i = K_s
        # limit q_i to available water
        q_i = np.minimum(surface / dt, q_i)
        # update
        surface = surface - q_i * dt
        # avoid small roundoff values
        surface = np.ma.where(surface < 1.e-11, 0., surface)

        # Surface runoff
        # everything left over runs off
        q_s = surface / dt

        # SUB-SURFACE

        # Baseflow
        S_max = np.ma.where(S_max > 0., S_max * S_top / 1000., 0.6)
        # add new infiltrated water
        subsurface = subsurface_prev + q_i * dt
        # if soil saturates route excess to surface runoff
        q_s = np.ma.where(subsurface > S_max,
                          q_s + (subsurface - S_max) / dt,
                          q_s)

        subsurface = np.ma.where(subsurface > S_max, S_max, subsurface)
        S_b_prime = S_max - subsurface
        with np.errstate(over='ignore'):
            q_b = (K_s / m) * np.exp(-Lambda) * np.exp(-S_b_prime / m)

        # Update sub-surface store
        # limit q_b to available water
        q_b = np.minimum(subsurface / dt, q_b)
        # remove water that has run off
        subsurface = subsurface - q_b * dt
        # avoid small roundoff values
        subsurface = np.ma.where(subsurface < 1.e-11, 0., subsurface)

        # /!\__ADDITION_CM4TWC__________________________________________
        soil_water_stress = subsurface / S_max
        # ______________________________________________________________

        # /!\__UPDATE_STATES_CM4TWC_____________________________________
        subsurface_store[0][:] = subsurface
        # ______________________________________________________________

        return (
            # to exchanger
            {
                'surface_runoff':
                    q_s * rho_lw,
                'subsurface_runoff':
                    q_b * rho_lw,
                'soil_water_stress':
                    soil_water_stress
            },
            # component outputs
            {}
        )

    def finalise(self, **kwargs):

        pass
