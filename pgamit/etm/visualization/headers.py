from typing import Union
from etm.etm_functions.polynomial import PolynomialFunction
from etm.etm_functions.periodic import PeriodicFunction
from etm.etm_functions.jumps import JumpFunction

class PrintHeader:
    def __init__(self, function: Union[PolynomialFunction, PeriodicFunction, JumpFunction]):
        self.function = function

    def _setup_velocity_from_model(self, velocity: np.ndarray) -> None:
        """Setup format string when velocity comes from external model"""
        self.format_str = (
            f"{self.config.get_label('position')} ({self.p.t_ref:.3f}) "
            "X: {:.3f} Y: {:.3f} Z: {:.3f} [m]\n"
            f"{self.config.get_label('velocity')} ({self.config.get_label('from_model')}) "
            f"N: {velocity[0, 0] * 1000:.2f} E: {velocity[1, 0] * 1000:.2f} "
            f"U: {velocity[2, 0] * 1000:.2f} [mm/yr]"
        )
        self.p.metadata = '[[n:pos, n:vel],[e:pos, e:vel],[u:pos, u:vel]]'

    def _setup_format_strings(self) -> None:
        """Setup format strings based on number of terms"""
        position_str = f"{self.config.get_label('position')} ({self.p.t_ref:.3f}) "
        position_str += "X: {:.3f} Y: {:.3f} Z: {:.3f} [m]"

        terms = self.config.modeling.poly_terms

        if terms == 1:
            self.format_str = position_str
            self.p.metadata = '[[n:pos],[e:pos],[u:pos]]'

        elif terms == 2:
            velocity_str = (f"{self.config.get_label('velocity')} "
                            "N: {:.2f} ± {:.2f} E: {:.2f} ± {:.2f} U: {:.2f} ± {:.2f} [mm/yr]")
            self.format_str = position_str + "\n" + velocity_str
            self.p.metadata = '[[n:pos, n:vel],[e:pos, e:vel],[u:pos, u:vel]]'

        elif terms == 3:
            velocity_str = (f"{self.config.get_label('velocity')} "
                            "N: {:.3f} ± {:.2f} E: {:.3f} ± {:.2f} U: {:.3f} ± {:.2f} [mm/yr]")
            accel_str = (f"{self.config.get_label('acceleration')} "
                         "N: {:.2f} ± {:.2f} E: {:.2f} ± {:.2f} U: {:.2f} ± {:.2f} [mm/yr²]")
            self.format_str = position_str + "\n" + velocity_str + "\n" + accel_str
            self.p.metadata = '[[n:pos, n:vel, n:acc],[e:pos, e:vel, e:acc],[u:pos, u:vel, u:acc]]'

        else:
            # Higher order terms
            velocity_str = (f"{self.config.get_label('velocity')} "
                            "N: {:.3f} ± {:.2f} E: {:.3f} ± {:.2f} U: {:.3f} ± {:.2f} [mm/yr]")
            accel_str = (f"{self.config.get_label('acceleration')} "
                         "N: {:.2f} ± {:.2f} E: {:.2f} ± {:.2f} U: {:.2f} ± {:.2f} [mm/yr²]")
            other_str = f"+ {terms - 3} {self.config.get_label('other')}"
            self.format_str = position_str + "\n" + velocity_str + "\n" + accel_str + " " + other_str
            self.p.metadata = '[[n:pos, n:vel, n:acc, n:tx...],[e:pos, e:vel, e:acc, e:tx...],[u:pos, u:vel, u:acc, u:tx...]]'

    def _setup_format_string_per(self) -> None:
        """Setup format string for parameter printing"""
        if self.frequency_count > 0:
            periods = 1 / (self.p.frequencies * 365.25)
            period_str = ', '.join(f'{p:.1f} yr' for p in periods)
            self.format_str = f"{self.config.get_label('periodic')} ({period_str}) N: %s E: %s U: %s [mm]"
        else:
            self.format_str = f"No {self.config.get_label('periodic').lower()} terms"

    def print_parameters(self, ref_xyz: np.ndarray, lat: float, lon: float) -> str:
        """Generate formatted parameter string"""
        if self.p.params.size == 0:
            return "No polynomial parameters estimated"

        from pgamit.Utils import lg2ct

        params = np.zeros((3,))

        # Position parameters (always first)
        if self.config.modeling.poly_terms >= 1:
            params[0], params[1], params[2] = lg2ct(
                self.p.params[0, 0], self.p.params[1, 0], self.p.params[2, 0],
                lat, lon
            )
            params += ref_xyz.flatten()

        # Add velocity and higher order terms
        format_params = []
        if self.config.modeling.poly_terms >= 2:
            for p in range(1, self.config.modeling.poly_terms):
                for comp in range(3):
                    param_val = self.p.params[comp, p] * 1000  # Convert to mm
                    sigma_val = self.p.sigmas[comp, p] * 1000 if self.p.sigmas.size > 0 else 0
                    format_params.extend([param_val, sigma_val])

        all_params = np.concatenate([params, format_params])
        return self.format_str.format(*all_params.tolist())

    def print_parameters_per(self) -> str:
        """Generate formatted parameter string"""
        if self.p.params.size == 0 or self.frequency_count == 0:
            return "No periodic parameters estimated"

        # Reshape parameters: [sin_freq1, sin_freq2, ..., cos_freq1, cos_freq2, ...]
        n_params = self.p.params[0, :]
        e_params = self.p.params[1, :]
        u_params = self.p.params[2, :]

        # Split into sin and cos components
        mid_point = self.frequency_count
        n_sin, n_cos = n_params[:mid_point], n_params[mid_point:]
        e_sin, e_cos = e_params[:mid_point], e_params[mid_point:]
        u_sin, u_cos = u_params[:mid_point], u_params[mid_point:]

        # Calculate amplitudes for each frequency
        n_amp = np.sqrt(n_sin ** 2 + n_cos ** 2) * 1000  # Convert to mm
        e_amp = np.sqrt(e_sin ** 2 + e_cos ** 2) * 1000
        u_amp = np.sqrt(u_sin ** 2 + u_cos ** 2) * 1000

        return self.format_str % (
            np.array_str(n_amp, precision=1),
            np.array_str(e_amp, precision=1),
            np.array_str(u_amp, precision=1)
        )

    def _setup_metadata(self) -> None:
        """Setup metadata description"""
        if self.frequency_count == 0:
            self.p.metadata = '[[],[],[]]'
            return

        metadata_parts = []
        for comp in ('n', 'e', 'u'):
            comp_meta = []
            for trig_func in ('sin', 'cos'):
                for freq in self.p.frequencies:
                    period = 1 / (freq * 365.25)
                    comp_meta.append(f'{comp}:{trig_func}({period:.1f} yr)')
            metadata_parts.append('[' + ','.join(comp_meta) + ']')

        self.p.metadata = '[' + ','.join(metadata_parts) + ']'