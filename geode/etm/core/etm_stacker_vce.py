"""
Variance Component Estimation (VCE) for EtmStacker
Implements iterative estimation of variance components for different equation groups
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt

from ...Utils import stationID

@dataclass
class VarianceComponent:
    """Stores information about a variance component group"""
    name: str
    sigma2: float  # Variance factor
    redundancy: int  # Effective degrees of freedom
    chi2: float  # Weighted sum of squared residuals
    n_equations: int  # Number of equations in this group


@dataclass
class VCEResults:
    """Results from VCE iteration"""
    iteration: int
    components: Dict[str, VarianceComponent]
    solution: np.ndarray
    covariance: np.ndarray
    global_variance: float
    converged: bool
    max_change: float


class EtmStackerVCE:
    """
    Variance Component Estimation for EtmStacker

    Estimates optimal relative weights between:
    - Observation equations (from trajectory fits)
    - Interseismic constraints
    - Coseismic constraints (per earthquake)
    - Postseismic constraints (per earthquake/relaxation)
    """

    def __init__(self, stacker, max_iterations: int = 20, tolerance: float = 1e-3,
                 min_variance: float = 1e-6):
        """
        Parameters
        ----------
        stacker : EtmStacker
            The stacker object with build_system() already called
        max_iterations : int
            Maximum VCE iterations
        tolerance : float
            Convergence threshold for relative change in variance factors
        min_variance : float
            Minimum allowed variance factor (for numerical stability)
        """
        self.stacker = stacker
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.min_variance = min_variance

        self.history: List[VCEResults] = []
        self.converged = False

    def run_vce(self, initial_weights: Optional[Dict[str, float]] = None,
                fix_observations: bool = True) -> VCEResults:
        """
        Run VCE to estimate variance components

        Parameters
        ----------
        initial_weights : dict, optional
            Initial relative weights for each group
            Keys: 'observations', 'interseismic', 'coseismic_<event_id>',
                  'postseismic_<event_id>_<relax>'
        fix_observations : bool
            If True, keep observation variance = 1.0 and scale others relative to it
            (recommended for geodetic networks)

        Returns
        -------
        VCEResults
            Final iteration results
        """

        # Initialize variance factors
        variance_factors = self._initialize_variance_factors(initial_weights)

        tqdm.write('\n' + '='*70)
        tqdm.write('Starting Variance Component Estimation')
        tqdm.write('='*70)

        for iteration in range(self.max_iterations):
            tqdm.write(f'\n--- VCE Iteration {iteration + 1} ---')

            # Apply current variance factors as weights
            self._apply_variance_factors_as_weights(variance_factors)

            # Solve system with current weights
            # if system is already solved, don't solve again, just compute residuals
            if iteration == 0 and self.stacker.solved:
                self.stacker.solve(interpolate_fields=False)

            # Compute residuals for each group
            residual_groups = self._compute_residual_groups()

            # Update variance factors
            variance_factors_new = {}
            components = {}

            for group_name, residual_info in residual_groups.items():
                chi2 = residual_info['chi2']
                redundancy = residual_info['redundancy']
                n_eq = residual_info['n_equations']

                # Estimate variance factor: σ² = v'Pv / r
                if redundancy > 0:
                    sigma2_new = chi2 / redundancy
                else:
                    sigma2_new = 1.0
                    tqdm.write(f'  WARNING: {group_name} has zero redundancy, setting σ²=1')

                # Enforce minimum variance
                sigma2_new = max(sigma2_new, self.min_variance)

                variance_factors_new[group_name] = sigma2_new

                components[group_name] = VarianceComponent(
                    name=group_name,
                    sigma2=sigma2_new,
                    redundancy=redundancy,
                    chi2=chi2,
                    n_equations=n_eq
                )

                old_sigma2 = variance_factors.get(group_name, 1.0)
                change = sigma2_new - old_sigma2

                tqdm.write(f'  {group_name:40s}: σ²={sigma2_new:8.4f}  '
                          f'(Δ={change:8.4f})  χ²={chi2:10.2f}  r={redundancy:6d}')

            # Check convergence
            max_change = self._compute_max_change(
                variance_factors, variance_factors_new, fix_observations
            )

            converged = self._check_convergence(
                variance_factors_old=variance_factors,
                variance_factors_new=variance_factors_new,
                abs_tol=self.tolerance
            )

            # Store results
            results = VCEResults(
                iteration=iteration + 1,
                components=components,
                solution=self.stacker.solution.copy(),
                covariance=None, # self.stacker.covariance.copy(),
                global_variance=self.stacker.variance,
                converged=converged,
                max_change=max_change
            )
            self.history.append(results)

            tqdm.write(f'\n     Max change: {max_change:.6f}')
            tqdm.write(f'  Global variance: {self.stacker.variance:.6f}')

            if converged:
                tqdm.write(f'\n*** VCE CONVERGED after {iteration + 1} iterations ***\n')
                self.converged = True
                break

            # Update for next iteration
            variance_factors = variance_factors_new

        if not converged:
            tqdm.write(f'\n*** VCE did NOT converge after {self.max_iterations} iterations ***')
            tqdm.write(f'    Consider increasing max_iterations or relaxing tolerance\n')

        return self.history[-1]

    @staticmethod
    def _check_convergence(variance_factors_old: Dict[str, float],
                           variance_factors_new: Dict[str, float],
                           abs_tol: float) -> bool:
        """
        Check convergence using absolute criterion only

        Converged if ALL variance components satisfy:
            |new - old| < abs_tol
        """
        max_abs_change = 0.0

        for key in variance_factors_new.keys():
            old_val = variance_factors_old.get(key, 1.0)
            new_val = variance_factors_new[key]

            abs_change = abs(new_val - old_val)
            max_abs_change = max(max_abs_change, abs_change)

            # If any component change exceeds tolerance, not converged
            if abs_change >= abs_tol:
                return False

        return True

    def _initialize_variance_factors(self, initial_weights: Optional[Dict[str, float]]) -> Dict[str, float]:
        """Initialize variance factors for all equation groups"""

        variance_factors = {'observations': 1.0}  # Observations start at 1.0

        # Interseismic
        if initial_weights and 'interseismic' in initial_weights:
            variance_factors['interseismic'] = initial_weights['interseismic']
        else:
            variance_factors['interseismic'] = 1.0

        # Coseismic and postseismic for each earthquake
        for event in self.stacker.earthquakes:
            coseis_key = f'coseismic_{event.id}'
            if initial_weights and coseis_key in initial_weights:
                variance_factors[coseis_key] = initial_weights[coseis_key]
            else:
                variance_factors[coseis_key] = 1.0

            for relax in self.stacker.config.relaxation:
                postseis_key = f'postseismic_{event.id}_{relax}'
                if initial_weights and postseis_key in initial_weights:
                    variance_factors[postseis_key] = initial_weights[postseis_key]
                else:
                    variance_factors[postseis_key] = 1.0

        return variance_factors

    def _apply_variance_factors_as_weights(self, variance_factors: Dict[str, float]):
        """
        Convert variance factors to weights and apply to stacker

        Weight = 1/σ² (scaled appropriately)
        """

        # Update interseismic
        inter_var = variance_factors.get('interseismic', 1.0)
        # Weight is proportional to 1/variance
        # Current weight is in config, we scale it by relative variance
        h_sigma = self.stacker.config.interseismic_h_sigma * np.sqrt(inter_var)
        v_sigma = self.stacker.config.interseismic_v_sigma * np.sqrt(inter_var)
        self.stacker.update_weights(constraint_type='interseismic',
                                   h_sigma=h_sigma, v_sigma=v_sigma)

        # Update coseismic and postseismic for each event
        for event in self.stacker.earthquakes:
            # Coseismic
            coseis_key = f'coseismic_{event.id}'
            coseis_var = variance_factors.get(coseis_key, 1.0)
            h_sigma = self.stacker.config.coseismic_h_sigma * np.sqrt(coseis_var)
            v_sigma = self.stacker.config.coseismic_v_sigma * np.sqrt(coseis_var)
            self.stacker.update_weights(event_id=event.id, constraint_type='coseismic',
                                        h_sigma=h_sigma, v_sigma=v_sigma)

            # Postseismic for each relaxation
            for relax in self.stacker.config.relaxation:
                postseis_key = f'postseismic_{event.id}_{relax}'
                postseis_var = variance_factors.get(postseis_key, 1.0)
                h_sigma = self.stacker.config.postseismic_h_sigma * np.sqrt(postseis_var)
                v_sigma = self.stacker.config.postseismic_v_sigma * np.sqrt(postseis_var)
                self.stacker.update_weights(event_id=event.id, constraint_type='postseismic',
                                            h_sigma=h_sigma, v_sigma=v_sigma, relax=relax)

    def _compute_residual_groups(self) -> Dict[str, Dict]:
        """
        Compute weighted residuals for each equation group

        Returns dict with structure:
        {
            'observations': {'chi2': float, 'redundancy': int, 'n_equations': int},
            'interseismic': {...},
            'coseismic_<event_id>': {...},
            'postseismic_<event_id>_<relax>': {...}
        }
        """

        residual_groups = {}

        # 1. Observations group
        obs_chi2 = 0.0
        obs_equations = 0
        obs_parameters = self.stacker.total_parameters * 3  # E, N, U

        for stn in self.stacker.stations:
            neq = stn.normal_equations
            for i in range(3):  # E, N, U
                x = self.stacker.solution[i, neq.parameter_range]
                v = neq.observation_vector[i] - neq.design_matrix @ x
                # Weighted sum of squared residuals (ensure scalar)
                chi2_component = v.T @ np.diag(neq.observation_weights[i]) @ v
                # Handle 0-d array from quadratic form
                if isinstance(chi2_component, np.ndarray):
                    chi2_component = chi2_component.item()
                obs_chi2 += chi2_component
                obs_equations += neq.equation_count

        obs_redundancy = obs_equations - obs_parameters

        residual_groups['observations'] = {
            'chi2': obs_chi2,
            'redundancy': max(obs_redundancy, 1),  # Avoid zero
            'n_equations': obs_equations
        }

        # 2. Constraint groups
        for constraint_type in self.stacker.constraint_registry.constraints.keys():
            for const in self.stacker.constraint_registry.constraints[constraint_type]:

                # Build group key
                if constraint_type == 'interseismic':
                    group_key = 'interseismic'
                elif constraint_type == 'coseismic':
                    group_key = f'coseismic_{const.event.id}'
                elif constraint_type == 'postseismic':
                    group_key = f'postseismic_{const.event.id}_{const.relaxation}'
                else:
                    continue

                # Compute residuals for this constraint
                chi2 = 0.0
                n_equations = 0

                for eq in [e for e in const.equations if e.is_active]:
                    ke, kn, ku = eq.constraint_design
                    se, sn, su = eq.constraint_sigma

                    # Compute residuals: v = -K*x (since z0 = 0)
                    v_e = ke @ self.stacker.solution.flatten()
                    v_n = kn @ self.stacker.solution.flatten()
                    v_u = ku @ self.stacker.solution.flatten()

                    # Compute chi² = vᵀPv = Σ(vᵢ/σᵢ)²
                    # This correctly handles both scalar and vector residuals
                    chi2 += np.sum((np.atleast_1d(v_e) / se) ** 2)
                    chi2 += np.sum((np.atleast_1d(v_n) / sn) ** 2)
                    chi2 += np.sum((np.atleast_1d(v_u) / su) ** 2)

                    n_equations += 3  # E, N, U

                # Redundancy for constraints
                # Approximation: number of constraint equations
                # (More sophisticated: trace of projection matrix)
                redundancy = n_equations

                residual_groups[group_key] = {
                    'chi2': chi2,
                    'redundancy': max(redundancy, 1),
                    'n_equations': n_equations
                }

        return residual_groups

    def _compute_max_change(self, old_factors: Dict[str, float],
                            new_factors: Dict[str, float],
                            fix_observations: bool) -> float:
        """Compute maximum relative change in variance factors"""

        max_change = 0.0

        for key in new_factors.keys():
            if fix_observations and key == 'observations':
                continue  # Skip observations if fixed

            old_val = old_factors.get(key, 1.0)
            new_val = new_factors[key]
            change = abs(new_val - old_val)
            max_change = max(max_change, change)

        return max_change

    def plot_convergence(self, filename: Optional[str] = None):
        """Plot VCE convergence history"""

        if not self.history:
            print("No VCE history to plot")
            return

        # Extract data
        iterations = [r.iteration for r in self.history]

        # Get all component names
        all_components = set()
        for result in self.history:
            all_components.update(result.components.keys())

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Variance factors over iterations
        ax = axes[0, 0]
        for comp_name in sorted(all_components):
            sigma2_values = []
            for result in self.history:
                if comp_name in result.components:
                    sigma2_values.append(result.components[comp_name].sigma2)
                else:
                    sigma2_values.append(np.nan)
            ax.plot(iterations, sigma2_values, 'o-', label=comp_name, linewidth=2)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Variance Factor σ²')
        ax.set_title('Convergence of Variance Components')
        # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Plot 2: Chi-squared values
        ax = axes[0, 1]
        for comp_name in sorted(all_components):
            chi2_values = []
            for result in self.history:
                if comp_name in result.components:
                    chi2_values.append(result.components[comp_name].chi2)
                else:
                    chi2_values.append(np.nan)
            ax.plot(iterations, chi2_values, 'o-', label=comp_name, linewidth=2)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('χ² (Weighted Sum of Squared Residuals)')
        ax.set_title('Chi-Squared by Component')
        # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Plot 3: Global variance
        ax = axes[1, 0]
        global_vars = [r.global_variance for r in self.history]
        ax.plot(iterations, global_vars, 'ko-', linewidth=2, markersize=8)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Global Variance Factor')
        ax.set_title('Global Variance of Unit Weight')
        ax.grid(True, alpha=0.3)

        # Plot 4: Max relative change
        ax = axes[1, 1]
        max_changes = [r.max_change for r in self.history]
        ax.semilogy(iterations, max_changes, 'ro-', linewidth=2, markersize=8)
        ax.axhline(self.tolerance, color='k', linestyle='--',
                  label=f'Tolerance ({self.tolerance})')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Max Relative Change')
        ax.set_title('Convergence Criterion')
        # ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            tqdm.write(f'Convergence plot saved to {filename}')
        else:
            plt.show()

        return fig

    def get_final_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Get final relative weights for each constraint type

        Returns
        -------
        dict with structure:
        {
            'interseismic': {'h_sigma': float, 'v_sigma': float},
            'coseismic': {event_id: {'h_sigma': float, 'v_sigma': float}},
            'postseismic': {(event_id, relax): {'h_sigma': float, 'v_sigma': float}}
        }
        """

        if not self.history:
            raise RuntimeError("No VCE has been run yet")

        final_result = self.history[-1]
        weights = {
            'interseismic': {},
            'coseismic': {},
            'postseismic': {}
        }

        for comp_name, component in final_result.components.items():
            if comp_name == 'observations':
                continue
            elif comp_name == 'interseismic':
                # Find the current constraint to get sigmas
                for const in self.stacker.constraint_registry.constraints['interseismic']:
                    # Get first equation to extract current sigmas
                    if const.equations:
                        eq = const.equations[0]
                        _, _, _ = eq.constraint_design
                        se, sn, su = eq.constraint_sigma
                        weights['interseismic'] = {
                            'h_sigma': (se + sn) / 2,  # Average of E and N
                            'v_sigma': su
                        }
                    break
            elif comp_name.startswith('coseismic_'):
                event_id = comp_name.replace('coseismic_', '')
                for const in self.stacker.constraint_registry.constraints['coseismic']:
                    if const.event.id == event_id and const.equations:
                        eq = const.equations[0]
                        se, sn, su = eq.constraint_sigma
                        weights['coseismic'][event_id] = {
                            'h_sigma': (se + sn) / 2,
                            'v_sigma': su
                        }
                        break
            elif comp_name.startswith('postseismic_'):
                parts = comp_name.replace('postseismic_', '').rsplit('_', 1)
                event_id = parts[0]
                relax = float(parts[1])
                for const in self.stacker.constraint_registry.constraints['postseismic']:
                    if const.event.id == event_id and const.relaxation == relax and const.equations:
                        eq = const.equations[0]
                        se, sn, su = eq.constraint_sigma
                        weights['postseismic'][(event_id, relax)] = {
                            'h_sigma': (se + sn) / 2,
                            'v_sigma': su
                        }
                        break

        return weights

    def print_summary(self):
        """Print summary of VCE results"""

        if not self.history:
            print("No VCE results to summarize")
            return

        final_result = self.history[-1]

        print("\n" + "="*70)
        print("VCE SUMMARY")
        print("="*70)
        print(f"Converged: {self.converged}")
        print(f"Iterations: {len(self.history)}")
        print(f"Final max relative change: {final_result.max_change:.6f}")
        print(f"Global variance: {final_result.global_variance:.6f}")
        print("\nFinal Variance Components:")
        print("-"*70)
        print(f"{'Component':<40s} {'σ²':>10s} {'χ²':>12s} {'Redundancy':>12s}")
        print("-"*70)

        for comp_name in sorted(final_result.components.keys()):
            comp = final_result.components[comp_name]
            print(f"{comp_name:<40s} {comp.sigma2:>10.4f} {comp.chi2:>12.2f} {comp.redundancy:>12d}")

        print("="*70 + "\n")


class EtmStackerVCEEnhanced(EtmStackerVCE):
    """
    Enhanced VCE that can estimate observation variance factor

    Key difference: Accounts for the fact that observations dominate
    the system (500k obs vs 2k constraints), so observation variance
    must be estimated properly.
    """

    def run_vce_with_observation_scaling(self,
                                         initial_obs_variance: float = 1.0,
                                         scale_per_station: bool = False) -> VCEResults:
        """
        Run VCE with proper observation variance estimation

        This is the recommended approach when:
        - σ₀² >> 1 even with very loose constraints
        - Observations heavily outnumber constraints (>100:1)
        - Individual trajectory weights are too tight for network solution

        Parameters
        ----------
        initial_obs_variance : float
            Initial variance factor for observations (start with σ₀² from loose-constraint solve)
        scale_per_station : bool
            If True, estimate different variance factors for each station
            If False, single global observation variance factor

        Returns
        -------
        VCEResults
            Final iteration results
        """

        if scale_per_station:
            return self._run_vce_per_station_variance(initial_obs_variance)
        else:
            return self._run_vce_global_obs_variance(initial_obs_variance)

    def _run_vce_global_obs_variance(self, initial_obs_variance: float) -> VCEResults:
        """
        VCE with single global observation variance factor

        Approach:
        1. Scale ALL observation weights by 1/σ²_obs
        2. Estimate constraint variances relative to scaled observations
        3. Iterate
        """

        tqdm.write('\n' + '=' * 70)
        tqdm.write('VCE with Global Observation Variance Estimation')
        tqdm.write('=' * 70)

        # Initialize variances
        variance_factors = self._initialize_variance_factors(None)
        obs_variance = initial_obs_variance

        for iteration in range(self.max_iterations):
            tqdm.write(f'\n--- VCE Iteration {iteration + 1} ---')

            if iteration > 0 or not self.stacker.solved:
                # if system is already solved, don't solve again, just compute residuals
                # 1. Scale observation weights by observation variance factor
                self._apply_observation_variance(obs_variance)
                # 2. Apply constraint weights
                self._apply_variance_factors_as_weights(variance_factors)

                # 3. Solve
                self.stacker.solve(interpolate_fields=False)

            # 4. Compute residual groups
            residual_groups = self._compute_residual_groups()

            # 5. Estimate NEW observation variance
            obs_info = residual_groups['observations']
            obs_variance_new = obs_info['chi2'] / obs_info['redundancy']
            obs_variance_new = max(obs_variance_new, self.min_variance)

            tqdm.write(f'  {"observations":45s}: σ²={obs_variance_new:8.4f}  '
                       f'(Δ={(obs_variance_new - obs_variance):8.4f})  '
                       f'χ²={obs_info["chi2"]:10.2f}  r={obs_info["redundancy"]:6d}')

            # 6. Estimate constraint variances (relative to current obs weights)
            variance_factors_new = {}
            components = {}

            components['observations'] = VarianceComponent(
                name='observations',
                sigma2=obs_variance_new,
                redundancy=obs_info['redundancy'],
                chi2=obs_info['chi2'],
                n_equations=obs_info['n_equations']
            )

            for group_name, residual_info in residual_groups.items():
                if group_name == 'observations':
                    continue

                chi2 = residual_info['chi2']
                redundancy = residual_info['redundancy']
                n_eq = residual_info['n_equations']

                if redundancy > 0:
                    sigma2_new = chi2 / redundancy
                else:
                    sigma2_new = 1.0

                sigma2_new = max(sigma2_new, self.min_variance)
                variance_factors_new[group_name] = sigma2_new

                components[group_name] = VarianceComponent(
                    name=group_name,
                    sigma2=sigma2_new,
                    redundancy=redundancy,
                    chi2=chi2,
                    n_equations=n_eq
                )

                old_sigma2 = variance_factors.get(group_name, 1.0)
                change = sigma2_new - old_sigma2

                tqdm.write(f'  {group_name:45s}: σ²={sigma2_new:8.4f}  '
                           f'(Δ={change:8.4f})  χ²={chi2:10.2f}  r={redundancy:6d}')

            # Check convergence on ALL variances (including observations)
            all_old = {'observations': obs_variance}
            all_old.update(variance_factors)
            all_new = {'observations': obs_variance_new}
            all_new.update(variance_factors_new)

            max_change = self._compute_max_change(all_old, all_new, fix_observations=False)

            converged = self._check_convergence(
                variance_factors_old=variance_factors,
                variance_factors_new=variance_factors_new,
                abs_tol=self.tolerance
            )

            # Store results
            results = VCEResults(
                iteration=iteration + 1,
                components=components,
                solution=self.stacker.solution.copy(),
                covariance=None, #self.stacker.covariance.copy(),
                global_variance=self.stacker.variance,
                converged=converged,
                max_change=max_change
            )
            self.history.append(results)

            tqdm.write('\n' + '-' * 70)
            tqdm.write(f'      Max change: {max_change:.6f}')
            tqdm.write(f' Global variance: {self.stacker.variance:.6f}')

            if converged:
                tqdm.write(f'\n*** VCE CONVERGED after {iteration + 1} iterations ***\n')
                self.converged = True
                break

            # Update for next iteration
            variance_factors = variance_factors_new
            obs_variance = obs_variance_new

        if not converged:
            tqdm.write(f'\n*** VCE did NOT converge after {self.max_iterations} iterations ***\n')

        # Final message about observation scaling
        final_obs_variance = self.history[-1].components['observations'].sigma2
        tqdm.write(f'\n{"=" * 70}')
        tqdm.write(f'RECOMMENDATION:')
        tqdm.write(f'{"=" * 70}')
        tqdm.write(f'Set observation_variance_factor = {final_obs_variance:.4f} in your config')
        tqdm.write(f'This will scale all observation weights by 1/{final_obs_variance:.4f}')
        tqdm.write(f'Then re-run VCE with fix_observations=True for constraint weights')
        tqdm.write(f'{"=" * 70}\n')

        return self.history[-1]

    def _apply_observation_variance(self, obs_variance: float):
        """
        Scale observation weights by observation variance factor

        This modifies the stacker's normal equations in place
        """
        for stn in self.stacker.stations:
            # Scale weights by 1/obs_variance
            scale_factor = 1.0 / obs_variance

            self.stacker.change_station_weight(stationID(stn), scale_factor, silent=True)

    def _run_vce_per_station_variance(self, initial_obs_variance: float) -> VCEResults:
        """
        VCE with per-station observation variance factors

        This is more flexible but has more parameters to estimate.
        Only use if you suspect significant station-to-station quality differences.
        """

        tqdm.write('\n' + '=' * 70)
        tqdm.write('VCE with Per-Station Observation Variance Estimation')
        tqdm.write('=' * 70)
        tqdm.write('WARNING: This estimates variance for EACH station separately')
        tqdm.write('         Only use if station quality varies significantly\n')

        # Store original weights
        original_weights = {}
        for stn in self.stacker.stations:
            neq = stn.normal_equations
            original_weights[neq.station] = {
                'observation_weights': [ow.copy() for ow in neq.observation_weights],
                'weight_scale': neq.weight_scale
            }

        # Initialize: one variance per station
        station_variances = {neq.station: initial_obs_variance
                             for neq in self.stacker.normal_equations}

        # Initialize constraint variances
        variance_factors = self._initialize_variance_factors(None)

        for iteration in range(self.max_iterations):
            tqdm.write(f'\n--- VCE Iteration {iteration + 1} ---')

            # 1. Apply per-station observation variances
            for stn in self.stacker.stations:
                neq = stn.normal_equations
                station_id = neq.station
                obs_var = station_variances[station_id]

                # Scale this station's weights
                orig = original_weights[station_id]
                scale_factor = 1.0 / obs_var

                for i in range(3):
                    neq.observation_weights[i] = orig['observation_weights'][i] * scale_factor
                    p_scaled = np.diag(neq.observation_weights[i])
                    neq.neq[i] = neq.design_matrix.T @ p_scaled @ neq.design_matrix
                    neq.ceq[i] = neq.design_matrix.T @ p_scaled @ neq.observation_vector[i]
                    neq.weighted_observations[i] = (neq.observation_vector[i].T @
                                                    p_scaled @ neq.observation_vector[i])

            # 2. Apply constraint weights
            self._apply_variance_factors_as_weights(variance_factors)

            # 3. Solve
            # if system is already solved, don't solve again, just compute residuals
            if iteration == 0 and self.stacker.solved:
                self.stacker.solve(interpolate_fields=False)

            # 4. Compute residuals PER STATION
            station_variances_new = {}

            for stn in self.stacker.stations:
                neq = stn.normal_equations
                station_id = neq.station

                # Compute chi2 for this station (ensure scalar)
                chi2 = 0.0
                for i in range(3):
                    x = self.stacker.solution[i, neq.parameter_range]
                    v = neq.observation_vector[i] - neq.design_matrix @ x
                    chi2_component = v.T @ np.diag(neq.observation_weights[i]) @ v
                    # Handle 0-d array from quadratic form
                    if isinstance(chi2_component, np.ndarray):
                        chi2_component = chi2_component.item()
                    chi2 += chi2_component

                # Redundancy for this station
                redundancy = neq.dof  # Already computed as n_eq - n_param

                if redundancy > 0:
                    station_variances_new[station_id] = chi2 / redundancy
                else:
                    station_variances_new[station_id] = station_variances[station_id]

            # 5. Constraint variances (same as before)
            residual_groups = self._compute_residual_groups()
            variance_factors_new = {}

            for group_name, residual_info in residual_groups.items():
                if group_name == 'observations':
                    continue
                chi2 = residual_info['chi2']
                redundancy = residual_info['redundancy']
                if redundancy > 0:
                    variance_factors_new[group_name] = chi2 / redundancy
                else:
                    variance_factors_new[group_name] = 1.0

            # Print summary (top 10 stations by variance)
            from operator import itemgetter
            top_stations = sorted(station_variances_new.items(),
                                  key=itemgetter(1), reverse=True)[:10]

            tqdm.write('  Top 10 stations by observation variance:')
            for station_id, var in top_stations:
                old_var = station_variances[station_id]
                change = var - old_var
                tqdm.write(f'    {station_id:15s}: σ²={var:8.4f} (Δ={change:8.4f})')

            # Check convergence
            max_station_change = max(
                abs(station_variances_new[sid] - station_variances[sid]) / station_variances[sid]
                for sid in station_variances.keys()
            )

            max_constraint_change = self._compute_max_change(
                variance_factors, variance_factors_new, fix_observations=False
            )

            max_change = max(max_station_change, max_constraint_change)
            converged = self._check_convergence(
                variance_factors_old=variance_factors,
                variance_factors_new=variance_factors_new,
                abs_tol=self.tolerance
            )

            tqdm.write(f'\n     Max change: {max_change:.6f}')
            tqdm.write(f'  Global variance: {self.stacker.variance:.6f}')

            if converged:
                tqdm.write(f'\n*** VCE CONVERGED after {iteration + 1} iterations ***\n')
                self.converged = True
                break

            # Update
            station_variances = station_variances_new
            variance_factors = variance_factors_new

        # Print final station variance statistics
        final_vars = list(station_variances_new.values())
        tqdm.write(f'\n{"=" * 70}')
        tqdm.write(f'FINAL STATION VARIANCE STATISTICS:')
        tqdm.write(f'{"=" * 70}')
        tqdm.write(f'Mean: {np.mean(final_vars):.4f}')
        tqdm.write(f'Median: {np.median(final_vars):.4f}')
        tqdm.write(f'Std: {np.std(final_vars):.4f}')
        tqdm.write(f'Min: {np.min(final_vars):.4f}')
        tqdm.write(f'Max: {np.max(final_vars):.4f}')
        tqdm.write(f'{"=" * 70}\n')

        # Store station variances in results (create custom structure)
        self.final_station_variances = station_variances_new

        return self.history[-1]


def quick_observation_variance_check(stacker):
    """
    Quick check: what observation variance factor brings σ₀² close to 1?

    This runs a single solve with very loose constraints to isolate
    the observation variance issue.

    Usage:
        from etm_stacker_vce_enhanced import quick_observation_variance_check
        obs_var_factor = quick_observation_variance_check(stacker)
    """

    print("\n" + "=" * 70)
    print("QUICK OBSERVATION VARIANCE CHECK")
    print("=" * 70)
    print("Setting all constraints to 100m (essentially unconstrained)")
    print("This isolates the observation variance issue\n")

    # Set very loose constraints
    stacker.update_weights(constraint_type='interseismic', h_sigma=100.0, v_sigma=100.0)
    for event in stacker.earthquakes:
        stacker.update_weights(event_id=event.id, constraint_type='coseismic',
                               h_sigma=100.0, v_sigma=100.0)
        stacker.update_weights(event_id=event.id, constraint_type='postseismic',
                               h_sigma=100.0, v_sigma=100.0)

    # Solve
    stacker.solve(interpolate_fields=False)

    sigma0_squared = stacker.variance
    sigma0 = np.sqrt(sigma0_squared)

    print(f"With 100m constraint sigmas:")
    print(f"  σ₀² = {sigma0_squared:.4f}")
    print(f"  σ₀  = {sigma0:.4f}")
    print(f"\nThis means observation weights should be scaled by: 1/{sigma0_squared:.4f}")
    print(f"Or equivalently, observation sigmas should be multiplied by: {sigma0:.4f}")
    print("\n" + "=" * 70)
    print("RECOMMENDATION:")
    print("=" * 70)
    print(f"1. Use observation_variance_factor = {sigma0_squared:.4f}")
    print(f"2. This will downweight observations to match network constraints")
    print(f"3. Then run VCE with fix_observations=True for fine-tuning")
    print("=" * 70 + "\n")

    return sigma0_squared


if __name__ == '__main__':
    print(__doc__)