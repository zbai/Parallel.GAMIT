"""
Project: Parallel.GAMIT
Date: 9/13/25 5:22 PM
Author: Demian D. Gomez
"""
from typing import List, Tuple
from dataclasses import dataclass, field
import numpy as np
import logging

logger = logging.getLogger(__name__)

# app
from etm.core.etm_config import ETMConfig
from etm.core.etm_config import JumpType
from etm.etm_functions.jumps import JumpFunction
from etm.data.solution_data import SolutionData

class JumpManager:
    """Comprehensive jump management system"""

    def __init__(self, solution_data: SolutionData, config: ETMConfig):
        self.config = config
        self.jumps: List[JumpFunction] = []
        self.auto_detected_jumps: List[JumpFunction] = []
        # max and min dates
        self.date_min = min(solution_data.date)
        self.date_max = max(solution_data.date)

    def build_jump_table(self, time_vector: np.ndarray) -> None:
        """
        Build complete jump table including database user_jumps and auto-detected user_jumps

        Args:
            time_vector: Time vector for jump evaluation
        """
        logger.info("Building jump table")

        # Load earthquake-based user_jumps
        if self.config.modeling.fit_earthquakes:
            self._load_earthquake_jumps(time_vector)

        # Load mechanical/generic user_jumps
        if self.config.modeling.fit_generic_jumps:
            # Load manual user_jumps from database
            self._load_manual_jumps(time_vector)

        if self.config.modeling.fit_metadata_jumps:
            self._load_mechanical_jumps(time_vector)

        # Add automatic jump detection
        # auto_jumps = self._create_auto_detected_jumps(solution_data, time_vector)
        # self.auto_detected_jumps = auto_jumps

        # Validate final jump configuration
        # self._validate_jump_constraints(time_vector)

        logger.info(f"Final jump table: {len(self.get_active_jumps())} active user_jumps")

    def _load_earthquake_jumps(self, time_vector: np.ndarray) -> None:
        """Load earthquake-based user_jumps from catalog"""

        for eq in self.config.modeling.earthquake_jumps:
            if (self.date_min <= eq.date <= self.date_max or (
                    self.date_min - self.config.modeling.post_seismic_back_lim  <= eq.date <= self.date_max
                    and eq.magnitude >= 7)):

                jump = JumpFunction(
                    config=self.config,
                    metadata=eq.build_metadata(),
                    time_vector=time_vector,
                    date=eq.date,
                    magnitude=eq.magnitude,
                    epi_distance=eq.distance,
                    jump_type=eq.jump_type)

                self.add_jump(jump)

    def _load_mechanical_jumps(self, time_vector: np.ndarray) -> None:
        """Load mechanical user_jumps from station metadata and database"""
        # Load antenna changes from station info
        try:
            records = self.config.metadata.station_information

            for i, record in enumerate(records[1:], 1):
                prev_record = records[i - 1]

                # Check if equipment actually changed
                if (prev_record['AntennaCode'] != record['AntennaCode'] or
                        prev_record['RadomeCode'] != record['RadomeCode']):

                    if prev_record['RadomeCode']:
                        pre_ant = prev_record['AntennaCode'] + ' ' + prev_record['RadomeCode']
                    else:
                        pre_ant = prev_record['AntennaCode'] + ' NONE'

                    if record['RadomeCode']:
                        new_ant = record['AntennaCode'] + ' ' + record['RadomeCode']
                    else:
                        new_ant = record['AntennaCode'] + ' NONE'

                    # Check database for manual override
                    jump = JumpFunction(
                        config=self.config,
                        metadata=f'Antenna: {pre_ant}->{new_ant}',
                        time_vector=time_vector,
                        date=record['DateStart'],
                        jump_type=JumpType.MECHANICAL_ANTENNA)

                    # jump outside the region with data, deactivate
                    if not self.date_min < record['DateStart'] < self.date_max:
                        jump.remove_from_fit()

                    self.add_jump(jump)

        except Exception as e:
            logger.warning(f"Could not load station info for mechanical jumps: {e}")

    def _create_auto_detected_jumps(self, solution_data: SolutionData,
                                    time_vector: np.ndarray) -> List[JumpFunction]:
        """Create jump functions for automatically detected discontinuities"""
        pass

    def _validate_jump_constraints(self, time_vector: np.ndarray) -> None:
        """Validate final jump configuration and apply constraints"""
        active_jumps = self.get_active_jumps()

        # Check last jump has sufficient data
        if active_jumps:
            last_jump = None
            for jump in reversed(active_jumps):
                if jump.jump_function.fit:
                    last_jump = jump
                    break

            if last_jump:
                # Check data availability after last jump
                data_after_jump = np.sum(time_vector > last_jump.date.fyear)
                min_data_required = max(
                    self.config.validation.min_data_for_jump,
                    last_jump.param_count * 10  # Heuristic: 10 observations per parameter
                )

                if data_after_jump < min_data_required:
                    logger.warning(
                        f"Insufficient data after last jump {last_jump.date.yyyyddd()}: "
                        f"{data_after_jump} < {min_data_required}"
                    )
                    # Convert complex jump to simple jump
                    if last_jump.p.jump_type == JumpType.COSEISMIC_JUMP_DECAY:
                        last_jump.p.jump_type = JumpType.COSEISMIC_ONLY
                        last_jump.param_count = 1
                        last_jump.design = last_jump._create_design_matrix(time_vector)
                        last_jump.rehash()

    def get_active_jumps(self) -> List[JumpFunction]:
        """Get list of user_jumps that are fitted"""
        return [jump for jump in self.jumps if jump.fit]

    def get_jump_functions(self) -> List[JumpFunction]:
        return [jump for jump in self.jumps]

    def get_parameter_count(self) -> int:
        """Get total parameter count for active user_jumps"""
        return sum(jump.param_count for jump in self.jumps if jump.fit)

    def add_jump(self, jump: JumpFunction) -> None:
        """Add jump with conflict resolution"""
        # Check for conflicts with existing user_jumps
        for existing_jump in self.jumps:
            if existing_jump.fit:
                is_equivalent, preferred = existing_jump.__eq__(jump)
                if is_equivalent:
                    if preferred is jump:
                        existing_jump.remove_from_fit()
                    else:
                        jump.remove_from_fit()
                    break

        user_action, user_type = self._get_jump_action(jump)

        # deactivate jump if user selected decided to
        if not self._should_fit_jump(user_action):
            jump.remove_from_fit(user_action=user_action)

        # check that type match! if types don't match, transform the jump
        if not jump.p.jump_type == user_type:
            jump.configure_behavior({'jump_type': user_type})

        if not jump.user_action == user_action:
            jump.user_action = user_action

        self.jumps.append(jump)

        self.jumps.sort()

    def _get_jump_action(self, jump) -> Tuple[str, JumpType]:
        """Get jump action override from database"""
        # find if user has activated or deactivated this event
        jump_params = self.config.modeling.get_user_jump(jump.date, jump.p.jump_type)
        if jump_params:
            return jump_params.action, jump_params.jump_type
        else:
            return 'A', jump.p.jump_type

    def _should_fit_jump(self, user_action: str) -> bool:
        """Determine if mechanical jump should be fitted. To decide in case of geophysical jump, if user action is +
        and the jump action is A, then deactivate it because it will be added later on as a manual jump by
        _load_manual_jumps
        """
        if user_action == '+':
            return True
        elif user_action == '-':
            return False
        else: # any other case, set user_action == 'A'
            return self.config.modeling.fit_generic_jumps

    def _load_manual_jumps(self, time_vector: np.ndarray) -> None:
        """Load manually specified user_jumps from database"""
        # Implementation for loading manual user_jumps
        jump_dates = [jump.date for jump in self.jumps]
        for j in self.config.modeling.user_jumps:
            # if jump date not in table already, then add it
            if j.date not in jump_dates:
                if j.jump_type == JumpType.MECHANICAL_MANUAL:
                    metadata = 'Manual mechanic jump'
                else:
                    metadata = 'Manual geophysical jump'

                jump = JumpFunction(
                    config=self.config,
                    time_vector=time_vector,
                    date=j.date,
                    jump_type=j.jump_type,
                    user_action=j.action,
                    metadata=metadata
                )
                self.add_jump(jump)

    def validate_all_jumps(self) -> List[str]:
        """Validate all user_jumps and return issues"""
        all_issues = []
        for jump in self.jumps:
            issues = jump.validate_parameters()
            all_issues.extend(issues)
        return all_issues

