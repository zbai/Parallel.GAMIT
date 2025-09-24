"""
Project: Parallel.GAMIT
Date: 9/21/25 4:54 PM
Author: Demian D. Gomez
"""

from enum import IntEnum, auto


class FitStatus(IntEnum):
    """Enum for fit status"""
    PREFIT = auto()
    POSTFIT = auto()


class PeriodicStatus(IntEnum):
    """Enum for periodic term status"""
    AUTOMATICALLY_ADDED = auto()
    ADDED_BY_USER = auto()
    UNABLE_TO_FIT = auto()


class EtmSolutionType(IntEnum):
    """Enum for jump types to replace scattered constants"""
    MODEL = auto()
    OBSERVATION = auto()
    @property
    def description(self) -> str:
        descriptions = {
            EtmSolutionType.MODEL: 'Solution from model',
            EtmSolutionType.OBSERVATION: 'Solution from direct observation'
        }
        return descriptions.get(self, 'UNKNOWN')


class JumpType(IntEnum):
    """Enum for jump types to replace scattered constants"""
    UNDETERMINED = -1
    MECHANICAL_MANUAL = 1
    MECHANICAL_ANTENNA = 2
    REFERENCE_FRAME = 5
    COSEISMIC_JUMP_DECAY = 10
    COSEISMIC_ONLY = 15
    POSTSEISMIC_ONLY = 20

    @property
    def color(self) -> str:
        """Get the plotting color for this jump type"""
        color_map = {
            JumpType.UNDETERMINED: 'gray',
            JumpType.MECHANICAL_MANUAL: 'darkcyan',
            JumpType.MECHANICAL_ANTENNA: 'blue',
            JumpType.REFERENCE_FRAME: 'green',
            JumpType.COSEISMIC_JUMP_DECAY: 'red',
            JumpType.COSEISMIC_ONLY: 'purple',
            JumpType.POSTSEISMIC_ONLY: 'orange'
        }
        return color_map.get(self, 'black')

    @property
    def description(self) -> str:
        descriptions = {
            JumpType.UNDETERMINED: 'UNDETERMINED',
            JumpType.MECHANICAL_MANUAL: 'MECHANICAL (MANUAL)',
            JumpType.MECHANICAL_ANTENNA: 'MECHANICAL (ANTENNA CHANGE)',
            JumpType.REFERENCE_FRAME: 'REFERENCE FRAME CHANGE',
            JumpType.COSEISMIC_JUMP_DECAY: 'CO+POSTSEISMIC',
            JumpType.COSEISMIC_ONLY: 'COSEISMIC ONLY',
            JumpType.POSTSEISMIC_ONLY: 'POSTSEISMIC ONLY'
        }
        return descriptions.get(self, 'UNKNOWN')


class NoiseModels(IntEnum):
    """Enum for noise models"""
    WHITE_ONLY = auto()
    WHITE_FLICKER = auto()
    WHITE_FLICKER_RW = auto()


class AdjustmentModels(IntEnum):
    """Enum for noise models"""
    ROBUST_LEAST_SQUARES = auto()
    LSQ_COLLOCATION = auto()
    @property
    def description(self) -> str:
        descriptions = {
            AdjustmentModels.ROBUST_LEAST_SQUARES: 'Robust LSQ',
            AdjustmentModels.LSQ_COLLOCATION: 'LSQ Collocation'
        }
        return descriptions.get(self, 'UNKNOWN')


class SolutionType(IntEnum):
    """Enum for noise models"""
    GAMIT = auto()
    PPP = auto()
    NGL = auto()