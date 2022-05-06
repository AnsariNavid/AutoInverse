from typing import Tuple

import numpy


def _determine_final_position(
            origin_position: numpy.ndarray,
            arm_angles: numpy.ndarray
        ) -> Tuple[numpy.ndarray, numpy.ndarray]:

    number_of_arms = 3
    arm_lengths = [0.5, 0.5, 1]
    positions = []

    current_position = numpy.zeros([len(origin_position), 2])
    current_position[:, 0] = 0
    current_position[:, 1] = origin_position
    positions.append(numpy.copy(current_position))

    arm_angles_diff = numpy.copy(arm_angles)
    arm_angles_diff[:, 1] -= arm_angles[:, 0]
    arm_angles_diff[:, 2] -= arm_angles[:, 1] + arm_angles[:, 0]

    for arm_index in range(number_of_arms):
        current_position[:, 0] += numpy.cos(arm_angles_diff[:, arm_index]) * arm_lengths[arm_index]
        current_position[:, 1] += numpy.sin(arm_angles_diff[:, arm_index]) * arm_lengths[arm_index]
        positions.append(numpy.copy(current_position))

    return current_position, numpy.array(positions)


def get_re_simulation_error(
            designs: numpy.ndarray,
            targets: numpy.ndarray
        ) -> float:

    current_positions, _ = _determine_final_position(origin_position=designs[:, 0], arm_angles=designs[:, 1:])
    return (numpy.square(targets - current_positions)).mean()


def get_surrogate_error(
            performances: numpy.ndarray,
            targets: numpy.ndarray
        ) -> float:

    return (numpy.square(targets - performances)).mean()
