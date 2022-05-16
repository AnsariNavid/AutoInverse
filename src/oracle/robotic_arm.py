import os
from typing import Tuple

import numpy

from .. import utils


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


def _get_re_simulation_error(
            designs: numpy.ndarray,
            targets: numpy.ndarray
        ) -> float:

    current_positions, _ = _determine_final_position(origin_position=designs[:, 0], arm_angles=designs[:, 1:])
    return (numpy.square(targets - current_positions)).mean()


def _get_surrogate_error(
            performances: numpy.ndarray,
            targets: numpy.ndarray
        ) -> float:

    return (numpy.square(targets - performances)).mean()


def evaluate_results(
            config: dict,
            root_path: str,
            target_name: str
        ) -> None:

    designs = utils.load_mat_file(path=os.path.join(root_path, 'design.mat'))['data']
    performances = utils.load_mat_file(path=os.path.join(root_path, 'performance.mat'))['data']
    targets = utils.load_mat_file(path=config[f'{target_name}_path'])['data']

    re_simulation_error = _get_re_simulation_error(designs=designs, targets=targets)
    surrogate_error = _get_surrogate_error(performances=performances, targets=targets)

    data = {
        "re_simulation_error": re_simulation_error,
        "surrogate_error": surrogate_error
    }

    utils.save_json_file(data=data, path=os.path.join(root_path, f'{target_name}_evaluation_results.json'))
