import os

import numpy
import scipy.io

BASE_PATH = r''


def load_mat_file(
            path: str,
            key: str = 'data'
        ) -> numpy.ndarray:

    return scipy.io.loadmat(path)[key]


def save_mat_file(
            data: numpy.ndarray,
            path: str,
            key: str = 'data'
        ) -> None:

    scipy.io.savemat(path, {key: data})


def main():
    repetition_ids = sorted(
        [
            int(directory)
            for directory in os.listdir(BASE_PATH)
            if directory.isdigit() and os.path.isdir(os.path.join(BASE_PATH, directory))
        ]
    )

    total_no_reduction_losses = []
    for i in range(len(repetition_ids)):
        repetition_id = repetition_ids[i]
        total_no_reduction_losses.append(
            load_mat_file(
                path=os.path.join(BASE_PATH, str(repetition_id), 'no_reduction_loss.mat')
            )
        )
    total_no_reduction_losses = numpy.array(total_no_reduction_losses).T
    batch_size = total_no_reduction_losses.shape[0]

    best_repetition_id_per_element = []
    best_design = []
    best_no_reduction_loss = []
    best_performance = []
    best_sigma_aleatory = []
    best_sigma_epistemic = []
    for element_id in range(batch_size):
        best_repetition_id = numpy.argmin(total_no_reduction_losses[element_id, :])

        best_repetition_id_per_element.append(best_repetition_id)
        best_design.append(
            load_mat_file(
                path=os.path.join(BASE_PATH, str(best_repetition_id), 'design.mat')
            )[element_id, :]
        )
        best_no_reduction_loss.append(
            load_mat_file(
                path=os.path.join(BASE_PATH, str(best_repetition_id), 'no_reduction_loss.mat')
            )[0, element_id]
        )
        best_performance.append(
            load_mat_file(
                path=os.path.join(BASE_PATH, str(best_repetition_id), 'performance.mat')
            )[element_id, :]
        )
        best_sigma_aleatory.append(
            load_mat_file(
                path=os.path.join(BASE_PATH, str(best_repetition_id), 'sigma_aleatory.mat')
            )[element_id, :]
        )
        best_sigma_epistemic.append(
            load_mat_file(
                path=os.path.join(BASE_PATH, str(best_repetition_id), 'sigma_epistemic.mat')
            )[element_id, :]
        )

    result_base_path = os.path.join(BASE_PATH, 'aggregated')
    if not os.path.exists(result_base_path):
        os.makedirs(result_base_path)

    save_mat_file(
        data=numpy.array(best_repetition_id_per_element),
        path=os.path.join(result_base_path, 'repetition_id_per_element.mat'),
    )
    save_mat_file(
        data=numpy.array(best_design),
        path=os.path.join(result_base_path, 'design.mat'),
    )
    save_mat_file(
        data=numpy.array(best_no_reduction_loss),
        path=os.path.join(result_base_path, 'no_reduction_loss.mat'),
    )
    save_mat_file(
        data=numpy.array(best_performance),
        path=os.path.join(result_base_path, 'performance.mat'),
    )
    save_mat_file(
        data=numpy.array(best_sigma_aleatory),
        path=os.path.join(result_base_path, 'sigma_aleatory.mat'),
    )
    save_mat_file(
        data=numpy.array(best_sigma_epistemic),
        path=os.path.join(result_base_path, 'sigma_epistemic.mat'),
    )


if __name__ == "__main__":
    main()
