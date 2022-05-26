# Auto-Inverse


You can download the whole repository from [here]().
For details about method MINI you can find the repository [here](https://gitlab.mpi-klsb.mpg.de/nansari/mixed-integer-neural-inverse-design).
For details about method INN you can find the repository [here](https://github.com/BensonRen/BDIMNNA).

--------------------------------------------------
--------------------------------------------------

### Requirements:
- PyTorch (CUDA)
- Numpy
- Scipy

--------------------------------------------------
--------------------------------------------------
First, please unzip [data.zip](https://drive.google.com/file/d/1E7inIOtmOUvFoDT-QIs_87h9GSS5HHEB/view?usp=sharing) and [results.zip](https://drive.google.com/file/d/1f8PnU8XseDAHoAYwPqgYJNAFRlzymTmw/view?usp=sharing) (if you want to do all the experiments from scratch then don't unzip results.zip, but if you want to use some parts of our results, like trained models, then you need to unzip results.zip) in the project directory. Now please find the details below, on how to run an experiment.

If you want to train the first mu model in the uana method for the multi_joint_robot experiment, you can use the following command:

- python main.py \--config configs/config1.json \--method uana \--step train_mu_model \--model_id 0

Or, if you want to invert the spectral_printer_noisy target data using na method, you can use the following command:

- python main.py \--config configs/config3.json \--method na \--step invert_target \--experiment_repetition_id 0 \--inversion_repetition_id 0 \--target_name evaluation

(if you want to invert the spectral_printer_noisy target data using na method for 50 times then you need to run above command with inversion_repetition_id from 0 to 49).

Or, if you want to aggregate results of the spectral_printer_sparse inverted targets using nae method, you can use the following command:

- python main.py \--config configs/config4.json \--method nae \--step aggregate_results \--experiment_repetition_id 0 \--target_name evaluation

Or, if you want to evaluate results of the spectral_printer_standard inverted targets using ua_tandem method, you can use the following command:

- python main.py \--config configs/config5.json \--method ua_tandem \--step evaluate_results \--experiment_repetition_id 0 \--target_name evaluation

--------------------------------------------------
--------------------------------------------------

### Methods:

--------------------------------------------------

#### NA:
##### Train Forward Model:
- python main.py \--config < config-file-path > \--method na \--step train_forward_model

##### Invert Target:
- python main.py \--config < config-file-path > \--method na \--step invert_target \--experiment_repetition_id < experiment-repetition-id > \--inversion_repetition_id < inversion-repetition-id > \--target_name < evaluation / tuning >

##### Aggregate Results:
- python main.py \--config < config-file-path > \--method na \--step aggregate_results \--experiment_repetition_id < experiment-repetition-id > \--target_name < evaluation / tuning >

##### Evaluate Results:
- python main.py \--config < config-file-path > \--method na \--step evaluate_results \--experiment_repetition_id < experiment-repetition-id > \--target_name < evaluation / tuning >

--------------------------------------------------

#### NAE:
##### Invert Target:
- python main.py \--config < config-file-path > \--method nae \--step invert_target \--experiment_repetition_id < experiment-repetition-id > \--inversion_repetition_id < inversion-repetition-id > \--target_name < evaluation / tuning >

##### Aggregate Results:
- python main.py \--config < config-file-path > \--method nae \--step aggregate_results \--experiment_repetition_id < experiment-repetition-id > \--target_name < evaluation / tuning >

##### Evaluate Results:
- python main.py \--config < config-file-path > \--method nae \--step evaluate_results \--experiment_repetition_id < experiment-repetition-id > \--target_name < evaluation / tuning >

--------------------------------------------------

#### UANA:
##### Train MU Model:
- python main.py \--config < config-file-path > \--method uana \--step train_mu_model \--model_id < model-id >

##### Train Sigma Model:
- python main.py \--config < config-file-path > \--method uana \--step train_sigma_model \--model_id < model-id >

##### Invert Target:
- python main.py \--config < config-file-path > \--method uana \--step invert_target \--experiment_repetition_id < experiment-repetition-id > \--inversion_repetition_id < inversion-repetition-id > \--target_name < evaluation / tuning >

##### Aggregate Results:
- python main.py \--config < config-file-path > \--method uana \--step aggregate_results \--experiment_repetition_id < experiment-repetition-id > \--target_name < evaluation / tuning >

##### Evaluate Results:
- python main.py \--config < config-file-path > \--method uana \--step evaluate_results \--experiment_repetition_id < experiment-repetition-id > \--target_name < evaluation / tuning >

--------------------------------------------------

#### Tandem:
##### Train Backward Model:
- python main.py \--config < config-file-path > \--method tandem \--step train_backward_model \--experiment_repetition_id < experiment-repetition-id >

##### Invert Target:
- python main.py \--config < config-file-path > \--method tandem \--step invert_target \--experiment_repetition_id < experiment-repetition-id > \--target_name < evaluation / tuning >

##### Evaluate Results:
- python main.py \--config < config-file-path > \--method tandem \--step evaluate_results \--experiment_repetition_id < experiment-repetition-id > \--target_name < evaluation / tuning >

--------------------------------------------------

#### UA Tandem:
##### Train Backward Model:
- python main.py \--config < config-file-path > \--method ua_tandem \--step train_backward_model \--experiment_repetition_id < experiment-repetition-id >

##### Invert Target:
- python main.py \--config < config-file-path > \--method ua_tandem \--step invert_target \--experiment_repetition_id < experiment-repetition-id > \--target_name < evaluation / tuning >

##### Evaluate Results:
- python main.py \--config < config-file-path > \--method ua_tandem \--step evaluate_results \--experiment_repetition_id < experiment-repetition-id > \--target_name < evaluation / tuning >
--------------------------------------------------
#### Help:
- **\--config**, flag to specify config file path (e.g., \--config configs/config1.json).
- **\--method**, flag to specify method (e.g., \--method ua_tandem).
- **\--step**, flag to specify step of the specified method (e.g., \--step aggregate_results).
- **\--model_id**, flag to specify the model id, starting from 0 (e.g., \--model_id 5).
- **\--experiment_repetition_id**, flag to specify the experiment repetition id (if you want to repeat an experiment, use the same experiment_repetition_id in diffrent steps of a method), starting from 0 (e.g., \--experiment_repetition_id 1).
- **\--inversion_repetition_id**, flag to specify the inversion repetition id, in methods that repeat the inversion it is necessary to provide this flag, starting from 0 (e.g., \--inversion_repetition_id 25).
- **\--target_name**, flag to specify the target name, if you want to tune the parameters in the config file then use 'tuning' as target_name otherwise use 'evaluation' to use the proper target file (results of tuning are not available in the results.zip, so if you want to use tuning for this flag then you have to run all the steps of a method that requires \--target_name from the beginning with tuning as value of this flag) (e.g., \--target_name evaluation).

--------------------------------------------------
--------------------------------------------------
### Configs:
- **multi_joint_robot:** config1.json
- **spectral_printer_noisy:** config3.json
- **spectral_printer_sparse:** config4.json
- **spectral_printer_standard:** config5.json
- **spectral_printer_sparse (ReLU):** config6.json
- **soft_robot_noisy:** config8.json
- **soft_robot_sparse:** config9.json
- **soft_robot_standard:** config10.json
