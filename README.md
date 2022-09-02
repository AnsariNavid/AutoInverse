

# Autoinverse

--------------------------------------------------
--------------------------------------------------

### Requirements:
- PyTorch (CUDA)
- Numpy
- Scipy

--------------------------------------------------
--------------------------------------------------

### Experiments:

First please download and unziping [data.zip](https://drive.google.com/file/d/1RQORP1mfD37XVs3IXLl_2eX13qb_DRVS/view?usp=sharing) in the project directory.<br>

*If you want to use our pre-trained networks, you can download and unziping [results.zip](https://drive.google.com/file/d/1oPBXO9ADnMSZxek9dszNl_92mAQ0efa9/view?usp=sharing) in the project directory as well, then you can skip some of the steps for training networks menthoned bellow.*

*Please refer to the section **Help** for more information about each flag we use bellow.*<br>

*Please note that the term **Oracle** used in the code is equivalent to the term **NFP** in the paper.*<br>

#### Section 4.2 (Quantitative comparison of different neural inversion methods) - Table 1:

(description)<br>

NA:
```bash
# Step 1: train forward model
python main.py --config configs\config1.json --method na --step train_forward_model

# Step 2: invert target
for (( inversion_repetition_id = 0; inversion_repetition_id < 250; inversion_repetition_id++ ))
do
	python main.py --config configs\config1.json --method na --step invert_target --experiment_repetition_id 0 --inversion_repetition_id $inversion_repetition_id --target_name evaluation
done

# Step 3: aggregate results
python main.py --config configs\config1.json --method na --step aggregate_results --experiment_repetition_id 0 --target_name evaluation

# Step 4: evaluate results
python main.py --config configs\config1.json --method na --step evaluate_results --experiment_repetition_id 0 --target_name evaluation
```
Results will be available in the following directory: XYZ<br>

UANA:
```bash
# Step 1: train mu models
for (( model_id = 0; model_id < 10; model_id++ ))
do
	python main.py --config configs\config1.json --method uana --step train_mu_model --model_id $model_id
done

# Step 2: train mu models
for (( model_id = 0; model_id < 10; model_id++ ))
do
	python main.py --config configs\config1.json --method uana --step train_sigma_model --model_id $model_id
done

# Step 4: invert target
for (( inversion_repetition_id = 0; inversion_repetition_id < 50; inversion_repetition_id++ ))
do
	python main.py --config configs\config1.json --method uana --step invert_target --experiment_repetition_id 0 --inversion_repetition_id $inversion_repetition_id --target_name evaluation
done

# Step 5: aggregate results
python main.py --config configs\config1.json --method uana --step aggregate_results --experiment_repetition_id 0 --target_name evaluation

# Step 6: evaluate results
python main.py --config configs\config1.json --method uana --step evaluate_results --experiment_repetition_id 0 --target_name evaluation
```
Results will be available in the following directory: XYZ<br>

Tandem:
```bash
# Step 1: train backward model
python main.py --config configs\config1.json --method tandem  --step train_backward_model --experiment_repetition_id 0

# Step 2: invert target
python main.py --config configs\config1.json --method tandem --step invert_target --experiment_repetition_id 0 --target_name evaluation

# Step 3: evaluate results
python main.py --config configs\config1.json --method tandem --step evaluate_results --experiment_repetition_id 0 --target_name evaluation
```
Results will be available in the following directory: XYZ<br>

UA-Tandem:
```bash
# Step 1: train backward model
python main.py --config configs\config1.json --method ua_tandem --step train_backward_model --experiment_repetition_id 0

# Step 2: invert target
python main.py --config configs\config1.json --method ua_tandem --step invert_target --experiment_repetition_id 0 --target_name evaluation

# Step 3: evaluate results
python main.py --config configs\config1.json --method ua_tandem --step evaluate_results --experiment_repetition_id 0 --target_name evaluation
```
Results will be available in the following directory: XYZ<br>

#### Section 4.3 (Neural inversion in the presence of imperfect data) - Table 2:

(description)<br>

For running this experiment please follow these steps:<br>

```bash

```

#### Section C.1 (Counterpart results of tandem and UA-tandem for spectral printer) - Table 5:

(description)<br>

For running this experiment please follow these steps:<br>

```bash

```

--------------------------------------------------
--------------------------------------------------

### Help:
- **\--config**, flag to specify config file path (*e.g., \--config configs/config1.json*).
- **\--method**, flag to specify method (*e.g., \--method ua_tandem*).
- **\--step**, flag to specify step of the specified method (*e.g., \--step aggregate_results*).
- **\--model_id**, flag to specify the model id, starting from 0 (*e.g., \--model_id 5*).
- **\--experiment_repetition_id**, flag to specify the experiment repetition id (if you want to repeat an experiment, use the same experiment_repetition_id in diffrent steps of a method), starting from 0 (*e.g., \--experiment_repetition_id 1*).
- **\--inversion_repetition_id**, flag to specify the inversion repetition id, in methods that repeat the inversion it is necessary to provide this flag, starting from 0 (*e.g., \--inversion_repetition_id 25*).
- **\--target_name**, flag to specify the target name, if you want to tune the parameters in the config file then use 'tuning' as target_name otherwise use 'evaluation' to use the proper target file (results of tuning are not available in the results.zip, so if you want to use tuning for this flag then you have to run all the steps of a method that requires \--target_name from the beginning with 'tuning' as value of this flag) (*e.g., \--target_name evaluation*).

--------------------------------------------------
--------------------------------------------------

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

### Examples
Start by unziping [data.zip](https://drive.google.com/file/d/1E7inIOtmOUvFoDT-QIs_87h9GSS5HHEB/view?usp=sharing) and [results.zip](https://drive.google.com/file/d/1f8PnU8XseDAHoAYwPqgYJNAFRlzymTmw/view?usp=sharing) in the project directory.
Note that you can either use the pretrained models or train your own deep ensembles from the scratch.
Use the following commands to run the experiments:

If you want to train the first mu model in the uana method for the multi_joint_robot experiment, you can use the following command:

- python main.py \--config configs/config1.json \--method uana \--step train_mu_model \--model_id 0

Or, if you want to invert the spectral_printer_noisy target data using na method, you can use the following command:

- python main.py \--config configs/config3.json \--method na \--step invert_target \--experiment_repetition_id 0 \--inversion_repetition_id 0 \--target_name evaluation

(if you want to invert the spectral_printer_noisy target data using na method for 50 times then you need to run above command with inversion_repetition_id from 0 to 49).

Or, if you want to aggregate results of the spectral_printer_sparse inverted targets using nae method, you can use the following command:

- python main.py \--config configs/config4.json \--method nae \--step aggregate_results \--experiment_repetition_id 0 \--target_name evaluation

Or, if you want to evaluate results of the spectral_printer_standard inverted targets using ua_tandem method, you can use the following command:

- python main.py \--config configs/config5.json \--method ua_tandem \--step evaluate_results \--experiment_repetition_id 0 \--target_name evaluation


### External libraries:
Please find the repositories of methods **MINI** and **INN** respectively [here](https://gitlab.mpi-klsb.mpg.de/nansari/mixed-integer-neural-inverse-design) and [here](https://github.com/BensonRen/BDIMNNA).
