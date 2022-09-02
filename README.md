

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

*Please refer to the section **Help** for more information about each flag we used bellow.*<br>

*Please note that the term **Oracle** used in the code is equivalent to the term **NFP** in the paper.*<br>

#### Section 4.2 (Quantitative comparison of different neural inversion methods) - Table 1:

(description)<br>

**NA:**
```bash
# Step 1: train forward model
python main.py --config configs/config1.json --method na --step train_forward_model

# Step 2: invert target
for (( inversion_repetition_id = 0; inversion_repetition_id < 250; inversion_repetition_id++ ))
do
	python main.py --config configs/config1.json --method na --step invert_target --experiment_repetition_id 0 --inversion_repetition_id $inversion_repetition_id --target_name evaluation
done

# Step 3: aggregate results
python main.py --config configs/config1.json --method na --step aggregate_results --experiment_repetition_id 0 --target_name evaluation

# Step 4: evaluate results
python main.py --config configs/config1.json --method na --step evaluate_results --experiment_repetition_id 0 --target_name evaluation
```
**Results will be available in the following directory: results/multi_joint_robot/config1/na/inversion/evaluation/0/aggregated/evaluation_results.json**<br>

**UANA:**
```bash
# Step 1: train mu models
for (( model_id = 0; model_id < 10; model_id++ ))
do
	python main.py --config configs/config1.json --method uana --step train_mu_model --model_id $model_id
done

# Step 2: train sigma models
for (( model_id = 0; model_id < 10; model_id++ ))
do
	python main.py --config configs/config1.json --method uana --step train_sigma_model --model_id $model_id
done

# Step 3: invert target
for (( inversion_repetition_id = 0; inversion_repetition_id < 50; inversion_repetition_id++ ))
do
	python main.py --config configs/config1.json --method uana --step invert_target --experiment_repetition_id 0 --inversion_repetition_id $inversion_repetition_id --target_name evaluation
done

# Step 4: aggregate results
python main.py --config configs/config1.json --method uana --step aggregate_results --experiment_repetition_id 0 --target_name evaluation

# Step 5: evaluate results
python main.py --config configs/config1.json --method uana --step evaluate_results --experiment_repetition_id 0 --target_name evaluation
```
**Results will be available in the following directory: results/multi_joint_robot/config1/uana/inversion/evaluation/2_0.2/0/aggregated/evaluation_results.json**<br>

**Tandem:**
```bash
# Please note that we use NA forward model as the forward model in this method. So, if you have not done it yet it is required that you run step 1 of NA prior to this.

# Step 1: train backward model
python main.py --config configs/config1.json --method tandem --step train_backward_model --experiment_repetition_id 0

# Step 2: invert target
python main.py --config configs/config1.json --method tandem --step invert_target --experiment_repetition_id 0 --target_name evaluation

# Step 3: evaluate results
python main.py --config configs/config1.json --method tandem --step evaluate_results --experiment_repetition_id 0 --target_name evaluation
```
**Results will be available in the following directory: results/multi_joint_robot/config1/tandem/inversion/evaluation/0/evaluation_results.json**<br>

**UA-Tandem:**
```bash
# Please note that we use UANA mu and sigma models as the mu and sigma models in this method. So, if you have not done it yet it is required that you run step 1 and 2 of UANA prior to this.

# Step 1: train backward model
python main.py --config configs/config1.json --method ua_tandem --step train_backward_model --experiment_repetition_id 0

# Step 2: invert target
python main.py --config configs/config1.json --method ua_tandem --step invert_target --experiment_repetition_id 0 --target_name evaluation

# Step 3: evaluate results
python main.py --config configs/config1.json --method ua_tandem --step evaluate_results --experiment_repetition_id 0 --target_name evaluation
```
**Results will be available in the following directory: results/multi_joint_robot/config1/ua_tandem/inversion/evaluation/0.5_0.05/0/evaluation_results.json**<br>

#### Section 4.3 (Neural inversion in the presence of imperfect data) - Table 2:

(description)<br>

**NA-Standard:**
```bash
# Step 1: train forward model
python main.py --config configs/config5.json --method na --step train_forward_model

# Step 2: invert target
for (( inversion_repetition_id = 0; inversion_repetition_id < 250; inversion_repetition_id++ ))
do
	python main.py --config configs/config5.json --method na --step invert_target --experiment_repetition_id 0 --inversion_repetition_id $inversion_repetition_id --target_name evaluation
done

# Step 3: aggregate results
python main.py --config configs/config5.json --method na --step aggregate_results --experiment_repetition_id 0 --target_name evaluation

# Step 4: evaluate results
python main.py --config configs/config5.json --method na --step evaluate_results --experiment_repetition_id 0 --target_name evaluation
```
**Results will be available in the following directory: results/spectral_printer_standard/config5/na/inversion/evaluation/0/aggregated/evaluation_results.json**<br>

**NA-Sparse:**
```bash
# Step 1: train forward model
python main.py --config configs/config4.json --method na --step train_forward_model

# Step 2: invert target
for (( inversion_repetition_id = 0; inversion_repetition_id < 250; inversion_repetition_id++ ))
do
	python main.py --config configs/config4.json --method na --step invert_target --experiment_repetition_id 0 --inversion_repetition_id $inversion_repetition_id --target_name evaluation
done

# Step 3: aggregate results
python main.py --config configs/config4.json --method na --step aggregate_results --experiment_repetition_id 0 --target_name evaluation

# Step 4: evaluate results
python main.py --config configs/config4.json --method na --step evaluate_results --experiment_repetition_id 0 --target_name evaluation
```
**Results will be available in the following directory: results/spectral_printer_sparse/config4/na/inversion/evaluation/0/aggregated/evaluation_results.json**<br>

**NA-Noisy:**
```bash
# Step 1: train forward model
python main.py --config configs/config3.json --method na --step train_forward_model

# Step 2: invert target
for (( inversion_repetition_id = 0; inversion_repetition_id < 250; inversion_repetition_id++ ))
do
	python main.py --config configs/config3.json --method na --step invert_target --experiment_repetition_id 0 --inversion_repetition_id $inversion_repetition_id --target_name evaluation
done

# Step 3: aggregate results
python main.py --config configs/config3.json --method na --step aggregate_results --experiment_repetition_id 0 --target_name evaluation

# Step 4: evaluate results
python main.py --config configs/config3.json --method na --step evaluate_results --experiment_repetition_id 0 --target_name evaluation
```
**Results will be available in the following directory: results/spectral_printer_noisy/config3/na/inversion/evaluation/0/aggregated/evaluation_results.json**<br>

**UANA-Standard:**
```bash
# Step 1: train mu models
for (( model_id = 0; model_id < 10; model_id++ ))
do
	python main.py --config configs/config5.json --method uana --step train_mu_model --model_id $model_id
done

# Step 2: train sigma models
for (( model_id = 0; model_id < 10; model_id++ ))
do
	python main.py --config configs/config5.json --method uana --step train_sigma_model --model_id $model_id
done

# Step 3: invert target
for (( inversion_repetition_id = 0; inversion_repetition_id < 50; inversion_repetition_id++ ))
do
	python main.py --config configs/config5.json --method uana --step invert_target --experiment_repetition_id 0 --inversion_repetition_id $inversion_repetition_id --target_name evaluation
done

# Step 4: aggregate results
python main.py --config configs/config5.json --method uana --step aggregate_results --experiment_repetition_id 0 --target_name evaluation

# Step 5: evaluate results
python main.py --config configs/config5.json --method uana --step evaluate_results --experiment_repetition_id 0 --target_name evaluation
```
**Results will be available in the following directory: results/spectral_printer_standard/config5/uana/inversion/evaluation/100_10/0/aggregated/evaluation_results.json**<br>

**UANA-Sparse:**
```bash
# Step 1: train mu models
for (( model_id = 0; model_id < 10; model_id++ ))
do
	python main.py --config configs/config4.json --method uana --step train_mu_model --model_id $model_id
done

# Step 2: train sigma models
for (( model_id = 0; model_id < 10; model_id++ ))
do
	python main.py --config configs/config4.json --method uana --step train_sigma_model --model_id $model_id
done

# Step 3: invert target
for (( inversion_repetition_id = 0; inversion_repetition_id < 50; inversion_repetition_id++ ))
do
	python main.py --config configs/config4.json --method uana --step invert_target --experiment_repetition_id 0 --inversion_repetition_id $inversion_repetition_id --target_name evaluation
done

# Step 4: aggregate results
python main.py --config configs/config4.json --method uana --step aggregate_results --experiment_repetition_id 0 --target_name evaluation

# Step 5: evaluate results
python main.py --config configs/config4.json --method uana --step evaluate_results --experiment_repetition_id 0 --target_name evaluation
```
**Results will be available in the following directory: results/spectral_printer_sparse/config4/uana/inversion/evaluation/100_10/0/aggregated/evaluation_results.json**<br>

**UANA-Noisy:**
```bash
# Step 1: train mu models
for (( model_id = 0; model_id < 10; model_id++ ))
do
	python main.py --config configs/config3.json --method uana --step train_mu_model --model_id $model_id
done

# Step 2: train sigma models
for (( model_id = 0; model_id < 10; model_id++ ))
do
	python main.py --config configs/config3.json --method uana --step train_sigma_model --model_id $model_id
done

# Step 3: invert target
for (( inversion_repetition_id = 0; inversion_repetition_id < 50; inversion_repetition_id++ ))
do
	python main.py --config configs/config3.json --method uana --step invert_target --experiment_repetition_id 0 --inversion_repetition_id $inversion_repetition_id --target_name evaluation
done

# Step 4: aggregate results
python main.py --config configs/config3.json --method uana --step aggregate_results --experiment_repetition_id 0 --target_name evaluation

# Step 5: evaluate results
python main.py --config configs/config3.json --method uana --step evaluate_results --experiment_repetition_id 0 --target_name evaluation
```
**Results will be available in the following directory: results/spectral_printer_noisy/config3/uana/inversion/evaluation/50_5/0/aggregated/evaluation_results.json**<br>

#### Section C.1 (Counterpart results of tandem and UA-tandem for spectral printer) - Table 5:

(description)<br>

**Tandem-Standard:**
```bash
# Please note that we use NA-Standard forward model as the forward model in this method. So, if you have not done it yet it is required that you run step 1 of NA-Standard prior to this.

# Step 1: train backward model
python main.py --config configs/config5.json --method tandem --step train_backward_model --experiment_repetition_id 0

# Step 2: invert target
python main.py --config configs/config5.json --method tandem --step invert_target --experiment_repetition_id 0 --target_name evaluation

# Step 3: evaluate results
python main.py --config configs/config5.json --method tandem --step evaluate_results --experiment_repetition_id 0 --target_name evaluation
```
**Results will be available in the following directory: results/spectral_printer_standard/config5/tandem/inversion/evaluation/0/evaluation_results.json**<br>

**Tandem-Sparse:**
```bash
# Please note that we use NA-Sparse forward model as the forward model in this method. So, if you have not done it yet it is required that you run step 1 of NA-Sparse prior to this.

# Step 1: train backward model
python main.py --config configs/config4.json --method tandem --step train_backward_model --experiment_repetition_id 0

# Step 2: invert target
python main.py --config configs/config4.json --method tandem --step invert_target --experiment_repetition_id 0 --target_name evaluation

# Step 3: evaluate results
python main.py --config configs/config4.json --method tandem --step evaluate_results --experiment_repetition_id 0 --target_name evaluation
```
**Results will be available in the following directory: results/spectral_printer_sparse/config4/tandem/inversion/evaluation/0/evaluation_results.json**<br>

**Tandem-Noisy:**
```bash
# Please note that we use NA-Noisy forward model as the forward model in this method. So, if you have not done it yet it is required that you run step 1 of NA-Noisy prior to this.

# Step 1: train backward model
python main.py --config configs/config3.json --method tandem --step train_backward_model --experiment_repetition_id 0

# Step 2: invert target
python main.py --config configs/config3.json --method tandem --step invert_target --experiment_repetition_id 0 --target_name evaluation

# Step 3: evaluate results
python main.py --config configs/config3.json --method tandem --step evaluate_results --experiment_repetition_id 0 --target_name evaluation
```
**Results will be available in the following directory: results/spectral_printer_noisy/config3/tandem/inversion/evaluation/0/evaluation_results.json**<br>

**UA-Tandem-Standard:**
```bash
# Please note that we use UANA-Standard mu and sigma models as the mu and sigma models in this method. So, if you have not done it yet it is required that you run step 1 and 2 of UANA-Standard prior to this.

# Step 1: train backward model
python main.py --config configs/config5.json --method ua_tandem --step train_backward_model --experiment_repetition_id 0

# Step 2: invert target
python main.py --config configs/config5.json --method ua_tandem --step invert_target --experiment_repetition_id 0 --target_name evaluation

# Step 3: evaluate results
python main.py --config configs/config5.json --method ua_tandem --step evaluate_results --experiment_repetition_id 0 --target_name evaluation
```
**Results will be available in the following directory: results/spectral_printer_standard/config5/ua_tandem/inversion/evaluation/0.5_0.05/0/evaluation_results.json**<br>

**UA-Tandem-Sparse:**
```bash
# Please note that we use UANA-Sparse mu and sigma models as the mu and sigma models in this method. So, if you have not done it yet it is required that you run step 1 and 2 of UANA-Sparse prior to this.

# Step 1: train backward model
python main.py --config configs/config4.json --method ua_tandem --step train_backward_model --experiment_repetition_id 0

# Step 2: invert target
python main.py --config configs/config4.json --method ua_tandem --step invert_target --experiment_repetition_id 0 --target_name evaluation

# Step 3: evaluate results
python main.py --config configs/config4.json --method ua_tandem --step evaluate_results --experiment_repetition_id 0 --target_name evaluation
```
**Results will be available in the following directory: results/spectral_printer_sparse/config4/ua_tandem/inversion/evaluation/2_0.2/0/evaluation_results.json**<br>

**UA-Tandem-Noisy:**
```bash
# Please note that we use UANA-Noisy mu and sigma models as the mu and sigma models in this method. So, if you have not done it yet it is required that you run step 1 and 2 of UANA-Noisy prior to this.

# Step 1: train backward model
python main.py --config configs/config3.json --method ua_tandem --step train_backward_model --experiment_repetition_id 0

# Step 2: invert target
python main.py --config configs/config3.json --method ua_tandem --step invert_target --experiment_repetition_id 0 --target_name evaluation

# Step 3: evaluate results
python main.py --config configs/config3.json --method ua_tandem --step evaluate_results --experiment_repetition_id 0 --target_name evaluation
```
**Results will be available in the following directory: results/spectral_printer_noisy/config3/ua_tandem/inversion/evaluation/10_1/0/evaluation_results.json**<br>

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

### External libraries:
Please find the repositories of methods **MINI** and **INN** respectively [here](https://gitlab.mpi-klsb.mpg.de/nansari/mixed-integer-neural-inverse-design) and [here](https://github.com/BensonRen/BDIMNNA).
