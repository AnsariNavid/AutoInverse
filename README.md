# AutoInverse

## Uncertainty Aware Inversion of Neural Networks

Research code for **Autoinverse: Uncertainty Aware Inversion of Neural Networks**, published at **[NeurIPS 2022, Spotlight](https://aidam.mpi-inf.mpg.de/?view=publications)**.

**Navid Ansari, Hans-Peter Seidel, Nima Vahidi Ferdowsi, and Vahid Babaei**

[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3942d2f7fc0a5af1bf70dcbdcf4c56aa-Abstract-Conference.html) · [arXiv](https://arxiv.org/abs/2208.13780) · [Supplementary material](https://proceedings.neurips.cc/paper_files/paper/2022/file/3942d2f7fc0a5af1bf70dcbdcf4c56aa-Supplemental-Conference.pdf)

[Getting started](#getting-started) · [Methods](#methods-and-prerequisites) · [Experiments](#experiment-configurations) · [Outputs](#outputs) · [Citation](#citation)

## Overview

Inverse design starts with a desired outcome and searches for the parameters that produce it. Examples include finding a robot configuration that reaches a target position or choosing ink densities that reproduce a target spectrum. Neural networks can approximate these forward processes and make the search efficient. However, a design that performs well under a learned surrogate may fail under the original simulator or physical process, particularly where the training data are sparse or noisy.

AutoInverse incorporates predictive uncertainty into the inverse problem. A deep ensemble estimates the expected response, uncertainty associated with noise in the observations, and uncertainty associated with disagreement between models. Inversion then balances matching the target with reducing these uncertainties, encouraging solutions in regions supported by reliable training data.

In the paper's multi-joint robot experiment, UANA reduces the mean error on the native forward process from **3.24 × 10⁻⁴ to 3.21 × 10⁻⁶** relative to neural adjoint, approximately two orders of magnitude across 1,000 target positions. See [Table 1](https://proceedings.neurips.cc/paper_files/paper/2022/file/3942d2f7fc0a5af1bf70dcbdcf4c56aa-Paper-Conference.pdf#page=6) for the reported comparison.

The paper studies this approach in **robotic inverse kinematics**, **spectral printing**, and **soft robot control**. This repository provides the neural inversion methods, experiment configurations, and evaluation code for the multi-joint robot and spectral printer. The soft robot configurations are partially implemented; see [implementation notes](#implementation-notes).

For the standard target-matching loss, the implementation uses this mean-reduced form of the paper's objective:

$$
\mathcal{L}(x; y^\star)
= \operatorname{MSE}\bigl(\bar{\mu}(x), y^\star\bigr)
+ \lambda_{\mathrm{epi}}\operatorname{mean}\bigl(v_{\mathrm{epi}}(x)\bigr)
+ \lambda_{\mathrm{ale}}\operatorname{mean}\bigl(v_{\mathrm{ale}}(x)\bigr),
$$

where $x$ is the design, $y^\star$ is the target, and $\bar{\mu}$ is the ensemble prediction. Epistemic uncertainty $v_{\mathrm{epi}}$ measures disagreement between predicted means; aleatoric uncertainty $v_{\mathrm{ale}}$ averages the predicted variances. The configuration keys are `epistemic_weight` and `aleatory_weight`.

**Terminology:** `oracle` in the code refers to the **native forward process (NFP)** used for evaluation in the paper. For spectral printing, this reference process is a separate pretrained printer simulator.

## Getting started

### 1. Install the dependencies

The Python code uses **PyTorch**, **NumPy**, and **SciPy**. The commands below use Bash and should be run from the repository root after cloning.

```bash
git clone --branch refactor1 https://github.com/AnsariNavid/AutoInverse.git
cd AutoInverse

python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy
```

Install **PyTorch** using the [official installation selector](https://pytorch.org/get-started/locally/), choosing a build appropriate for your operating system and hardware. Then check the installation and available commands:

```bash
python -c "import torch, numpy, scipy; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python main.py --help
```

Device selection is automatic in [src/settings.py](src/settings.py): CUDA is used when available, otherwise CPU. A CUDA-capable GPU is recommended for ensemble training; see the [spectral printer evaluation note](#implementation-notes) when using one. This repository does not pin dependency versions. Record the Python, PyTorch, NumPy, and SciPy versions used for reproducible experiments.

### 2. Download and organize the data

The datasets and inversion checkpoints are distributed separately from the source code:

| Archive | Purpose |
| --- | --- |
| [data.zip](https://drive.google.com/file/d/1RQORP1mfD37XVs3IXLl_2eX13qb_DRVS/view?usp=sharing) | Training design/response pairs and target data. |
| [results.zip](https://drive.google.com/file/d/1oPBXO9ADnMSZxek9dszNl_92mAQ0efa9/view?usp=sharing) | Optional pretrained networks and saved experiment outputs. |

Download and extract the archives into the project directory. The resulting top-level directories must be `data/` and, when using pretrained models, `results/`. If extraction creates an additional enclosing directory, move its contents so that the paths match the configuration files.

These are the primary download links from the original README. Check the extracted files against the configuration paths below before running an experiment.

<details>
<summary>Additional archive links from the original README</summary>

The original README also listed [this data archive](https://drive.google.com/file/d/1E7inIOtmOUvFoDT-QIs_87h9GSS5HHEB/view?usp=sharing) and [this results archive](https://drive.google.com/file/d/1f8PnU8XseDAHoAYwPqgYJNAFRlzymTmw/view?usp=sharing). Their contents have not been verified as equivalent to the downloads above. Check the extracted paths and checkpoint names before substituting them.

</details>

For the multi-joint robot example, [configs/config1.json](configs/config1.json) expects:

| Path | Contents |
| --- | --- |
| `data/multi_joint_robot/design.mat` | Training designs, with four parameters per sample. |
| `data/multi_joint_robot/performance.mat` | Corresponding two-dimensional endpoint positions. |
| `data/multi_joint_robot/tuning_target.mat` | Targets for selecting hyperparameters. |
| `data/multi_joint_robot/evaluation_target.mat` | Targets for evaluation. |

Each MATLAB file is loaded using the variable name **`data`**. Samples occupy rows. Training design and response arrays must have the same number of rows, with column counts matching `design_size` and `performance_size` in the configuration. For `config1.json`, these shapes are `N × 4` and `N × 2`, and target arrays have two columns. Store targets as MATLAB `double` / NumPy `float64` for compatibility with the current evaluation code.

To use pretrained models, preserve the hierarchy `results/<experiment_name>/<config_name>/...`. In particular, UANA requires both the variance models and the fine-tuned mean models. The printer simulator checkpoints in `src/oracle/spectral_printer_oracle_models/` are already included in the repository; these are separate from the inversion models in `results/`.

### 3. Run UANA on the multi-joint robot

UANA is the uncertainty-aware neural adjoint method: it optimizes the design directly through the pretrained ensemble.

#### Train the ensemble, or use compatible pretrained checkpoints

The supplied configurations use ten ensemble members, indexed from `0` to `9`. First train the mean models, then train the variance models. The second stage also fine-tunes each corresponding mean model.

```bash
for model_id in {0..9}; do
    python main.py --config configs/config1.json --method uana \
        --step train_mu_model --model_id "$model_id"
done

for model_id in {0..9}; do
    python main.py --config configs/config1.json --method uana \
        --step train_sigma_model --model_id "$model_id"
done
```

Skip both loops only if the matching checkpoints are already present. For each member `i`, UANA loads:

- `results/multi_joint_robot/config1/uana/mu_model/i/model_i_fine_tuned.pt`
- `results/multi_joint_robot/config1/uana/sigma_model/i/model_i.pt`

Here `i` denotes the actual member index. When training a custom ensemble, the number of members is determined by the length of `uana.mu_model.activation_functions`.

#### Invert, aggregate, and evaluate

This example performs one randomly initialized inversion for every evaluation target, selects the available solution for each target, and evaluates it using the analytical robot forward process. If you have already generated or extracted inversion results, replace `--experiment_repetition_id 0` with an unused ID in every command below.

```bash
python main.py --config configs/config1.json --method uana \
    --step invert_target --experiment_repetition_id 0 \
    --inversion_repetition_id 0 --target_name evaluation

python main.py --config configs/config1.json --method uana \
    --step aggregate_results --experiment_repetition_id 0 \
    --target_name evaluation

python main.py --config configs/config1.json --method uana \
    --step evaluate_results --experiment_repetition_id 0 \
    --target_name evaluation
```

With the unchanged configuration, the evaluation summary is written to:

```text
results/multi_joint_robot/config1/uana/inversion/evaluation/2_0.2/0/aggregated/evaluation_results.json
```

The directory component `2_0.2` records the epistemic and aleatoric weights. A single inversion is an introductory run; the original experiment instructions use multiple initializations.

#### Repeat the inversion

For the original 50-initialization UANA workflow, replace the single `invert_target` command with:

```bash
for inversion_id in {0..49}; do
    python main.py --config configs/config1.json --method uana \
        --step invert_target --experiment_repetition_id 0 \
        --inversion_repetition_id "$inversion_id" --target_name evaluation
done
```

Then run `aggregate_results` and `evaluate_results` as above. Aggregation selects the lowest per-target inversion loss across all saved initializations; UANA's selection loss includes the uncertainty penalties. Keep inversion IDs consecutive from `0`, as the aggregation implementation assumes this numbering.

Use a new `experiment_repetition_id` to keep a separate set of inversion outputs. Reusing an existing ID can overwrite that initialization's files, and aggregation includes all numbered initialization directories already present for the selected run.

## Methods and prerequisites

| CLI method | Description | Required models before inversion |
| --- | --- | --- |
| `na` | Neural adjoint using one forward surrogate. | Train `na / train_forward_model`. |
| `nae` | Neural adjoint using an ensemble, without uncertainty penalties. | Train every `uana / train_mu_model`; NAE loads the original mean checkpoints. |
| `uana` | Neural adjoint with epistemic and aleatoric uncertainty penalties. | Train all mean models, then all variance models. |
| `tandem` | An inverse network trained through a forward surrogate. | Train the NA forward model, then `tandem / train_backward_model`. |
| `ua_tandem` | An inverse network trained with the uncertainty-aware ensemble loss. | Complete both UANA training stages, then `ua_tandem / train_backward_model`. |

Compatible pretrained checkpoints can replace the corresponding training steps. NA, NAE, and UANA require `aggregate_results` before `evaluate_results`, even for a single initialization. Tandem methods predict designs in a forward pass and have no aggregation step or `inversion_repetition_id` argument requirement.

### NA and NAE

Train the NA surrogate with:

```bash
python main.py --config configs/config1.json --method na \
    --step train_forward_model
```

Then follow the inversion, aggregation, and evaluation sequence above with `--method na`. The original README specifies 250 NA initializations, using IDs `0` through `249`.

For NAE, train all ten mean models using `uana / train_mu_model`, then use `--method nae` in the same three-step sequence. NAE does not have its own training command and does not require the variance training stage.

### UA-tandem

After completing both UANA ensemble training stages:

```bash
python main.py --config configs/config1.json --method ua_tandem \
    --step train_backward_model --experiment_repetition_id 0

python main.py --config configs/config1.json --method ua_tandem \
    --step invert_target --experiment_repetition_id 0 --target_name evaluation

python main.py --config configs/config1.json --method ua_tandem \
    --step evaluate_results --experiment_repetition_id 0 --target_name evaluation
```

For standard tandem, first train the NA forward model, then use the same three commands with `--method tandem`.

## Experiment configurations

| Configuration | Experiment | Design / response dimensions | Implementation status |
| --- | --- | --- | --- |
| [config1.json](configs/config1.json) | Multi-joint robot | 4 / 2 | All five methods; analytical NFP evaluation. |
| [config3.json](configs/config3.json) | Spectral printer, noisy data | 8 / 31 | NA, UANA, tandem, UA-tandem; evaluation includes noise in the affected ink region. |
| [config4.json](configs/config4.json) | Spectral printer, sparse data | 8 / 31 | NA, UANA, tandem, UA-tandem; standard printer reference model. |
| [config5.json](configs/config5.json) | Spectral printer, standard data | 8 / 31 | NA, UANA, tandem, UA-tandem; standard printer reference model. |
| [config6.json](configs/config6.json) | Spectral printer, sparse data, ReLU ensemble | 8 / 31 | UANA only; activation-function comparison. |
| [config8.json](configs/config8.json) | Soft robot, noisy data | 40 / 206 | Partial configuration; see notes below. |
| [config9.json](configs/config9.json) | Soft robot, sparse data | 40 / 206 | Partial configuration; see notes below. |
| [config10.json](configs/config10.json) | Soft robot, standard data | 40 / 206 | Forward-model training configuration. |

The multi-joint robot designs contain a base translation and three joint angles. Printer responses contain 31 spectral samples. Soft-robot responses contain 206 shape coordinates; the supplied soft-robot loss evaluates the two tip coordinates at indices `122:124`.

The multi-joint robot experiment corresponds to Section 4.2 and Table 1 of the paper. The spectral printing configurations cover the standard, sparse, and noisy settings in Section 4.3 and Table 2; the tandem counterparts are reported in Supplementary Section C.1 and Table 5. Reproducing a table requires its complete protocol, including repetitions and external baselines, beyond the introductory run above.

For spectral printing, change `--config` in the workflows to the relevant file and use that configuration consistently for training, inversion, aggregation, and evaluation. Configurations 3, 4, and 5 omit the `nae.inversion` section, and configuration 6 contains UANA settings only. Compare configurations 4 and 6 for the activation-diversity experiment in Section 4.5. Models and outputs are separated by `experiment_name` and `config_name`.

Use `--target_name tuning` for hyperparameter selection and `--target_name evaluation` for evaluation. These select `tuning_path` and `evaluation_path`, respectively. Keep the selected target set consistent across inversion, aggregation, and evaluation. To generate tuning outputs, run these steps with `tuning`; evaluation outputs are not reused automatically. The original release notes state that tuning outputs are not included in the results archive.

## Outputs

All experiment outputs are written below:

```text
<results_base_path>/<experiment_name>/<config_name>/<method>/
```

The configuration defaults to `results` for `results_base_path`. [src/utils.py](src/utils.py) defines the full paths, including target sets, uncertainty weights, experiment IDs, and initialization IDs.

| Output | Meaning |
| --- | --- |
| `model.pt`, `model_i.pt`, `model_i_fine_tuned.pt` | Model state dictionaries, with names determined by the training stage. |
| `info.json` | Training or iterative-inversion timing and loss information. |
| `design.mat` | Inferred design parameters. |
| `performance.mat` | Forward surrogate predictions recorded during inversion. |
| `sigma_epistemic.mat`, `sigma_aleatory.mat` | Uncertainty estimates saved by UANA and UA-tandem. |
| `no_reduction_loss.mat` | Per-target loss used to select among iterative inversions. |
| `aggregated/` | Selected designs and predictions for NA, NAE, and UANA. |
| `repetition_id_per_element.mat` | Selected restart for each target after aggregation. |
| `evaluation_results.json` | Error measured using the configured reference process. |
| `oracle_performance.mat` | Spectral printer reference predictions. |

For the multi-joint robot, `evaluation_results.json` contains both `nfp_error` and `surrogate_error`. For spectral printing, it contains `nfp_error` and `ink_density`, which counts how many designs use each ink at density at least `0.4`. The counts describe ink use at that threshold. Printer evaluation clips inferred densities to `[0, 1]` before calling the reference model, so inspect the raw `design.mat` when assessing whether inversion itself produced in-range designs. Surrogate error and NFP error measure different aspects of the inverse solution; NFP error is the relevant check against the reference process.

## Code organization

| Path | Role |
| --- | --- |
| [main.py](main.py) | Command-line interface and dispatch to method stages. |
| [configs/](configs/) | Dataset paths, network architectures, and optimization settings. |
| [src/method/](src/method/) | Training, inversion, aggregation, and evaluation workflows. |
| [src/model/](src/model/) | Model construction and checkpoint loading. |
| [src/train/](src/train/) | Model training loops. |
| [src/test/](src/test/) | Loss evaluation during training. |
| [src/inversion/](src/inversion/) | Direct design optimization and inverse-network prediction. |
| [src/loss/](src/loss/) | Target-matching, uncertainty, and soft robot losses. |
| [src/oracle/](src/oracle/) | Reference processes for the multi-joint robot and spectral printer. |
| [src/utils.py](src/utils.py) | Data I/O, checkpoint handling, and output paths. |
| [src/settings.py](src/settings.py) | Device selection and numerical/logging settings. |

### Command-line reference

| Argument | Purpose |
| --- | --- |
| `--config`, `-c` | JSON configuration path. Required. |
| `--method`, `-m` | `na`, `nae`, `uana`, `tandem`, or `ua_tandem`. Required. |
| `--step`, `-s` | Stage to execute. Valid stages depend on the method. Required. |
| `--model_id`, `-n` | Ensemble member index for mean or variance training. |
| `--experiment_repetition_id`, `-e` | Output group for an inversion experiment; also identifies a tandem inverse model. |
| `--inversion_repetition_id`, `-i` | Initialization index for NA, NAE, or UANA. Use consecutive IDs from `0`. |
| `--target_name`, `-t` | `tuning` or `evaluation`. Required for inversion, aggregation, and evaluation. |

Changing `experiment_repetition_id` does not retrain or separate NA/UANA forward checkpoints. To retain independently trained ensembles, use a distinct `config_name` or `results_base_path` and train the required models there.

To adapt the method to another problem, begin with a working configuration and update the paired data paths, design/response dimensions, model settings, and target objective. Add an evaluator under `src/oracle/` and export it in `src/oracle/__init__.py` to use `evaluate_results`. The [configuration template](configs/config.template.json) provides the field structure; fill its placeholder values before use.

## Implementation notes

These notes describe the current `refactor1` implementation.

- **Soft robot experiments are incomplete in this branch.** Configs 8 and 9 omit direct-inversion settings and have an empty `oracle` field. Config 10 also has empty target paths. The finite-element evaluation process is not included in `src/oracle/`, so the supplied configurations do not reproduce the complete soft robot pipeline without additional integration.
- **Spectral printer evaluation needs CPU execution or a device correction.** Both printer evaluators create design tensors on CPU but place models on `settings.DEVICE`. On a CUDA machine, run the evaluation command with the Bash prefix `CUDA_VISIBLE_DEVICES=""`, or move the design tensors to the same device as the models in the evaluator.
- **Evaluation data types:** a NumPy `float32` error value can fail when written to `evaluation_results.json`. Store target matrices as MATLAB `double` / NumPy `float64`, or convert computed metrics to Python `float` in the evaluator.
- **Randomness is not fixed by the CLI.** Training splits, network initialization, inversion initialization, and noisy printer evaluation can vary between runs. Repetition IDs organize files; they do not set random seeds.
- **Uncertainty provides a soft preference.** Its effectiveness depends on the learned ensemble and the chosen weights. The reported improvements in feasibility and robustness are empirical, and physical or simulator evaluation remains necessary for assessing a design.

## External baselines

The paper also compares against methods maintained in separate repositories:

- [MINI: Mixed-Integer Neural Inverse Design](https://gitlab.mpi-klsb.mpg.de/nansari/mixed-integer-neural-inverse-design)
- [INN implementation used in the comparison](https://github.com/BensonRen/BDIMNNA)

These baselines are not available through this repository's `--method` argument.

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{ansari2022autoinverse,
  title     = {Autoinverse: Uncertainty Aware Inversion of Neural Networks},
  author    = {Ansari, Navid and Seidel, Hans-Peter and
               Vahidi Ferdowsi, Nima and Babaei, Vahid},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {35},
  pages     = {8675--8686},
  publisher = {Curran Associates, Inc.},
  year      = {2022},
  doi       = {10.52202/068431-0631},
  url       = {https://proceedings.neurips.cc/paper_files/paper/2022/hash/3942d2f7fc0a5af1bf70dcbdcf4c56aa-Abstract-Conference.html}
}
```
