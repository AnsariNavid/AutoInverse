import argparse
import json
from typing import Any

import src


def pars_arguments() -> Any:
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-m', '--method', type=str, required=True,
                        choices=['uana', 'na', 'nae', 'tandem'])
    parser.add_argument('-s', '--step', type=str, required=True)

    parser.add_argument('-n', '--model_id', type=int, required=False)
    parser.add_argument('-e', '--experiment_repetition_id', type=int, required=False)
    parser.add_argument('-i', '--inversion_repetition_id', type=int, required=False)

    args = parser.parse_args()

    if args.method == 'uana':
        if args.step == 'train_mu_model':
            if args.model_id is None:
                parser.error("--model_id is required.")

        elif args.step == 'train_sigma_model':
            if args.model_id is None:
                parser.error("--model_id is required.")

        elif args.step == 'inverse_design':
            if (args.experiment_repetition_id is None) or (args.inversion_repetition_id is None):
                parser.error("--experiment_repetition_id and --inversion_repetition_id are required.")

        elif args.step == 'aggregate_results':
            if args.experiment_repetition_id is None:
                parser.error("--experiment_repetition_id is required.")

        elif args.step == 'validate':
            if args.experiment_repetition_id is None:
                parser.error("--experiment_repetition_id is required.")

        else:
            parser.error("invalid --step.")

    if args.method == 'na':
        if args.step == 'train_forward_model':
            pass

        elif args.step == 'inverse_design':
            if (args.experiment_repetition_id is None) or (args.inversion_repetition_id is None):
                parser.error("--experiment_repetition_id and --inversion_repetition_id are required.")

        elif args.step == 'aggregate_results':
            if args.experiment_repetition_id is None:
                parser.error("--experiment_repetition_id is required.")

        elif args.step == 'validate':
            if args.experiment_repetition_id is None:
                parser.error("--experiment_repetition_id is required.")

        else:
            parser.error("invalid --step.")

    if args.method == 'nae':
        if args.step == 'inverse_design':
            if (args.experiment_repetition_id is None) or (args.inversion_repetition_id is None):
                parser.error("--experiment_repetition_id and --inversion_repetition_id are required.")

        elif args.step == 'aggregate_results':
            if args.experiment_repetition_id is None:
                parser.error("--experiment_repetition_id is required.")

        elif args.step == 'validate':
            if args.experiment_repetition_id is None:
                parser.error("--experiment_repetition_id is required.")

        else:
            parser.error("invalid --step.")

    if args.method == 'tandem':
        if args.step == 'train_backward_model':
            if args.experiment_repetition_id is None:
                parser.error("--experiment_repetition_id is required.")

        elif args.step == 'inverse_design':
            if args.experiment_repetition_id is None:
                parser.error("--experiment_repetition_id is required.")

        elif args.step == 'validate':
            if args.experiment_repetition_id is None:
                parser.error("--experiment_repetition_id is required.")

        else:
            parser.error("invalid --step.")

    if args.method == 'ua_tandem':
        if args.step == 'train_backward_model':
            if args.experiment_repetition_id is None:
                parser.error("--experiment_repetition_id is required.")

        elif args.step == 'inverse_design':
            if args.experiment_repetition_id is None:
                parser.error("--experiment_repetition_id is required.")

        elif args.step == 'validate':
            if args.experiment_repetition_id is None:
                parser.error("--experiment_repetition_id is required.")

        else:
            parser.error("invalid --step.")

    return args


if __name__ == '__main__':
    args = pars_arguments()
    config = src.utils.load_json_file(path=args.config)

    print(vars(args))
    print(json.dumps(config, sort_keys=False, indent=4))
    print()

    if args.method == 'uana':
        if args.step == 'train_mu_model':
            src.method.uana.train_mu_model(
                config=config,
                model_id=args.model_id
            )

        elif args.step == 'train_sigma_model':
            src.method.uana.train_sigma_model(
                config=config,
                model_id=args.model_id
            )

        elif args.step == 'inverse_design':
            src.method.uana.inverse_design(
                config=config,
                experiment_repetition_id=args.experiment_repetition_id,
                inversion_repetition_id=args.inversion_repetition_id
            )

        elif args.step == 'aggregate_results':
            src.method.uana.aggregate_results(
                config=config,
                experiment_repetition_id=args.experiment_repetition_id
            )

        elif args.step == 'validate':
            src.method.uana.validate(
                config=config,
                experiment_repetition_id=args.experiment_repetition_id
            )

    if args.method == 'na':
        if args.step == 'train_forward_model':
            src.method.na.train_forward_model(
                config=config
            )

        elif args.step == 'inverse_design':
            src.method.na.inverse_design(
                config=config,
                experiment_repetition_id=args.experiment_repetition_id,
                inversion_repetition_id=args.inversion_repetition_id
            )

        elif args.step == 'aggregate_results':
            src.method.na.aggregate_results(
                config=config,
                experiment_repetition_id=args.experiment_repetition_id
            )

        elif args.step == 'validate':
            src.method.na.validate(
                config=config,
                experiment_repetition_id=args.experiment_repetition_id
            )

    if args.method == 'nae':
        if args.step == 'inverse_design':
            src.method.nae.inverse_design(
                config=config,
                experiment_repetition_id=args.experiment_repetition_id,
                inversion_repetition_id=args.inversion_repetition_id
            )

        elif args.step == 'aggregate_results':
            src.method.nae.aggregate_results(
                config=config,
                experiment_repetition_id=args.experiment_repetition_id
            )

        elif args.step == 'validate':
            src.method.nae.validate(
                config=config,
                experiment_repetition_id=args.experiment_repetition_id
            )

    if args.method == 'tandem':
        if args.step == 'train_backward_model':
            src.method.tandem.train_backward_model(
                config=config,
                experiment_repetition_id=args.experiment_repetition_id
            )

        elif args.step == 'inverse_design':
            src.method.tandem.inverse_design(
                config=config,
                experiment_repetition_id=args.experiment_repetition_id
            )

        elif args.step == 'validate':
            src.method.tandem.validate(
                config=config,
                experiment_repetition_id=args.experiment_repetition_id
            )

    if args.method == 'ua_tandem':
        if args.step == 'train_backward_model':
            src.method.ua_tandem.train_backward_model(
                config=config,
                experiment_repetition_id=args.experiment_repetition_id
            )

        elif args.step == 'inverse_design':
            src.method.ua_tandem.inverse_design(
                config=config,
                experiment_repetition_id=args.experiment_repetition_id
            )

        elif args.step == 'validate':
            src.method.ua_tandem.validate(
                config=config,
                experiment_repetition_id=args.experiment_repetition_id
            )
