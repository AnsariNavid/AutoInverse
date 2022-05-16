import argparse
import json
from typing import Any

import src


def pars_arguments() -> Any:
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-m', '--method', type=str, required=True,
                        choices=['uana', 'na', 'nae', 'tandem', 'ua_tandem'])
    parser.add_argument('-s', '--step', type=str, required=True)

    parser.add_argument('-n', '--model_id', type=int, required=False)
    parser.add_argument('-e', '--experiment_repetition_id', type=int, required=False)
    parser.add_argument('-i', '--inversion_repetition_id', type=int, required=False)
    parser.add_argument('-t', '--target_name', type=str, required=False, choices=['tuning', 'evaluation'])

    args = parser.parse_args()

    if args.method == 'uana':
        if args.step == 'train_mu_model':
            if args.model_id is None:
                parser.error("--model_id is required.")

        elif args.step == 'train_sigma_model':
            if args.model_id is None:
                parser.error("--model_id is required.")

        elif args.step == 'invert_target':
            if (args.experiment_repetition_id is None) or (args.inversion_repetition_id is None) or\
                 (args.target_name is None):
                parser.error("--experiment_repetition_id and --inversion_repetition_id and --target_name are required.")

        elif args.step == 'aggregate_results':
            if (args.experiment_repetition_id is None) or (args.target_name is None):
                parser.error("--experiment_repetition_id and --target_name are required.")

        elif args.step == 'evaluate_results':
            if (args.experiment_repetition_id is None) or (args.target_name is None):
                parser.error("--experiment_repetition_id and --target_name are required.")

        else:
            parser.error("invalid --step.")

    if args.method == 'na':
        if args.step == 'train_forward_model':
            pass

        elif args.step == 'invert_target':
            if (args.experiment_repetition_id is None) or (args.inversion_repetition_id is None) or\
                 (args.target_name is None):
                parser.error("--experiment_repetition_id and --inversion_repetition_id and --target_name are required.")

        elif args.step == 'aggregate_results':
            if (args.experiment_repetition_id is None) or (args.target_name is None):
                parser.error("--experiment_repetition_id and --target_name are required.")

        elif args.step == 'evaluate_results':
            if (args.experiment_repetition_id is None) or (args.target_name is None):
                parser.error("--experiment_repetition_id and --target_name are required.")

        else:
            parser.error("invalid --step.")

    if args.method == 'nae':
        if args.step == 'invert_target':
            if (args.experiment_repetition_id is None) or (args.inversion_repetition_id is None)\
                 or (args.target_name is None):
                parser.error("--experiment_repetition_id and --inversion_repetition_id and --target_name are required.")

        elif args.step == 'aggregate_results':
            if (args.experiment_repetition_id is None) or (args.target_name is None):
                parser.error("--experiment_repetition_id and --target_name are required.")

        elif args.step == 'evaluate_results':
            if (args.experiment_repetition_id is None) or (args.target_name is None):
                parser.error("--experiment_repetition_id and --target_name are required.")

        else:
            parser.error("invalid --step.")

    if args.method == 'tandem':
        if args.step == 'train_backward_model':
            if args.experiment_repetition_id is None:
                parser.error("--experiment_repetition_id is required.")

        elif args.step == 'invert_target':
            if (args.experiment_repetition_id is None) or (args.target_name is None):
                parser.error("--experiment_repetition_id and --target_name are required.")

        elif args.step == 'evaluate_results':
            if (args.experiment_repetition_id is None) or (args.target_name is None):
                parser.error("--experiment_repetition_id and --target_name are required.")

        else:
            parser.error("invalid --step.")

    if args.method == 'ua_tandem':
        if args.step == 'train_backward_model':
            if args.experiment_repetition_id is None:
                parser.error("--experiment_repetition_id is required.")

        elif args.step == 'invert_target':
            if (args.experiment_repetition_id is None) or (args.target_name is None):
                parser.error("--experiment_repetition_id and --target_name are required.")

        elif args.step == 'evaluate_results':
            if (args.experiment_repetition_id is None) or (args.target_name is None):
                parser.error("--experiment_repetition_id and --target_name are required.")

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

        elif args.step == 'invert_target':
            src.method.uana.invert_target(
                config=config,
                experiment_repetition_id=args.experiment_repetition_id,
                inversion_repetition_id=args.inversion_repetition_id,
                target_name=args.target_name
            )

        elif args.step == 'aggregate_results':
            src.method.uana.aggregate_results(
                config=config,
                experiment_repetition_id=args.experiment_repetition_id,
                target_name=args.target_name
            )

        elif args.step == 'evaluate_results':
            src.method.uana.evaluate_results(
                config=config,
                experiment_repetition_id=args.experiment_repetition_id,
                target_name=args.target_name
            )

    if args.method == 'na':
        if args.step == 'train_forward_model':
            src.method.na.train_forward_model(
                config=config
            )

        elif args.step == 'invert_target':
            src.method.na.invert_target(
                config=config,
                experiment_repetition_id=args.experiment_repetition_id,
                inversion_repetition_id=args.inversion_repetition_id,
                target_name=args.target_name
            )

        elif args.step == 'aggregate_results':
            src.method.na.aggregate_results(
                config=config,
                experiment_repetition_id=args.experiment_repetition_id,
                target_name=args.target_name
            )

        elif args.step == 'evaluate_results':
            src.method.na.evaluate_results(
                config=config,
                experiment_repetition_id=args.experiment_repetition_id,
                target_name=args.target_name
            )

    if args.method == 'nae':
        if args.step == 'invert_target':
            src.method.nae.invert_target(
                config=config,
                experiment_repetition_id=args.experiment_repetition_id,
                inversion_repetition_id=args.inversion_repetition_id,
                target_name=args.target_name
            )

        elif args.step == 'aggregate_results':
            src.method.nae.aggregate_results(
                config=config,
                experiment_repetition_id=args.experiment_repetition_id,
                target_name=args.target_name
            )

        elif args.step == 'evaluate_results':
            src.method.nae.evaluate_results(
                config=config,
                experiment_repetition_id=args.experiment_repetition_id,
                target_name=args.target_name
            )

    if args.method == 'tandem':
        if args.step == 'train_backward_model':
            src.method.tandem.train_backward_model(
                config=config,
                experiment_repetition_id=args.experiment_repetition_id
            )

        elif args.step == 'invert_target':
            src.method.tandem.invert_target(
                config=config,
                experiment_repetition_id=args.experiment_repetition_id,
                target_name=args.target_name
            )

        elif args.step == 'evaluate_results':
            src.method.tandem.evaluate_results(
                config=config,
                experiment_repetition_id=args.experiment_repetition_id,
                target_name=args.target_name
            )

    if args.method == 'ua_tandem':
        if args.step == 'train_backward_model':
            src.method.ua_tandem.train_backward_model(
                config=config,
                experiment_repetition_id=args.experiment_repetition_id
            )

        elif args.step == 'invert_target':
            src.method.ua_tandem.invert_target(
                config=config,
                experiment_repetition_id=args.experiment_repetition_id,
                target_name=args.target_name
            )

        elif args.step == 'evaluate_results':
            src.method.ua_tandem.evaluate_results(
                config=config,
                experiment_repetition_id=args.experiment_repetition_id,
                target_name=args.target_name
            )
