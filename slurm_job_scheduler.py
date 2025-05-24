import os
import time
from itertools import product

import yaml


def load_config(filename):
    with open(filename, 'r') as file:
        return yaml.safe_load(file)


def construct_args(params, combination, variable_keys):
    args = []
    long = False
    save_dir = ""
    exp_name_parts = [params["exp_name"][0]] if "exp_name" in params else []
    for key, value in zip(params.keys(), combination):
        if key in ["scheduler", "intensity_aug", "validate_extern", "distributed"]:
            if not value:
                args.append(f"--no-{key}")
            else:
                args.append(f"--{key}")
        elif key == "auto_resubmit":
            long = value
        elif key == "exp_name":
            pass
        else:
            args.append(f"--{key} {value}")
            if key == "save_path":
                save_dir = value
            if key in variable_keys:
                exp_name_parts.append(f"{key}{value}")
    exp_name = "-".join(exp_name_parts) or "experiment"
    args.append(f"--exp_name {exp_name}")
    save_dir = os.path.join(save_dir, exp_name)
    return long, save_dir, " ".join(args), exp_name


if __name__ == "__main__":
    config = load_config("config.yaml")
    params = config['params']
    variable_keys = [k for k, v in params.items() if len(v) > 1]

    for combination in product(*params.values()):
        long, save_dir, args, exp_name = construct_args(params, combination, variable_keys)
        os.makedirs(save_dir, exist_ok=False)
        if not long:
            command = f"sbatch --output {save_dir}/slurm.out aff_train.sh {args}"
        else:
            command = f"sbatch --output {save_dir}/slurm.out --export=ALL,SAVE_DIR={save_dir},LONG_JOB=TRUE,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True aff_train.sh {args}"
        print(f"Executing command: {command}")
        os.system(command)
        time.sleep(1)

    print("All jobs started")
