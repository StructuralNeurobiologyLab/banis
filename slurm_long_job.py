import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit a job with custom save_dir and pass other arguments.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the model and logs")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--resubmit", action=argparse.BooleanOptionalAction, default=False, help="Continue already existing training")
    args, unknown_args = parser.parse_known_args()

    save_path = args.save_path
    exp_name = args.exp_name
    save_dir = os.path.join(save_path, exp_name)
    if not args.resubmit:
        try:
            os.makedirs(f"{save_path}/{exp_name}", exist_ok=False)
        except FileExistsError as error:
            print(f"Error: Experiment already exists: {save_path}/{exp_name}")
            exit(1)

        command = f"sbatch --export=ALL,SAVE_DIR={save_dir},LONG_JOB=TRUE,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --job-name {exp_name} --output {save_dir}/slurm-log.txt aff_train.sh {' '.join(unknown_args)}  --long_training --save_path {save_path} --exp_name {exp_name}"

    else:
        command = f"sbatch --export=ALL,RESUME=TRUE,SAVE_DIR={save_dir},LONG_JOB=TRUE,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,LONG_JOB_ARGS='{' '.join(unknown_args)}  --long_training --save_path {save_path} --exp_name {exp_name}' --job-name {exp_name} --output {save_dir}/slurm-log.txt aff_train.sh {' '.join(unknown_args)}  --long_training --save_path {save_path} --exp_name {exp_name}"

    # Execute the command
    print(command)
    os.system(command)
