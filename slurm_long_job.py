import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit a job with custom save_dir and pass other arguments.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the model and logs")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    args, unknown_args = parser.parse_known_args()

    save_path = args.save_path
    exp_name = args.exp_name
    try:
        save_dir = os.path.join(save_path, exp_name)
        os.makedirs(f"{save_path}/{exp_name}", exist_ok=False)
    except FileExistsError as error:
        print(f"Error: Experiment already exists: {save_path}/{exp_name}")
        exit(1)

    command = f"sbatch --export=ALL,SAVE_DIR={save_dir},LONG_JOB=TRUE --job-name {exp_name} --output {save_dir}/slurm-log.txt aff_train.sh {' '.join(unknown_args)} --save_path {save_path} --exp_name {exp_name}"

    # Execute the command
    os.system(command)
