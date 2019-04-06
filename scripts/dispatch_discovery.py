import os
from subprocess import call
from argparse import ArgumentParser
import yaml


template = '''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_path}
#SBATCH --error={log_path}
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8192

echo start at $(date)
python {runner_path} --env {env} --history-len {history_len} --return-type {return_type} --seed {seed} &> {results_path}
echo end at $(date)
'''


def mkdir(name):
    path = os.path.join(os.getcwd(), name)
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def make_name(env, len, return_type, seed):
    return '_'.join(['a3c', env.replace('_', '-'), 'len' + str(len), return_type, 'seed' + str(seed)])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('target', type=str, help='Name of experiment to run. Use \'all\' to run everything.')
    args = parser.parse_args()

    # Make sure we are in the repository's root directory
    work_dir = os.getcwd()
    runner_path = os.path.join(work_dir, 'run_a3c_atari.py')
    assert os.path.exists(runner_path)

    config_path = os.path.join(work_dir, 'scripts/_automate.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Make directories for slurm, log, and results files
    slurm_dir   = mkdir('slurm')
    log_dir     = mkdir('logs')
    results_dir = mkdir('results')

    # Begin dispatching experiments
    experiments = config.values() if args.target == 'all' else [config[args.target]]

    for exp in experiments:
        for env in exp['env']:
            for len in exp['history_len']:
                for return_type in exp['return_type']:
                    for seed in range(exp['num_seeds']):
                        # Generate job name and paths
                        job_name = make_name(env, len, return_type, seed)
                        slurm_path   = os.path.join(slurm_dir,   job_name + '.slurm')
                        log_path     = os.path.join(log_dir,     job_name + '.txt')
                        results_path = os.path.join(results_dir, job_name + '.txt')

                        # If results already exist, do not overwrite
                        if os.path.exists(results_path):
                            print('Warning: skipped', job_name, 'because results already exist')
                            continue

                        # Fill in template and save to slurm directory
                        with open(slurm_path, 'w') as file:
                            slurm = template.format(
                                job_name=job_name,
                                log_path=log_path,
                                runner_path=runner_path,
                                env=env,
                                history_len=len,
                                return_type=return_type,
                                seed=seed,
                                results_path=results_path,
                            )
                            file.write(slurm)

                        # Call sbatch to queue the job
                        print('Dispatching', job_name)
                        call(['sbatch', slurm_path])
