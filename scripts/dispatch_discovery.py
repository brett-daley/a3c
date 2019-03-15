import os
from subprocess import call


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
python {runner_path} --env {env} --history-len {history_len} --lambd {lambd} --seed {seed} {renorm} &> {results_path}
echo end at $(date)
'''

lambdas = [0.7, 1.0]
history_lens = [1]
seeds = [0, 1, 2]

environments = [
    'breakout',
    'beam_rider',
    'pong',
    'qbert',
    'seaquest',
    'space_invaders',
]


def mkdir(name):
    path = os.path.join(os.getcwd(), name)
    if not os.path.exists(path):
        os.mkdir(path)
    return path


if __name__ == '__main__':
    # Make sure we are in the repository's root directory
    work_dir = os.getcwd()
    runner_path = os.path.join(work_dir, 'run_a3c_atari.py')
    assert os.path.exists(runner_path)

    # Make directories for slurm, log, and results files
    slurm_dir   = mkdir('slurm')
    log_dir     = mkdir('logs')
    results_dir = mkdir('results')

    # Begin dispatching experiments
    for env in environments:
        for lambd in lambdas:
            for len in history_lens:
                for renorm in ([False, True] if lambd != 1.0 else [False]):
                    for seed in seeds:
                        # Generate job name and paths
                        job_name = '_'.join([
                            'a3c',
                            env,
                            'len' + str(len),
                            ('renorm' if renorm else '') + 'lambda' + str(lambd),
                            'seed' + str(seed),
                        ])
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
                                lambd=lambd,
                                seed=seed,
                                renorm=('--renorm' if renorm else ''),
                                results_path=results_path,
                            )
                            file.write(slurm)

                        # Call sbatch to queue the job
                        print('Dispatching', job_name)
                        call(['sbatch', slurm_path])
