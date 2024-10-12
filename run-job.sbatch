#!/bin/bash

#SBATCH --nodes=3                        # requests 3 compute servers
#SBATCH --ntasks-per-node=2              # runs 2 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --time=1:00:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --account=pr_169_general
#SBATCH --job-name=python-self-cached-decoding
#SBATCH --output=spec.out

module purge
module load python/intel/3.8.6

module purge;
source venv/bin/activate;

hostname=$(hostname)
ip_address=$(hostname -I | awk '{print $1}')

echo "Model server running on node: $hostname with IP: $ip_address"

python ./main.py
