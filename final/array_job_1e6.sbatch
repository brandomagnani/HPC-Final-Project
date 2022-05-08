#!/bin/bash

## sbatch array_job.sbatch --array=1-8

#SBATCH --job-name=cpp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=3:00:00
#SBATCH --mem=16GB
#SBATCH --output=output2/1e6/%a_threads_sgd.csv

module purge
module load gcc/10.2.0

cd /home/aeb783/HPC-Final-Project/

./main $SLURM_ARRAY_TASK_ID 1000000
