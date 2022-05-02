#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=cpp
#SBATCH --output=%j_threads_8.out

module purge
module load gcc/10.2.0

cd /home/aeb783/HPC-Final-Project/

make
./main_gd
