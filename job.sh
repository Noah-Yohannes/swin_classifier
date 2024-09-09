#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=test_job
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --account= kuin0085
#SBATCH --output=WER.%j.out
#SBATCH --error=WER.%j.err 


module load miniconda/3 -q
module load cuda/11.7 -q
module load utilities/1.0

pip3 requirements.txt

