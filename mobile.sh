#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=swin
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=prod
#SBATCH --account=kunf0085
#SBATCH --output=mobilenet.%j.out
#SBATCH --error=mobilenet.%j.err

module load miniconda/3
module load cuda/11.7
module load utilities/1.0
pip3 install pillow
python mobilenet_only.py

