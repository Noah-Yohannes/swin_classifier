#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=swin
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --account=kunf0085
#SBATCH --output=smobile.%j.out
#SBATCH --error=smobile.%j.err

module load miniconda/3
module load cuda/11.7
module load utilities/1.0
pip3 install pillow
python swin_mobilenet.py

