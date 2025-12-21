#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --partition=gpu_h100
#SBATCH --time=01-00:00:00

#SBATCH -o /gpfs/work4/0/gus20642/dwu18/log/out.pipeline.%j.o
#SBATCH -e /gpfs/work4/0/gus20642/dwu18/log/out.pipeline.%j.e

source activate py38cuda11
# source activate py39

export HF_HUB_CACHE=/gpfs/work4/0/gus20642/dwu18/cache
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python test_pipeline.py
