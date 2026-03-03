#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
export PYTHONUNBUFFERED=1
export STDOUT_LINE_BUFFERED=1
export HF_HOME=/local/scratch/jdfinch/.cache/
export TRANSFORMERS_CACHE=/local/scratch/jdfinch/.cache/huggingface/transformers/
export XDG_CACHE_HOME=/local/scratch/jdfinch/.cache/
/local/scratch/jdfinch/miniconda3/envs/dextrous_l3/bin/python dextrous/experiment.py $1