#!/bin/bash
#SBATCH --job-name=corpus_2025
#SBATCH --output=logs/corpus_2025_%j.out
#SBATCH --error=logs/corpus_2025_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Move to project directory
cd /home/elhanan/PROJECTS/CHERRY_PICKER_AR

# Run your script with the correct Python environment
/home/elhanan/PROJECTS/CHERRY_PICKER_AR/conda_envs/alons_env/bin/python \
  /home/elhanan/PROJECTS/CHERRY_PICKER_AR/code/fixed_build_corpus.py
