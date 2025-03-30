#!/bin/bash
#SBATCH -p cola02
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --job-name=squeezenet_gpu_ds
#SBATCH --output=../../outputs/train/squeezenet_deepspeed_%j.log

apptainer exec --writable-tmpfs --nv /software/singularity/Informatica/mia-idl-apptainer/mia_idl_2.2.sif \
    deepspeed ../../scripts/train/entrenamiento_squeezenet_deepspeed.py \
    --deepspeed --deepspeed_config ../../config/deepspeed_config.json
