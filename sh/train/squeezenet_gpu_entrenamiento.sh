#!/bin/bash
#SBATCH -p cola02
#SBATCH --gres=gpu:1 #Si se usase la cola02 (con GPU)
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --job-name=squeezenet_gpu # Nombre del job
#SBATCH --output=/home/estebanbecerraf/outputs/train/squeezenet_gpu_%j.log # Archivo de salida

# Ejecutar el c�digo Python dentro del contenedor Apptainer
apptainer exec --writable-tmpfs --nv /software/singularity/Informatica/mia-idl-apptainer/mia_idl_2.1.sif \
    accelerate launch --config_file /home/estebanbecerraf/config/config_gpubase.yaml \
    /home/estebanbecerraf/scripts/train/entrenamiento_squeezenet_gpu.py