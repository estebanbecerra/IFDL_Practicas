#!/bin/bash
#SBATCH -p cola02
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --job-name=vgg16_gpu_ds
#SBATCH --output=../../outputs/train/vgg16_gpu_ds_%j.log

apptainer exec --writable-tmpfs --nv /software/singularity/Informatica/mia-idl-apptainer/mia_idl_2.2.sif \
bash -c "
cd /home/estebanbecerraf/IFDL_Practicas-main &&
/opt/conda/bin/pip install deepspeed &&
/opt/conda/bin/deepspeed scripts/train/entrenamiento_vgg16_deepspeed.py \
--deepspeed --deepspeed_config config/deepspeed_config.json
"
