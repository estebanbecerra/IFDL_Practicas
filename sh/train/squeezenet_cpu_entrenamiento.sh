#!/bin/bash
#SBATCH -p cola01             # Usamos la cola03 (multi-GPU)
#SBATCH -c 4                   # N�mero de cores de CPU
#SBATCH --mem=15G              # Memoria RAM asignada
#SBATCH --nodes=1              # N�mero de nodos
#SBATCH --ntasks=1             # N�mero de tareas
#SBATCH --time=04:00:00        # Tiempo m�ximo de ejecuci�n
#SBATCH --job-name=squeezenet_cpu # Nombre del job
#SBATCH --output=../../outputs/train/squeezenet_cpu_%j.log # Archivo de salida

# Ejecutar el c�digo Python dentro del contenedor Apptainer
apptainer exec --writable-tmpfs --nv /software/singularity/Informatica/mia-idl-apptainer/mia_idl_2.1.sif \
    accelerate launch --config_file ../../config/config_cpubase.yaml \
    ../../scripts/train/entrenamiento_squeezenet_cpu.py