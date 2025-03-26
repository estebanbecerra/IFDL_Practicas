#!/bin/bash
#SBATCH -p cola01             # Usamos la cola01 (multi-GPU)
#SBATCH -c 4                   # N�mero de cores de CPU
#SBATCH --mem=15G              # Memoria RAM asignada
#SBATCH --nodes=1              # N�mero de nodos
#SBATCH --ntasks=1             # N�mero de tareas
#SBATCH --time=00:10:00        # Tiempo m�ximo de ejecuci�n
#SBATCH --job-name=mobilnet_cpu # Nombre del job
#SBATCH --output=../../outputs/inference/mobilenet_cpu_%j.log # Archivo de salida

# Ejecutar el c�digo Python dentro del contenedor Apptainer
apptainer exec --writable-tmpfs --nv /software/singularity/Informatica/mia-idl-apptainer/mia_idl_2.1.sif \
    accelerate launch --config_file ../../config/config_cpubase.yaml \
    ../../scripts/inference/inferencia_mobilenet_cpu.py