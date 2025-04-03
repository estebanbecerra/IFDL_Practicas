# Instrucciones para replicar los experimentos

Para poder ejecutar los experimentos en este proyecto, sigue los siguientes pasos:

1. **Pre-requisitos**:
   Asegúrate de tener acceso a un entorno con el sistema de colas SLURM, ya que los experimentos se ejecutan mediante el comando `sbatch`.

2. **Ejecutar los experimentos**:
   - Dirígete a la carpeta `sh` donde se encuentran los archivos de script `.sh` que definen los experimentos.
   - Para ejecutar un experimento, utiliza el siguiente comando para enviar el trabajo al sistema de colas SLURM:
     ```bash
     sbatch nombre_del_script.sh
     ```
   - Reemplaza `nombre_del_script.sh` por el archivo correspondiente que deseas ejecutar.

3. **Resultados**:
   - Todos los resultados de los experimentos se guardarán automáticamente en la carpeta `outputs`.
   - Revisa esta carpeta para acceder a los resultados generados después de la ejecución de los experimentos.


