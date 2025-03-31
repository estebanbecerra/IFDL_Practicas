import torch
import torchvision.models as models
from accelerate import Accelerator, ProfileKwargs
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision.datasets as datasets
import yaml
import time
import torch.multiprocessing as mp
from functools import partial

# Load configuration from YAML
with open("../../config/config_cpubase.yaml", "r") as f:
    config = yaml.safe_load(f)

batch_size = config.get("batch_size", 32)

# Inicializar `Accelerator` con configuración para CPU
profiler_kwargs = ProfileKwargs(
    activities=["cpu"],
    record_shapes=True,
    profile_memory=True
)
accelerator = Accelerator(cpu=True, kwargs_handlers=[profiler_kwargs])
device = accelerator.device

# Transformaciones de imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Crear datos sintéticos
input_images = torch.rand((batch_size * 4, 3, 224, 224))  # Más datos para distribuir (4 batches)
dummy_labels = torch.randint(0, 10, (batch_size * 4,))    # Etiquetas dummy
test_dataset = TensorDataset(input_images, dummy_labels)

# Función para realizar inferencia en un subconjunto de datos
def run_inference(model, dataloader, accelerator, rank):
    model.eval()
    inference_time = 0
    with torch.no_grad():
        for batch in dataloader:
            input_tensor, _ = batch
            input_tensor = input_tensor.to(accelerator.device)
            start_time = time.time()
            with accelerator.profile() as prof:
                output = model(input_tensor)
            end_time = time.time()
            inference_time += end_time - start_time
    return inference_time, prof

# Función principal para cada proceso
def inference_process(rank, world_size, dataset, model, accelerator):
    # Dividir el dataset entre procesos
    indices = list(range(len(dataset)))
    subset_indices = indices[rank::world_size]  # Dividir datos entre procesos
    subset = Subset(dataset, subset_indices)
    
    # Crear DataLoader para este proceso
    dataloader = DataLoader(
        subset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,  # Reducir workers por proceso para no saturar CPU
        pin_memory=True
    )
    
    # Preparar el modelo para este proceso
    model = accelerator.prepare(model)
    
    # Ejecutar inferencia
    inf_time, prof = run_inference(model, dataloader, accelerator, rank)
    
    # Guardar resultados por proceso
    with open(f"../../outputs/inference/efficientnet_cpu_process_{rank}.txt", "w") as f:
        f.write(f"Tiempo de inferencia (proceso {rank}): {inf_time:.4f} segundos\n")
        f.write("\nResumen del perfilado:\n")
        f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

if __name__ == '__main__':
    # Cargar EfficientNet preentrenado
    efficientnet = models.efficientnet_v2_l(pretrained=True)
    
    # Configurar multiprocessing
    world_size = 4  # Número de procesos (ajusta según núcleos de CPU, e.g., mp.cpu_count())
    mp.set_start_method('spawn')  # Necesario para evitar problemas en algunas plataformas
    
    # Iniciar procesos
    start_time = time.time()
    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target=inference_process, 
            args=(rank, world_size, test_dataset, efficientnet, accelerator)
        )
        processes.append(p)
        p.start()
    
    # Esperar a que todos los procesos terminen
    for p in processes:
        p.join()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Guardar tiempo total
    with open("../../outputs/inference/efficientnet_inference_cpu_total.txt", "w") as f:
        f.write(f"Tiempo total de inferencia distribuida: {total_time:.4f} segundos\n")
