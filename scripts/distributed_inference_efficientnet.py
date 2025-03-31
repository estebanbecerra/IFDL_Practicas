import torch
import torchvision.models as models
from torchvision.models import EfficientNet_V2_L_Weights
from accelerate import Accelerator, ProfileKwargs
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import yaml
import time
import os

# Configurar profiling para distributed inference
profiler_kwargs = ProfileKwargs(
    activities=["cuda"],
    record_shapes=True,
    profile_memory=True
)

# Inicializar Accelerator primero para el entorno distribuido
accelerator = Accelerator(cpu=False, kwargs_handlers=[profiler_kwargs])
device = accelerator.device

# Cargar configuración desde YAML 
with open("../../config/config_gpubase.yaml", "r") as f:
    config = yaml.safe_load(f)

# Parámetros del experimento 
global_batch_size = 128       # Batch total deseado
num_classes = 10     
dataset_path = "./data"  

# Habilitar cuDNN autotuner para eficiencia (si los tamaños son fijos)
torch.backends.cudnn.benchmark = True

# Transformaciones de imagen 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Generar dataset sintético
input_images = torch.rand((global_batch_size, 3, 224, 224))
labels = torch.randint(0, num_classes, (global_batch_size,))
dataset = TensorDataset(input_images, labels)

# Batch size por proceso
per_process_batch_size = global_batch_size // accelerator.num_processes

# Usar DistributedSampler para distribuir datos entre GPUs
sampler = DistributedSampler(dataset, num_replicas=accelerator.num_processes, 
                             rank=accelerator.process_index, shuffle=False)
dataloader = DataLoader(dataset, batch_size=per_process_batch_size, sampler=sampler, num_workers=2)

# Cargar modelo EfficientNet V2 preentrenado
efficientnet = models.efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
efficientnet.eval()
efficientnet = efficientnet.to(device)

# Preparar modelo y dataloader para entorno distribuido
efficientnet, dataloader = accelerator.prepare(efficientnet, dataloader)

# Obtener un batch del dataloader
data_iter = iter(dataloader)
input_tensor, _ = next(data_iter)
input_tensor = input_tensor.to(device)

# Sincronización previa
accelerator.wait_for_everyone()
start_time = time.time()

# Inferencia distribuida con perfilado
with accelerator.profile() as prof, torch.no_grad():
    output = efficientnet(input_tensor)

accelerator.wait_for_everyone()
end_time = time.time()
inference_time = end_time - start_time

# Crear carpeta de salida si no existe
output_dir = "../../outputs/distributed_inference"
os.makedirs(output_dir, exist_ok=True)

# Guardar resultados solo desde el proceso principal
output_file = os.path.join(output_dir, f"efficientnet_inference_gpu_results{accelerator.process_index}.txt")
if accelerator.is_main_process:
    with open(output_file, "w") as f:
        f.write(f"Inference Time: {inference_time:.4f} seconds\n")
        f.write(f"Process Rank: {accelerator.process_index}\n")
        f.write(f"Number of Processes: {accelerator.num_processes}\n")
        f.write("\nProfiling Summary:\n")
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(f"Inference completed. Results saved to {output_file}")
else:
    print(f"Inference completed on rank {accelerator.process_index}")

accelerator.end_training()
