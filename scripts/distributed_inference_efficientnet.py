import torch
import torchvision.models as models
from accelerate import Accelerator, ProfileKwargs
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import yaml
import time

# Inicializar Accelerator primero para el entorno distribuido
accelerator = Accelerator(cpu=False)
device = accelerator.device

# Cargar configuración desde YAML 
with open("/home/estebanbecerraf/config/config_gpubase.yaml", "r") as f:
    config = yaml.safe_load(f)

# Parámetros del experimento 
batch_size = 128      
num_classes = 10     
dataset_path = "./data"  

# Habilitar cuDNN autotuner
torch.backends.cudnn.benchmark = True

# Transformaciones de imagen 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Generar dataset sintético
input_images = torch.rand((batch_size, 3, 224, 224))
labels = torch.randint(0, num_classes, (batch_size,))

dataset = TensorDataset(input_images, labels)
# Usar DistributedSampler para distribuir datos entre GPUs
sampler = DistributedSampler(dataset, num_replicas=accelerator.num_processes, 
                           rank=accelerator.process_index)
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=2)

# Configurar profiling para distributed inference
profiler_kwargs = ProfileKwargs(
    activities=["cuda"],
    record_shapes=True,
    profile_memory=True
)

# Cargar y preparar modelo para distributed inference
efficientnet = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
efficientnet = efficientnet.to(device)
efficientnet, dataloader = accelerator.prepare(efficientnet, dataloader)
efficientnet.eval()

# Obtener un batch del dataloader
data_iter = iter(dataloader)
input_tensor, _ = next(data_iter)

# Medir tiempo de inferencia con sincronización entre procesos
accelerator.wait_for_everyone()
start_time = time.time()
with accelerator.profile() as prof, torch.no_grad():
    output = efficientnet(input_tensor)
accelerator.wait_for_everyone()
end_time = time.time()

inference_time = end_time - start_time

# Guardar resultados solo en el proceso principal
output_file = f"/home/estebanbecerraf/outputs/distributed_inference/efficientnet_inference_gpu_results{accelerator.process_index}.txt"
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

# Limpieza
accelerator.end_training()