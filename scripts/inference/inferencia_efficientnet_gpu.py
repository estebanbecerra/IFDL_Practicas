import torch
import torchvision.models as models
from accelerate import Accelerator, ProfileKwargs
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import yaml
import time

# Cargar configuración desde YAML 
with open("/home/estebanbecerraf/config/config_gpubase.yaml", "r") as f:
    config = yaml.safe_load(f)

# Parámetros del experimento 
batch_size = 128      
num_classes = 10     
dataset_path = "./data"  

# Habilitar cuDNN autotuner para un mejor rendimiento en GPU
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
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Inicializar Accelerator para GPU con perfilado
profiler_kwargs = ProfileKwargs(
    activities=["cuda"],
    record_shapes=True,
    profile_memory=True
)
accelerator = Accelerator(cpu=False, kwargs_handlers=[profiler_kwargs])
device = accelerator.device

# Cargar EfficientNet_v2_L 
efficientnet = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
efficientnet = efficientnet.to(device)
efficientnet = accelerator.prepare(efficientnet)
efficientnet.eval()

# Obtener un batch del dataloader para realizar la inferencia
data_iter = iter(dataloader)
input_tensor, _ = next(data_iter)
input_tensor = input_tensor.to(device)

# Medir tiempo de inferencia con perfilado
start_time = time.time()
with accelerator.profile() as prof, torch.no_grad():
    output = efficientnet(input_tensor)
end_time = time.time()

inference_time = end_time - start_time

# Guardar resultados en fichero
output_file = "/home/estebanbecerraf/outputs/train/efficientnet_inference_gpu_results.txt"
with open(output_file, "w") as f:
    f.write(f"Inference Time: {inference_time:.4f} seconds\n")
    f.write("\nProfiling Summary:\n")
    f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print(f"Inference completed. Results saved to {output_file}")
