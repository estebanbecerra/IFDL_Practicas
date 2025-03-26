import torch
import torchvision.models as models
from accelerate import Accelerator, ProfileKwargs
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import yaml
import time

with open("/home/estebanbecerraf/config/config_gpubase.yaml", "r") as f:
    config = yaml.safe_load(f)

# Define experiment parameters directly in the script
batch_size = 128      
num_epochs = 1        
num_classes = 10     
dataset_path = "./data"  

# Enable cuDNN autotuner for performance
torch.backends.cudnn.benchmark = True

# Image transformations (incluye normalización típica de ImageNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Generate synthetic dataset (random images and labels)
input_images = torch.rand((batch_size, 3, 224, 224))
labels = torch.randint(0, num_classes, (batch_size,))

dataset = TensorDataset(input_images, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Inicializar `Accelerator` con configuración para GPU
profiler_kwargs = ProfileKwargs(
    activities=["cuda"],
    record_shapes=True,
    profile_memory=True
)
accelerator = Accelerator(cpu=False, kwargs_handlers=[profiler_kwargs])
device = accelerator.device

# Cargar EfficientNet preentrenado
efficientnet = models.efficientnet_v2_l(pretrained=True)
efficientnet = accelerator.prepare(efficientnet.to(device))
efficientnet.eval()

# Obtener un batch
data_iter = iter(test_loader)
input_tensor, _ = next(data_iter)
input_tensor = input_tensor.to(device)

# Medir inferencia
start_time = time.time()
with accelerator.profile() as prof, torch.no_grad():
    output = efficientnet(input_tensor)
end_time = time.time()

# Capturar tiempo total
inference_time = end_time - start_time

# Guardar resultados
with open("efficientnet_inference_gpu_results.txt", "w") as f:
    f.write(f"Tiempo de inferencia: {inference_time:.4f} segundos\n")
    f.write("\nResumen del perfilado:\n")
    f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
