import torch
import torchvision.models as models
from accelerate import Accelerator, ProfileKwargs
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import yaml
import time

# Load configuration from YAML
with open("/../../config/config_cpubase.yaml", "r") as f:
    config = yaml.safe_load(f)

# Inicializar `Accelerator` con configuración para CPU
profiler_kwargs = ProfileKwargs(
    activities=["cpu"],
    record_shapes=True,
    profile_memory=True
)
accelerator = Accelerator(cpu=True, kwargs_handlers=[profiler_kwargs])
device = accelerator.device

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define general experiment parameters
dataset_path = "./data"
batch_size = 128  # Reduced for RAM and CPU
num_epochs = 1    # Minimum for quick execution
num_classes = 10

# Generate synthetic dataset with random images and labels
input_images = torch.rand((batch_size, 3, 224, 224))  # Random image batch
labels = torch.randint(0, num_classes, (batch_size,))  # Random labels for 10 classes

dataset = TensorDataset(input_images, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)  # Added num_workers for CPU

# Cargar EfficientNet preentrenado
efficientnet = models.efficientnet_v2_l(pretrained=True)
efficientnet = accelerator.prepare(efficientnet.to(device))
efficientnet.eval()

# Medir inferencia
start_time = time.time()
with accelerator.profile() as prof, torch.no_grad():
    output = efficientnet(input_tensor)
end_time = time.time()

# Capturar tiempo total
inference_time = end_time - start_time

# Guardar resultados
with open("efficientnet_inference_cpu_results.txt", "w") as f:
    f.write(f"Tiempo de inferencia: {inference_time:.4f} segundos\n")
    f.write("\nResumen del perfilado:\n")
    f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
