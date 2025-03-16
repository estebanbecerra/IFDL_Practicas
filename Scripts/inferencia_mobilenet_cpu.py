import torch
import torchvision.models as models
from accelerate import Accelerator, ProfileKwargs
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import yaml
import time

# Cargar configuración desde YAML
with open("config_cpubase.yaml", "r") as f:
    config = yaml.safe_load(f)

dataset_path = config.get("dataset_path", "./data")
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

# Cargar CIFAR-10
test_dataset = datasets.CIFAR10(root=dataset_path, train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Cargar MobileNet preentrenado
mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet = accelerator.prepare(mobilenet.to(device))
mobilenet.eval()

# Obtener un batch
data_iter = iter(test_loader)
input_tensor, _ = next(data_iter)
input_tensor = input_tensor.to(device)

# Medir inferencia
start_time = time.time()
with accelerator.profile() as prof, torch.no_grad():
    output = mobilenet(input_tensor)
end_time = time.time()

# Capturar tiempo total
inference_time = end_time - start_time

# Guardar resultados
with open("mobilenet_inference_cpu_results.txt", "w") as f:
    f.write(f"Tiempo de inferencia: {inference_time:.4f} segundos\n")
    f.write("\nResumen del perfilado:\n")
    f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))