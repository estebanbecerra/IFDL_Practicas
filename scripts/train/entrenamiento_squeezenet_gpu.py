import torch
import torchvision.models as models
from accelerate import Accelerator, ProfileKwargs
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import yaml
import time

# Cargar configuración desde YAML
with open("config_gpubase.yaml", "r") as f:
    config = yaml.safe_load(f)

dataset_path = config.get("dataset_path", "./data")
batch_size = config.get("batch_size", 32)

# Inicializar `Accelerator` con configuración para GPU
profiler_kwargs = ProfileKwargs(
    activities=["cuda"],
    record_shapes=True,
    profile_memory=True
)
accelerator = Accelerator(cpu=False, kwargs_handlers=[profiler_kwargs])
device = accelerator.device

# Transformaciones de imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Cargar CIFAR-10
train_dataset = datasets.CIFAR10(root=dataset_path, train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

# Cargar modelo
torch.backends.cudnn.benchmark = True  # Acelera GPU
squeezenet = models.squeezenet1_1(pretrained=True).to(device)

# Definir pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(squeezenet.parameters(), lr=0.001)

# Preparar con Accelerate
train_loader, optimizer, squeezenet = accelerator.prepare(train_loader, optimizer, squeezenet)

# Entrenamiento
num_epochs = 1
start_time = time.time()

with accelerator.profile() as prof:
    for epoch in range(num_epochs):
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = squeezenet(X)
            loss = criterion(outputs, y)
            accelerator.backward(loss)
            optimizer.step()

# Medir tiempo
torch.cuda.empty_cache()
train_time = time.time() - start_time

# Guardar resultados
with open("squeezenet_train_gpu_results.txt", "w") as f:
    f.write(f"Tiempo de entrenamiento: {train_time:.2f} segundos\n")
    f.write("\nResumen del perfilado:\n")
    f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))