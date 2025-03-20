import torch
import torchvision.models as models
from accelerate import Accelerator, ProfileKwargs
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import yaml
import time

# Cargar configuración desde YAML
with open("../../config/config_cpubase.yaml", "r") as f:
    config = yaml.safe_load(f)


# Cargamos el modelo

model = models.squeezenet1_1(pretrained=True)
model = torch.compile(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Transformaciones de imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Definimos parámtros generales para el experimento
dataset_path = "./data"
batch_size = 128 # Reducido para tu RAM y CPU
num_epochs = 1   # Mínimo para que sea rápido
num_classes = 10


# Generamos el dataset con imágenes y etiquetas aleatorias (del 1 al 10)
input_images = torch.rand((batch_size, 3, 224, 224))  # Random image batch
labels = torch.randint(0, 10, (batch_size,))  # Random labels for 10 classes


dataset = TensorDataset(input_images, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Inicializar `Accelerator` con configuración para CPU
profiler_kwargs = ProfileKwargs(
    activities=["cpu"],
    record_shapes=True,
    profile_memory=True
)
accelerator = Accelerator(cpu=True, kwargs_handlers=[profiler_kwargs])


# Prepare the model, optimizer, and data loader for CPU execution
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

model.train()

device = accelerator.device


# Medir tiempo
start_time = time.time()


with accelerator.profile() as prof:
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}")

train_time = time.time() - start_time

# Guardar resultados
with open("../../outputs/train/squeezenet_train_cpu_results.txt", "w") as f:
    f.write(f"Tiempo de entrenamiento: {train_time:.2f} segundos\n")
    f.write("\nResumen del perfilado:\n")
    f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))