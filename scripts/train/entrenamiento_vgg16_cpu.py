import torch
import torchvision.models as models
from accelerate import Accelerator, ProfileKwargs
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import yaml
import time

# Cargar YAML 
with open("../../config/config_cpubase.yaml", "r") as f:
    config = yaml.safe_load(f)

# Definir parámetros de entrenamiento 
batch_size = 128
num_epochs = 1
num_classes = 10

# Cargar el modelo 
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Transformaciones de imagen 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Crear datos sintéticos (un batch de imágenes aleatorias y sus etiquetas)
input_images = torch.rand((batch_size, 3, 224, 224))
labels = torch.randint(0, num_classes, (batch_size,))

dataset = TensorDataset(input_images, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Inicializar Accelerator explícitamente para CPU
profiler_kwargs = ProfileKwargs(
    activities=["cpu"],
    record_shapes=True,
    profile_memory=True
)
accelerator = Accelerator(cpu=True, kwargs_handlers=[profiler_kwargs])

# Preparar modelo, optimizador y dataloader para ejecución en CPU
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

model.train()
device = accelerator.device  # Será 'cpu'

# Medir el tiempo de entrenamiento
start_time = time.time()

# Bucle de entrenamiento con perfilado
with accelerator.profile() as prof:
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targets.to(device))
            accelerator.backward(loss)
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader):.4f}")

train_time = time.time() - start_time

# Guardar resultados en fichero
output_file = "../../outputs/train/vgg16_train_cpu_results.txt"
with open(output_file, "w") as f:
    f.write(f"Training Time: {train_time:.2f} seconds\n")
    f.write("\nProfiling Summary:\n")
    f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

print(f"Training completed. Results saved to {output_file}")
