import torch
import torchvision.models as models
from accelerate import Accelerator, ProfileKwargs
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import yaml
import time

# Load config from YAML 
with open("home/estebanbecerraf/config/config_gpubase.yaml", "r") as f:
    config = yaml.safe_load(f)

# Ajusta parámetros de entrenamiento directamente
batch_size = 128
num_epochs = 1
num_classes = 10

# Habilitar cuDNN benchmark para mejor rendimiento en GPU
torch.backends.cudnn.benchmark = True

# Definir transformaciones (incluye normalización)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Generar datos sintéticos (imágenes y etiquetas aleatorias)
input_images = torch.rand((batch_size, 3, 224, 224))
labels = torch.randint(0, num_classes, (batch_size,))

dataset = TensorDataset(input_images, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Cargar el modelo 
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Inicializar Accelerator con GPU y configurar perfilado
profiler_kwargs = ProfileKwargs(
    activities=["cuda"],
    record_shapes=True,
    profile_memory=True
)
accelerator = Accelerator(cpu=False, kwargs_handlers=[profiler_kwargs])

# Preparar el modelo, optimizador y dataloader para la GPU
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

model.train()
device = accelerator.device  # Será 'cuda' si hay GPU disponible

start_time = time.time()

# Entrenamiento con perfilado
with accelerator.profile() as prof:
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            accelerator.backward(loss)
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader):.4f}")

# Limpiar caché de la GPU 
torch.cuda.empty_cache()
train_time = time.time() - start_time

# Guardar resultados en un fichero
output_file = "home/estebanbecerraf/outputs/train/vgg16_train_gpu_results.txt"
with open(output_file, "w") as f:
    f.write(f"Training Time: {train_time:.2f} seconds\n")
    f.write("\nProfiling Summary:\n")
    f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print(f"Training completed. Results saved to {output_file}")
