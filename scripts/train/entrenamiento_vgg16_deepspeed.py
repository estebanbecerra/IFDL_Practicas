import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import deepspeed
import time
import json
import os
from torch.profiler import profile, record_function, ProfilerActivity

# Parametros basicos
batch_size = 128
num_epochs = 1
num_classes = 10

# Dataset sintetico
input_images = torch.rand((batch_size, 3, 224, 224))
labels = torch.randint(0, num_classes, (batch_size,))
dataset = TensorDataset(input_images, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Modelo VGG16
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
criterion = nn.CrossEntropyLoss()

# Cargar configuracion DeepSpeed
with open('config/deepspeed_config.json', 'r') as f:
    ds_config = json.load(f)

# Inicializar DeepSpeed
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config
)

model.train()
device = model.device

# Carpeta de resultados
output_dir = "outputs/train"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "vgg16_train_gpu_deepspeed_results.txt")

# Profiler estilo Accelerate
with profile(
    activities=[ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:

    start_time = time.time()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device).half(), targets.to(device)

            with record_function("forward"):
                outputs = model(inputs)

            with record_function("loss"):
                loss = criterion(outputs, targets)

            with record_function("backward"):
                model.backward(loss)

            with record_function("step"):
                model.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader):.4f}")

    torch.cuda.empty_cache()
    train_time = time.time() - start_time

# Guardar resultados como con Accelerate
with open(output_file, "w") as f:
    f.write(f"Training Time: {train_time:.2f} seconds\n")
    f.write("\nProfiling Summary:\n")
    f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print(f"Entrenamiento completado con DeepSpeed. Resultados guardados en {output_file}")
