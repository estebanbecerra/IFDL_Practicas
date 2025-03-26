import torch
import torchvision.models as models
from accelerate import Accelerator, ProfileKwargs
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import yaml
import time

# Cargar configuración desde YAML
with open("../../config/config_gpubase.yaml", "r") as f:
    config = yaml.safe_load(f)

batch_size = config.get("batch_size", 32)

# Inicializar `Accelerator` con configuración para GPU
profiler_kwargs = ProfileKwargs(
    activities=["cuda"],  # Cambiado de "cpu" a "cuda"
    record_shapes=True,
    profile_memory=True
)
accelerator = Accelerator(cpu=False, kwargs_handlers=[profiler_kwargs])  # cpu=False para usar GPU
device = accelerator.device  # Será 'cuda' si hay GPU disponible

# Transformaciones de imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Crear datos sintéticos en lugar de CIFAR-10
input_images = torch.rand((batch_size, 3, 224, 224))  # Batch de imágenes aleatorias
dummy_labels = torch.randint(0, 10, (batch_size,))    # Etiquetas dummy
test_dataset = TensorDataset(input_images, dummy_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Cargar MobileNet preentrenado
mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet = accelerator.prepare(mobilenet.to(device))  # Preparado para GPU
mobilenet.eval()

# Obtener un batch
data_iter = iter(test_loader)
input_tensor, _ = next(data_iter)
input_tensor = input_tensor.to(device)  # Mover a GPU

# Sincronizar GPU antes de medir tiempo
if torch.cuda.is_available():
    torch.cuda.synchronize()

# Medir inferencia
start_time = time.time()
with accelerator.profile() as prof, torch.no_grad():
    output = mobilenet(input_tensor)
    
# Sincronizar GPU después de la inferencia
if torch.cuda.is_available():
    torch.cuda.synchronize()
end_time = time.time()

# Capturar tiempo total
inference_time = end_time - start_time

# Guardar resultados
with open("../../outputs/inference/mobilenet_inference_gpu_results.txt", "w") as f:
    f.write(f"Tiempo de inferencia: {inference_time:.4f} segundos\n")
    f.write("\nResumen del perfilado:\n")
    f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))  # Cambiado a "cuda_time_total"

print(f"Inferencia completada en GPU. Tiempo: {inference_time:.4f} segundos")
