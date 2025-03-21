import torch
import torchvision.models as models
from accelerate import Accelerator, ProfileKwargs
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
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

# Load the model with the new 'weights=' syntax
model = models.squeezenet1_1(weights='SqueezeNet1_1_Weights.IMAGENET1K_V1')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize Accelerator with GPU and profiling
profiler_kwargs = ProfileKwargs(
    activities=["cuda"],
    record_shapes=True,
    profile_memory=True
)
accelerator = Accelerator(cpu=False, kwargs_handlers=[profiler_kwargs])

# Prepare model, optimizer, and dataloader for GPU execution with Accelerate
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

model.train()
device = accelerator.device  # Debería ser 'cuda' si hay GPU

# Measure training time
start_time = time.time()

# Training loop with profiling
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

# Optional: free GPU cache
torch.cuda.empty_cache()

train_time = time.time() - start_time

# Save results
output_file = "/home/estebanbecerraf/outputs/train/squeezenet_train_gpu_results.txt"
with open(output_file, "w") as f:
    f.write(f"Training Time: {train_time:.2f} seconds\n")
    f.write("\nProfiling Summary:\n")
    f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print(f"Training completed. Results saved to {output_file}")
