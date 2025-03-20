import torch
import torchvision.models as models
from accelerate import Accelerator, ProfileKwargs
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import yaml
import time

# Load configuration from YAML
with open("../../config/config_cpubase.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load the model
model = models.squeezenet1_1(weights='SqueezeNet1_1_Weights.IMAGENET1K_V1')  # Updated 'pretrained' to 'weights'
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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

# Initialize Accelerator explicitly for CPU
profiler_kwargs = ProfileKwargs(
    activities=["cpu"],
    record_shapes=True,
    profile_memory=True
)
accelerator = Accelerator(cpu=True, kwargs_handlers=[profiler_kwargs])

# Prepare model, optimizer, and dataloader for CPU execution
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

model.train()
device = accelerator.device  # Will be 'cpu'

# Measure training time
start_time = time.time()

# Training loop with profiling
with accelerator.profile() as prof:
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            accelerator.backward(loss)  # Use accelerator.backward instead of loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}")

train_time = time.time() - start_time

# Save results
output_file = "../../outputs/train/squeezenet_train_cpu_results.txt"
with open(output_file, "w") as f:
    f.write(f"Training Time: {train_time:.2f} seconds\n")
    f.write("\nProfiling Summary:\n")
    f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

print(f"Training completed. Results saved to {output_file}")