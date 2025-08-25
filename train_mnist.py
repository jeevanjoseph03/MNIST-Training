# =============================================================================
# Section 1: Imports
# =============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =============================================================================
# Section 2: Model Definition (Our blueprint from before)
# =============================================================================
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(28*28, 128) # 28*28 = 784
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

# =============================================================================
# Section 3: Prepare the Data
# =============================================================================
# Define a transform to convert images to PyTorch Tensors
transform = transforms.ToTensor()

# Download and load the training data
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Download and load the testing data
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# =============================================================================
# Section 4: Instantiate Model, Loss, and Optimizer
# =============================================================================
model = SimpleNet()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =============================================================================
# Section 5: The Training Loop
# =============================================================================
num_epochs = 5 # How many times to loop over the entire dataset

print("Starting training...")
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Step 1: Forward Pass
        predictions = model(images)
        
        # Step 2: Calculate Loss
        loss = loss_function(predictions, labels)
        
        # Step 3 & 4: Backpropagation and Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training finished!")

# =============================================================================
# Section 6: Evaluate the Model
# =============================================================================
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0

with torch.no_grad():  # We don't need to calculate gradients for testing
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy of the model on the test images: {accuracy:.2f}%")
