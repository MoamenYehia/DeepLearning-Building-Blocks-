import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 1. Data Engine Setup
X_raw = torch.randn(100, 4)
y_raw = torch.randint(0, 3, (100,))

dataset = TensorDataset(X_raw, y_raw)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 2. Architecture Definition
model = nn.Sequential(
    nn.Linear(in_features=4, out_features=8),
    nn.ReLU(),
    nn.Linear(in_features=8, out_features=3)
)

# 3. Training Configurations
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
num_epochs = 20  

# 4. The Master Training Loop
print("Starting training pipeline...")
print("-" * 30)

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_X, batch_y in dataloader:
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(batch_X)
        
        # Loss calculation
        loss = criterion(predictions, batch_y)
        
        # Backward pass
        loss.backward()
        
        # Weight updates
        optimizer.step()
        
        epoch_loss += loss.item()
        
    average_loss = epoch_loss / len(dataloader)
    print(f"Epoch: {epoch:02d} | Average Loss: {average_loss:.4f}")

print("-" * 30)
print("Training complete!")