import torch
import torch.nn as nn
import torch.optim as optim
import time

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple model definition
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10000, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Generate random data
x_train = torch.rand(10000, 10000).to(device)
y_train = torch.randint(0, 10, (10000,)).to(device)

# Training function
def train_model(epochs):
    model.train()
    for epoch in range(epochs):
        start_time = time.time()
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        elapsed_time = time.time() - start_time
        print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}, Time taken: {elapsed_time:.2f} seconds')

# Run training for a specified number of epochs
train_model(1000)
