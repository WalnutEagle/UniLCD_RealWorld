'''This is to test the gpu shared memory working and all'''

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
        self.fc1 = nn.Linear(20000, 4096)  # Input size remains the same
        self.fc2 = nn.Linear(4096, 4096)    # Hidden layer size
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNN().to(device)
model = nn.DataParallel(model).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Increase the size of x_train and y_train
# Previous batch size was 100,000; double it to 200,000
batch_size = 200000  # Increased batch size
x_train = torch.rand(batch_size, 20000).to(device)  # This will be about 15.2 GB
y_train = torch.randint(0, 10, (batch_size * 50,)).to(device)  # Increased to 50 times

# Training function
def train_model(epochs):
    model.train()
    for epoch in range(epochs):
        start_time = time.time()
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train[:batch_size])  # Make sure to match sizes
        loss.backward()
        optimizer.step()
        elapsed_time = time.time() - start_time
        print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}, Time taken: {elapsed_time:.2f} seconds')

# Run training for a specified number of epochs
train_model(100)