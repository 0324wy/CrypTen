#!/usr/bin/env python3

# Import necessary libraries
import logging
import crypten
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from examples.meters import AccuracyMeter
from examples.util import NoopContextManager

try:
    from crypten.nn.tensorboard import SummaryWriter
except ImportError:  # tensorboard not installed
    SummaryWriter = None

# Define the CNN model for MNIST
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # MNIST images are 1x28x28
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # Output: 32x26x26
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # Output: 64x24x24
        self.fc1 = nn.Linear(64 * 24 * 24, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes for digits 0-9

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64 * 24 * 24)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to train the model
def train_model(model, data_folder='./data', num_epochs=1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Mean and std for MNIST
    ])
    train_dataset = datasets.MNIST(data_folder, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
    print('Training complete.')

# Function to run the experiment
def run_experiment(
    model,
    data_folder='./data',
    tensorboard_folder=None,
    num_samples=None,
    context_manager=None,
):
    """Runs inference using the specified vision model on the MNIST dataset."""

    crypten.init()

    if context_manager is None:
        context_manager = NoopContextManager()

    # Load the MNIST test dataset
    with context_manager:
        model.eval()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # Mean and std for MNIST
        ])
        dataset = datasets.MNIST(data_folder, train=False, download=True, transform=transform)

    # Encrypt the model
    dummy_input = torch.rand(1, 1, 28, 28)  # MNIST images are 1x28x28
    encrypted_model = crypten.nn.from_pytorch(model, dummy_input=dummy_input)
    encrypted_model.encrypt()

    # Optionally, show encrypted model in tensorboard
    if SummaryWriter is not None and tensorboard_folder is not None:
        writer = SummaryWriter(log_dir=tensorboard_folder)
        writer.add_graph(encrypted_model)
        writer.close()

    # Initialize the accuracy meter
    meter = AccuracyMeter()

    # Loop over the test dataset
    for idx, (image, target) in enumerate(dataset):
        # Preprocess sample
        image = image.unsqueeze(0)  # Add batch dimension
        target = torch.tensor([target], dtype=torch.long)

        # Encrypt the image
        encrypted_image = crypten.cryptensor(image)
        # Perform inference using the encrypted model
        encrypted_output = encrypted_model(encrypted_image)

        # Decrypt the output and measure accuracy
        output = encrypted_output.get_plain_text()
        meter.add(output, target)

        # Display progress
        if (idx + 1) % 1000 == 0:
            logging.info(
                "[Sample %d of %d] Current Accuracy: %f" % (idx + 1, len(dataset), meter.value()[1])
            )
        if num_samples is not None and idx == num_samples - 1:
            break

    # Print final accuracy
    logging.info("Final Accuracy on %d samples: %f" % (idx + 1, meter.value()[1]))

# Main execution
if __name__ == '__main__':
    # Create an instance of the model
    model = SimpleCNN()

    # Train the model
    train_model(model, num_epochs=1)  # Increase num_epochs for better accuracy

    # Run the experiment
    run_experiment(
        model=model,
        data_folder='./data',
        tensorboard_folder=None,  # Set a path if you want to use TensorBoard
        num_samples=None,  # Set to a number to limit samples
        context_manager=None,
    )