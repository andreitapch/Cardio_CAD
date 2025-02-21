import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AutoencoderDm(nn.Module):
    def __init__(self, input_dim, encoding_dim, dropout_rate):
        super(AutoencoderDm, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),

            nn.Linear(64, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def train_autoencoder_with_tensorboard(model, data, epochs, log_dir, batch_size=64, learning_rate=1e-4):
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    writer = SummaryWriter(log_dir)

    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size].to(device)
            optimizer.zero_grad()
            _, decoded = model(batch)
            loss = criterion(decoded, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / (len(data) // batch_size)
        writer.add_scalar('Loss/Train', avg_loss, epoch)

        if (epoch + 1) % 50 == 0:
            log_reconstruction(writer, model, data, epoch)

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

    writer.close()


def log_reconstruction(writer, model, data, epoch):
    model.eval()
    with torch.no_grad():
        _, decoded = model(data.to(device))

    fig, axes = plt.subplots(3, 1, figsize=(10, 6))
    for i in range(3):
        axes[i].plot(data[i].cpu().numpy(), label="Original")
        axes[i].plot(decoded[i].cpu().numpy(), label="Reconstructed")
        axes[i].legend()

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = torch.tensor(np.array(plt.imread(buf))).permute(2, 0, 1)
    writer.add_image(f'Reconstruction/Epoch_{epoch + 1}', image)
    buf.close()
    plt.close(fig)


def evaluate_reconstruction(model, data):
    data = torch.FloatTensor(data).to(device)
    model.eval()

    with torch.no_grad():
        _, decoded = model(data)
        reconstruction_error = nn.MSELoss()(decoded, data)
    return reconstruction_error.item()


def calculate_autoencoder(encoding_dim, array_curves, epochs, log_dir):
    data = torch.FloatTensor(array_curves).to(device)
    dropout_rate = 0.3
    input_dim = array_curves.shape[1]
    model = AutoencoderDm(input_dim, encoding_dim, dropout_rate=dropout_rate).to(device)
    train_autoencoder_with_tensorboard(model, data, epochs, log_dir, batch_size=32, learning_rate=1e-3)
    reconstruction_error = evaluate_reconstruction(model, array_curves)
    print(f'Reconstruction Error: {reconstruction_error:.4f}')

    model.eval()
    with torch.no_grad():
        encoded, _ = model(data)

    return encoded.cpu().numpy(), model


def reconstruct_traces_autoencoder(encoded_data, model):
    """
    Reconstructs the traces using the decoder part of the autoencoder.

    Args:
        encoded_data (numpy.ndarray): The encoded data (latent representation).
        model (nn.Module): The trained autoencoder model.

    Returns:
        numpy.ndarray: The reconstructed input data.
    """
    encoded_tensor = torch.FloatTensor(encoded_data).to(device)
    model.eval()
    with torch.no_grad():
        reconstructed = model.decoder(encoded_tensor)
    return reconstructed.cpu().numpy()

