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


class ConvAutoencoderDm(nn.Module):
    """1D convolutional autoencoder for strain curves.

    The dense AutoencoderDm treats the 128 time points as unrelated features, so
    it has to learn from scratch that samples 40 and 41 are adjacent - which at
    n=408 studies costs 151k parameters on one segment, and 1.27M on concatenated
    traces, where it diverged to a negative R2. Convolutions share one small
    kernel across time instead, encoding local curve shape once, so the same job
    needs roughly an order of magnitude fewer weights.

    The interface matches AutoencoderDm exactly - 2D (n_samples, n_features) in
    and out, `.encoder` / `.decoder` callable on their own, forward returning
    (encoded, decoded) - so the trainer, the metric helpers and
    reconstruct_traces_autoencoder work unchanged.
    """

    def __init__(self, input_dim, encoding_dim, dropout_rate=0.1, channels=(16, 32, 64)):
        super(ConvAutoencoderDm, self).__init__()
        self.input_dim = input_dim

        conv, in_ch = [], 1
        for out_ch in channels:
            conv += [nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=2, padding=2),
                     nn.BatchNorm1d(out_ch), nn.ReLU(), nn.Dropout(dropout_rate)]
            in_ch = out_ch
        self.conv = nn.Sequential(*conv)

        # Halving is only exact for even lengths, so the encoded length is
        # measured rather than assumed - this has to hold for 128-sample segments
        # and 2304-sample concatenated traces alike.
        with torch.no_grad():
            self.encoded_len = self.conv(torch.zeros(1, 1, input_dim)).shape[-1]
        flat = channels[-1] * self.encoded_len

        self.to_latent = nn.Linear(flat, encoding_dim)
        self.from_latent = nn.Linear(encoding_dim, flat)

        deconv, rev = [], list(channels[::-1])
        for i, out_ch in enumerate(rev[1:] + [1]):
            deconv += [nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
                       nn.Conv1d(rev[i], out_ch, kernel_size=5, padding=2)]
            if out_ch != 1:
                deconv += [nn.BatchNorm1d(out_ch), nn.ReLU()]
        self.deconv = nn.Sequential(*deconv)
        self.channels = channels

    def encoder(self, x):
        h = self.conv(x.unsqueeze(1))
        return self.to_latent(h.flatten(1))

    def decoder(self, z):
        h = self.from_latent(z).view(-1, self.channels[-1], self.encoded_len)
        out = self.deconv(h)
        # Upsampling returns a power-of-two length; trim or stretch to the exact
        # input length so the output always lines up with the original trace.
        if out.shape[-1] != self.input_dim:
            out = nn.functional.interpolate(out, size=self.input_dim,
                                            mode='linear', align_corners=False)
        return out.squeeze(1)

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)


def batch_bounds(n_samples, batch_size):
    """Slice points for one epoch.

    BatchNorm1d raises on a batch of one while training, so a lone trailing
    sample is folded into the previous batch rather than dropped.
    """
    bounds = [(i, min(i + batch_size, n_samples)) for i in range(0, n_samples, batch_size)]
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] == 1:
        bounds[-2] = (bounds[-2][0], bounds[-1][1])
        bounds.pop()
    return bounds


def mean_squared_error(model, data):
    """Reconstruction MSE in eval mode, without touching the training state."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        _, decoded = model(data.to(device))
        error = nn.MSELoss()(decoded, data.to(device)).item()
    if was_training:
        model.train()
    return error


def explained_variance(model, data):
    """Fraction of variance the reconstruction accounts for: 1 - MSE / Var(X).

    Puts the autoencoder on the same axis as PCA's cumulative explained
    variance, so the two can be compared at equal latent dimension.
    """
    total_variance = float(((data - data.mean(dim=0)) ** 2).mean().item())
    if total_variance == 0:
        return float('nan')
    return 1.0 - mean_squared_error(model, data) / total_variance


def train_autoencoder_with_tensorboard(model, data, epochs, log_dir, batch_size=64, learning_rate=1e-4,
                                       val_data=None, patience=150, seed=42):
    """Train, tracking a validation split and keeping the best weights.

    Batches are reshuffled every epoch: with fixed slices the network sees the
    same partition 1000 times, and BatchNorm's running statistics track that
    partition rather than the data - which is what made the eval-mode error
    worse than the training loss.
    """
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)

    if len(data) < 2:
        raise ValueError(
            f'Need at least 2 studies to train the autoencoder, got {len(data)}: '
            'BatchNorm1d has no batch statistics to compute from a single sample.')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                     patience=max(patience // 4, 5))
    writer = SummaryWriter(log_dir)
    generator = torch.Generator(device='cpu').manual_seed(seed)

    best_loss, best_state, best_epoch, epochs_without_gain = float('inf'), None, 0, 0

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(data), generator=generator)
        shuffled = data[order]

        epoch_loss, n_batches = 0, 0
        for start, end in batch_bounds(len(shuffled), batch_size):
            batch = shuffled[start:end].to(device)
            optimizer.zero_grad()
            _, decoded = model(batch)
            loss = criterion(decoded, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Divide by the batches actually run: len(data) // batch_size is 0 for a
        # cohort smaller than one batch, and undercounts a partial trailing batch.
        avg_loss = epoch_loss / max(n_batches, 1)
        writer.add_scalar('Loss/Train', avg_loss, epoch)

        # Selection loss is measured in eval mode, the same mode the encoding is
        # later produced in, so the weights kept are the ones that encode best.
        monitored = mean_squared_error(model, val_data if val_data is not None else data)
        writer.add_scalar('Loss/Validation' if val_data is not None else 'Loss/TrainEval',
                          monitored, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        scheduler.step(monitored)

        if monitored < best_loss - 1e-6:
            best_loss, best_epoch, epochs_without_gain = monitored, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_gain += 1

        if (epoch + 1) % 50 == 0 and os.environ.get('CARDIO_AE_PLOTS') == '1':
            log_reconstruction(writer, model, data, epoch)

        if (epoch + 1) % 100 == 0:
            label = 'val' if val_data is not None else 'eval'
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, {label}: {monitored:.4f}')

        if epochs_without_gain >= patience:
            print(f'Early stop at epoch {epoch + 1}: no improvement for {patience} epochs.')
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f'Restored best weights from epoch {best_epoch + 1} ({best_loss:.4f}).')

    writer.close()
    return {'best_loss': best_loss, 'best_epoch': best_epoch}


def log_reconstruction(writer, model, data, epoch):
    """Log original-vs-reconstructed curves to TensorBoard as an image.

    Opt-in via CARDIO_AE_PLOTS=1: matplotlib's savefig pulls conda's libomp.dll,
    which aborts the process ("OMP: Error #15") once torch's libiomp5md.dll is
    already initialised. Safe from a standalone script, fatal inside the server.
    """
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


def calculate_autoencoder(encoding_dim, array_curves, epochs, log_dir, val_fraction=0.15,
                          patience=150, seed=42, kind='conv'):
    """Fit the autoencoder and return the latent codes for every study.

    A held-out split decides which weights to keep: without it the run simply
    returns whatever epoch 1000 happened to land on, which with an oscillating
    loss is not the best model it saw.

    kind='conv' (the default) uses the convolutional encoder; 'mlp' selects the
    original dense one. The dense network has no notion of time ordering and at
    this sample size it overfits badly - on concatenated traces it reached a
    held-out R2 of -2.8, worse than predicting the cohort mean - so it is kept
    only for comparison.
    """
    data = torch.FloatTensor(array_curves).to(device)
    input_dim = array_curves.shape[1]
    if kind == 'conv':
        # Lighter dropout: the convolutional model has far fewer weights to
        # regularise, and 0.3 on top of that underfits.
        model = ConvAutoencoderDm(input_dim, encoding_dim, dropout_rate=0.1).to(device)
    elif kind == 'mlp':
        model = AutoencoderDm(input_dim, encoding_dim, dropout_rate=0.3).to(device)
    else:
        raise ValueError(f"Unknown autoencoder kind: {kind!r} (expected 'conv' or 'mlp')")

    train_data, val_data = split_train_val(data, val_fraction, seed)
    if val_data is None:
        print('[INFO] cohort too small for a validation split; selecting on training data.')

    train_autoencoder_with_tensorboard(model, train_data, epochs, log_dir, batch_size=32,
                                       learning_rate=1e-3, val_data=val_data,
                                       patience=patience, seed=seed)

    # R2 alongside MSE so this is directly comparable to PCA's cumulative
    # explained variance at the same latent dimension.
    print(f'Reconstruction Error: {mean_squared_error(model, data):.4f} '
          f'(R2 {explained_variance(model, data):.4f}, {encoding_dim} components)')
    if val_data is not None:
        print(f'  held-out: MSE {mean_squared_error(model, val_data):.4f}, '
              f'R2 {explained_variance(model, val_data):.4f}')

    model.eval()
    with torch.no_grad():
        encoded, _ = model(data)

    return encoded.cpu().numpy(), model


def split_train_val(data, val_fraction, seed):
    """Hold out a slice for model selection, or None when there is too little data.

    BatchNorm1d needs at least 2 training samples, so a split that would leave
    fewer is refused rather than silently breaking the run.
    """
    n_samples = len(data)
    n_val = int(round(n_samples * val_fraction))
    if val_fraction <= 0 or n_val < 1 or n_samples - n_val < 2:
        return data, None

    order = torch.randperm(n_samples, generator=torch.Generator(device='cpu').manual_seed(seed))
    val_idx, train_idx = order[:n_val], order[n_val:]
    return data[train_idx], data[val_idx]


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

