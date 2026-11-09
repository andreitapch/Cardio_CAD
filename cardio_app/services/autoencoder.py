import numpy as np
import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout=0.1, channels=(16, 32, 64)):
        super().__init__()
        self.input_dim, self.channels = input_dim, channels
        layers, previous = [], 1
        for width in channels:
            layers += [nn.Conv1d(previous, width, 5, stride=2, padding=2), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(dropout)]
            previous = width
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            self.encoded_len = self.conv(torch.zeros(1, 1, input_dim)).shape[-1]
        flat = channels[-1] * self.encoded_len
        self.to_latent, self.from_latent = nn.Linear(flat, latent_dim), nn.Linear(latent_dim, flat)
        layers, reversed_channels = [], list(channels[::-1])
        for i, width in enumerate(reversed_channels[1:] + [1]):
            layers += [nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
                       nn.Conv1d(reversed_channels[i], width, 5, padding=2)]
            if width != 1:
                layers += [nn.BatchNorm1d(width), nn.ReLU()]
        self.deconv = nn.Sequential(*layers)

    def encode(self, x):
        return self.to_latent(self.conv(x.unsqueeze(1)).flatten(1))

    def decode(self, z):
        out = self.deconv(self.from_latent(z).view(-1, self.channels[-1], self.encoded_len))
        if out.shape[-1] != self.input_dim:
            out = nn.functional.interpolate(out, size=self.input_dim, mode="linear", align_corners=False)
        return out.squeeze(1)

    def forward(self, x):
        z = self.encode(x)
        return z, self.decode(z)


def batches(n, size):
    bounds = [(i, min(i + size, n)) for i in range(0, n, size)]
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] == 1:
        bounds[-2] = (bounds[-2][0], bounds[-1][1])
        bounds.pop()
    return bounds


def tensor(x):
    return torch.FloatTensor(np.asarray(x, dtype=np.float32)).to(DEVICE)


def mse(model, data):
    model.eval()
    with torch.no_grad():
        _, decoded = model(data)
        return nn.MSELoss()(decoded, data).item()


def r2(model, data):
    total = float(((data - data.mean(dim=0)) ** 2).mean().item())
    return float("nan") if total == 0 else 1.0 - mse(model, data) / total


def split_indices(n, val_fraction=0.15, seed=42):
    n_val = int(round(n * val_fraction))
    if n_val < 1 or n - n_val < 2:
        return None
    order = torch.randperm(n, generator=torch.Generator(device="cpu").manual_seed(seed))
    return order[:n_val].numpy(), order[n_val:].numpy()


def train(model, data, val_data, epochs, batch_size=32, learning_rate=1e-3, patience=150, seed=42, progress=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=max(patience // 4, 5))
    criterion = nn.MSELoss()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    monitored_data = val_data if val_data is not None else data
    best, best_state, idle, history = float("inf"), None, 0, []
    for epoch in range(epochs):
        model.train()
        shuffled = data[torch.randperm(len(data), generator=generator)]
        total = 0.0
        for start, end in batches(len(shuffled), batch_size):
            batch = shuffled[start:end]
            optimizer.zero_grad()
            loss = criterion(model(batch)[1], batch)
            loss.backward()
            optimizer.step()
            total += loss.item() * (end - start)
        monitored = mse(model, monitored_data)
        history.append((total / len(shuffled), monitored))
        scheduler.step(monitored)
        if monitored < best - 1e-6:
            best, idle = monitored, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            idle += 1
        if progress:
            progress(epoch + 1)
        if idle >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return np.array(history)


def fit(x, latent_dim, epochs, seed=42, progress=None):
    data = tensor(x)
    model = ConvAutoencoder(x.shape[1], latent_dim).to(DEVICE)
    split = split_indices(len(data), seed=seed)
    train_data, val_data = (data, None) if split is None else (data[split[1]], data[split[0]])
    history = train(model, train_data, val_data, epochs, seed=seed, progress=progress)
    return model, history


def encode(model, x):
    model.eval()
    with torch.no_grad():
        return model.encode(tensor(x)).cpu().numpy()


def decode(model, codes):
    model.eval()
    with torch.no_grad():
        return model.decode(tensor(codes)).cpu().numpy()
