from torch import nn

class AutoencoderClassifier(nn.Module):
    def __init__(self, input_dim, encoding_dim, num_classes):
        super(AutoencoderClassifier, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.Tanh(),
            nn.Linear(100, 50),
            nn.Tanh(),
            nn.Linear(50, 25),
            nn.Tanh(),
            nn.Linear(25, 12),
            nn.Tanh(),
            nn.Linear(12, 6),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(6, 12),
            nn.Tanh(),
            nn.Linear(12, 25),
            nn.Tanh(),
            nn.Linear(25, 50),
            nn.Tanh(),
            nn.Linear(50, 100),
            nn.Tanh(),
            nn.Linear(100, input_dim),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(6, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        reconstructed_x = self.decoder(x)
        class_probs = self.classifier(x)
        return reconstructed_x, class_probs