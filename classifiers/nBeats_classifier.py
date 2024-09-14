import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator

class NBEATSBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NBEATSBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# class NBEATSClassifier(nn.Module):
#     def __init__(self, input_size, hidden_size, num_blocks, num_classes):
#         super(NBEATSClassifier, self).__init__()
#         self.blocks = nn.ModuleList([NBEATSBlock(input_size, hidden_size) for _ in range(num_blocks)])
#         self.classifier = nn.Linear(input_size, num_classes)
#
#     def forward(self, x):
#         for block in self.blocks:
#             x = block(x)
#         x = self.classifier(x)
#         return x


class NBEATSClassifier(nn.Module, BaseEstimator):
    def __init__(self, input_size, hidden_size, num_blocks, num_classes):
        super(NBEATSClassifier, self).__init__()
        self.blocks = nn.ModuleList([NBEATSBlock(input_size, hidden_size) for _ in range(num_blocks)])
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.classifier(x)
        return x

    def fit(self, X, y, epochs=10, batch_size=32, learning_rate=0.001):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        data_loader = DataLoader(dataset=YourDataset(X, y), batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.train()
            for inputs, labels in data_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            outputs = self(X)
            _, predicted = torch.max(outputs, 1)
        return predicted
