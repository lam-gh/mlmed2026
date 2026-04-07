import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.utils import resample
from torch import nn
from torch.utils.data import DataLoader, Dataset


def balance_data(df):
    df_0 = df[df.iloc[:, -1] == 0]
    df_1 = df[df.iloc[:, -1] == 1]
    df_2 = df[df.iloc[:, -1] == 2]
    df_3 = df[df.iloc[:, -1] == 3]
    df_4 = df[df.iloc[:, -1] == 4]

    n_samples = 20000
    df_0_resample = resample(df_0, replace=True, n_samples=n_samples, random_state=123)
    df_1_resample = resample(df_1, replace=True, n_samples=n_samples, random_state=123)
    df_2_resample = resample(df_2, replace=True, n_samples=n_samples, random_state=123)
    df_3_resample = resample(df_3, replace=True, n_samples=n_samples, random_state=123)
    df_4_resample = resample(df_4, replace=True, n_samples=n_samples, random_state=123)

    return pd.concat([df_0_resample, df_1_resample, df_2_resample, df_3_resample, df_4_resample])


class ECGDataset(Dataset):
    def __init__(self, df):
        self.data = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=5, stride=2)

    def forward(self, x):
        res = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x += res
        x = F.relu(x)
        return self.pool(x)


class ECGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.blocks = nn.Sequential(*[ResidualBlock(32) for _ in range(5)])

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 32)
        self.classifier = nn.Linear(32, 5)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.classifier(x)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for signals, labels in loader:
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / len(loader.dataset)
    return total_loss / len(loader), accuracy


def main():
    train = pd.read_csv("data/prac1/mitbih_train.csv")
    test = pd.read_csv("data/prac1/mitbih_test.csv")

    train_balanced = balance_data(train)

    train_loader = DataLoader(ECGDataset(train_balanced), batch_size=32, shuffle=True)
    test_loader = DataLoader(ECGDataset(test), batch_size=32, shuffle=False)

    model = ECGNet()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.75)
    criterion = nn.CrossEntropyLoss()

    device = torch.device("mps")
    model.to(device)

    epochs = 50
    best_acc = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (signals, labels) in enumerate(train_loader):
            signals, labels = signals.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (epoch * len(train_loader) + i) % 10000 == 0:
                scheduler.step()

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch + 1}/{epochs} - Loss: {running_loss / len(train_loader):.4f} - Test Acc: {test_acc:.2f}% - Test loss: {test_loss:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_model.pth")

    model.load_state_dict(torch.load("best_model.pth"))
    _, final_acc = evaluate(model, test_loader, criterion, device)
    print(f"Final accuracy: {final_acc:.2f}%")


if __name__ == "__main__":
    main()
