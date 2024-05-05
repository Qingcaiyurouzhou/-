import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from astropy.io import fits
import numpy as np

# 数据集类
class SpectraDataset(Dataset):
    def __init__(self, file_list):
        self.fluxes = []
        self.labels = []
        self.objids = []
        for file_name in file_list:
            with fits.open(file_name) as hdulist:
                self.fluxes.append(hdulist[0].data)
                self.labels.append(hdulist[1].data['label'])
                self.objids.append(hdulist[1].data['objid'])
        self.fluxes = np.vstack(self.fluxes)
        self.labels = np.concatenate(self.labels)
        self.objids = np.concatenate(self.objids)

    def __len__(self):
        return len(self.fluxes)

    def __getitem__(self, idx):
        flux = torch.tensor(self.fluxes[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return flux, label

# LSTM神经网络
class LSTMNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        x = x.unsqueeze(1) if x.dim() == 2 else x
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 训练函数
def train(model, dataloader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for i, (flux, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(flux)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

# 评估函数
def evaluate(model, dataloader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for flux, labels in dataloader:
            outputs = model(flux)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test set: {accuracy:.2f}%')

# 主程序
def main():
    file_list = [f'train_data_0{i}.fits' for i in range(1, 10)] + ['train_data_10.fits']
    dataset = SpectraDataset(file_list)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    model = LSTMNN(input_size=3000, hidden_size=128, num_layers=2, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, train_loader, criterion, optimizer, epochs=5)
    evaluate(model, test_loader)

if __name__ == '__main__':
    main()
