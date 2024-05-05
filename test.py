import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from astropy.io import fits
import numpy as np

# 数据集类
class SpectraDataset(Dataset):
    # 初始化数据集
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

    # 返回数据集大小
    def __len__(self):
        return len(self.fluxes)

    # 获取数据和标签
    def __getitem__(self, idx):
        flux = torch.tensor(self.fluxes[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return flux, label

# 简单的全连接神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3000, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 输出层有3个神经元，对应3种类型的光谱
        )

    def forward(self, x):
        return self.fc(x)

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
    # 创建数据集
    file_list = [f'train_data_0{i}.fits' for i in range(1, 10)] + ['train_data_10.fits']
    dataset = SpectraDataset(file_list)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 定义模型、损失函数和优化器
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train(model, dataloader, criterion, optimizer, epochs=5)

    # 评估模型
    evaluate(model, dataloader)

if __name__ == '__main__':
    main()
